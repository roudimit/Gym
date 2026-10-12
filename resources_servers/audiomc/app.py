# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AudioMC: rubric-based LLM-judge scoring of multi-turn spoken conversations.

Each task has rubric items. The judge decides each item independently from the
transcript history plus the candidate's final answer. A task passes when every
item is met (APR, average pass rate); ARS is the fraction of all items met.
"""

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion
from nemo_gym.reward_profile import compute_aggregate_metrics
from resources_servers.audiomc.scoring import candidate_prediction, conversation_history, parse_judgment
from resources_servers.audiomc.task_data import AXES, AudioMCMetadata, Axis


MetricValue = int | float | bool | None
_PLACEHOLDERS = ("{conversation_history}", "{rubric_item}")
EMPTY_CANDIDATE_RAW = "AUTO_EMPTY_CANDIDATE"


class JudgePrompt(BaseModel):
    """Versioned judge messages; ``{conversation_history}`` and ``{rubric_item}`` are filled with str.format."""

    model_config = ConfigDict(extra="forbid")
    system: str | None = None
    user: str


class AudioMCConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    scoring_version: Literal["legacy_v1"] = "legacy_v1"
    # None scores every axis; set one to run a single-axis instance.
    axis: Axis | None = None
    expected_num_rows: int = Field(gt=0)
    expected_sample_ids_file: str | None = None
    judge_model_server: ModelServerRef
    judge_prompt_file: str
    # Reference profile: temperature 0.0, top_p unset, 4096 tokens, 3 attempts.
    judge_temperature: float | None = 0.0
    judge_top_p: float | None = None
    judge_max_tokens: int = Field(default=4096, gt=0)
    judge_attempts: int = Field(default=3, gt=0)
    judge_max_concurrency: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def require_prompt_file(self) -> "AudioMCConfig":
        if not Path(self.judge_prompt_file).is_file():
            raise ValueError(f"judge_prompt_file does not exist: {self.judge_prompt_file}")
        return self


class AudioMCVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: AudioMCMetadata


class AudioMCVerifyResponse(AudioMCVerifyRequest, BaseVerifyResponse):
    sample_id: str
    axis: Axis
    scoring_version: str
    candidate_answer: str
    candidate_status: str | None
    candidate_valid: bool
    candidate_thinking: list[str]
    candidate_incomplete_details: dict[str, str] | None
    rubrics_total: int
    rubrics_met: int
    task_pass: bool
    judge_errors: int
    empty_candidate: bool
    judge_prompt_sha256: str
    rubric_judgments: list[dict[str, Any]]


def _expected_ids(path: str) -> set[str]:
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        values = json.loads(text)
    else:
        values = []
        for line in text.split("\n"):
            if not line.strip():
                continue
            row = json.loads(line) if line.lstrip().startswith("{") else line.strip()
            if isinstance(row, dict):
                row = row.get("verifier_metadata", row)["id"]
            values.append(row)
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("expected_sample_ids_file requires a nonempty string list or Gym JSONL rows")
    if len(set(values)) != len(values):
        raise ValueError("expected_sample_ids_file contains duplicate IDs")
    return set(values)


class AudioMCServer(SimpleResourcesServer):
    config: AudioMCConfig
    _prompt: JudgePrompt = PrivateAttr()
    _prompt_sha256: str = PrivateAttr()
    _slots: asyncio.Semaphore = PrivateAttr()
    _expected_sample_ids: set[str] | None = PrivateAttr(default=None)

    def model_post_init(self, context: object) -> None:
        prompt_bytes = Path(self.config.judge_prompt_file).read_bytes()
        self._prompt = JudgePrompt.model_validate(yaml.safe_load(prompt_bytes))
        combined = (self._prompt.system or "") + self._prompt.user
        missing = [placeholder for placeholder in _PLACEHOLDERS if placeholder not in combined]
        if missing:
            raise ValueError(f"judge prompt must contain {', '.join(missing)}")
        self._prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        self._slots = asyncio.Semaphore(self.config.judge_max_concurrency)
        if self.config.expected_sample_ids_file:
            self._expected_sample_ids = _expected_ids(self.config.expected_sample_ids_file)
            if len(self._expected_sample_ids) != self.config.expected_num_rows:
                raise ValueError("expected_sample_ids_file count differs from expected_num_rows")
        super().model_post_init(context)

    def _messages(self, history: str, rubric_item: str) -> list[dict[str, str]]:
        values = {"conversation_history": history, "rubric_item": rubric_item}
        messages = []
        if self._prompt.system is not None:
            messages.append({"role": "system", "content": self._prompt.system.format(**values)})
        messages.append({"role": "user", "content": self._prompt.user.format(**values)})
        return messages

    async def _judge_one(self, index: int, rubric_item: str, history: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": self._messages(history, rubric_item),
            "max_tokens": self.config.judge_max_tokens,
            "stream": False,
        }
        if self.config.judge_temperature is not None:
            body["temperature"] = self.config.judge_temperature
        if self.config.judge_top_p is not None:
            body["top_p"] = self.config.judge_top_p
        result: dict[str, Any] = {"rubric_index": index, "rubric_item": rubric_item}
        calls: list[dict[str, Any]] = []
        started = perf_counter()
        for attempt in range(1, self.config.judge_attempts + 1):
            raw = ""
            call: dict[str, Any] = {"attempt": attempt, "finish_reason": None, "usage": None, "error": None}
            calls.append(call)
            try:
                async with self._slots:
                    judged = await call_judge(
                        self.server_client,
                        server_name=self.config.judge_model_server.name,
                        url_path="/v1/chat/completions",
                        json=body,
                        response_model=NeMoGymChatCompletion,
                    )
                call["usage"] = judged.usage.model_dump(mode="json") if judged.usage is not None else None
                call["finish_reason"] = judged.choices[0].finish_reason if judged.choices else None
                raw = judged.choices[0].message.content or ""
                explanation, met = parse_judgment(raw)
            except (JudgeError, ValueError, IndexError) as error:
                call["error"] = f"{type(error).__name__}: {error}"
                # Like the reference, an item still failing after every attempt is not met.
                result.update(
                    judge_raw=raw,
                    explanation="",
                    criteria_met=False,
                    judge_error=f"{type(error).__name__}: {error}",
                    judge_attempts=attempt,
                )
                continue
            result.update(
                judge_raw=raw,
                explanation=explanation,
                criteria_met=met,
                judge_error=None,
                judge_attempts=attempt,
                judge_model=judged.model,
            )
            break
        # The final call's finish reason and usage; every attempt is kept in judge_calls.
        result["judge_finish_reason"] = calls[-1]["finish_reason"]
        result["judge_usage"] = calls[-1]["usage"]
        result["judge_calls"] = calls
        result["latency_s"] = round(perf_counter() - started, 3)
        return result

    async def verify(self, body: AudioMCVerifyRequest) -> AudioMCVerifyResponse:
        metadata = body.verifier_metadata
        if self.config.axis is not None and metadata.axis != self.config.axis:
            raise ValueError(f"resource for axis {self.config.axis} cannot score axis {metadata.axis}")
        prediction = candidate_prediction(body.response)
        empty = not prediction.strip()
        if empty:
            judgments = [
                {
                    "rubric_index": index,
                    "rubric_item": item,
                    "judge_raw": EMPTY_CANDIDATE_RAW,
                    "explanation": "The model response is empty.",
                    "criteria_met": False,
                    "judge_error": None,
                    "judge_attempts": 0,
                    "judge_finish_reason": None,
                    "judge_usage": None,
                    "judge_calls": [],
                }
                for index, item in enumerate(metadata.rubric)
            ]
        else:
            history = conversation_history(metadata.turns, prediction)
            judgments = list(
                await asyncio.gather(
                    *(self._judge_one(index, str(item), history) for index, item in enumerate(metadata.rubric))
                )
            )
        met = sum(row["criteria_met"] is True for row in judgments)
        errors = sum(row["judge_error"] is not None for row in judgments)
        task_pass = all(row["criteria_met"] is True and row["judge_error"] is None for row in judgments)
        return AudioMCVerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": float(task_pass),
                "sample_id": metadata.id,
                "axis": metadata.axis,
                "scoring_version": self.config.scoring_version,
                "candidate_answer": prediction,
                "candidate_status": body.response.status,
                "candidate_valid": body.response.status == "completed" and body.response.incomplete_details is None,
                "candidate_thinking": [
                    part.text
                    for item in body.response.output
                    if item.type == "reasoning"
                    for part in (item.summary or [])
                ],
                "candidate_incomplete_details": body.response.incomplete_details.model_dump()
                if body.response.incomplete_details is not None
                else None,
                "rubrics_total": len(judgments),
                "rubrics_met": met,
                "task_pass": task_pass,
                "judge_errors": errors,
                "empty_candidate": empty,
                "judge_prompt_sha256": self._prompt_sha256,
                "rubric_judgments": judgments,
            }
        )

    def compute_metrics(self, tasks: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, MetricValue]:
        """APR and ARS overall and per axis, plus coverage and judge diagnostics.

        Rows whose verification was masked stay in the coverage counts but cannot
        contribute a pass, so failures never inflate the score.
        """
        rows = [row for task in tasks for row in task]
        identities = [row.get("sample_id") if isinstance(row.get("sample_id"), str) else None for row in rows]
        observed = {identity for identity in identities if identity is not None}
        scored = [row for row in rows if not row.get("mask_sample") and isinstance(row.get("rubrics_total"), int)]

        def summarize(selected: list[Mapping[str, object]], prefix: str) -> dict[str, MetricValue]:
            rubrics_total = sum(int(row["rubrics_total"]) for row in selected)
            rubrics_met = sum(int(row["rubrics_met"]) for row in selected)
            tasks_pass = sum(row.get("task_pass") is True for row in selected)
            return {
                f"{prefix}/tasks_total": len(selected),
                f"{prefix}/tasks_pass": tasks_pass,
                f"{prefix}/apr": tasks_pass / len(selected) if selected else 0.0,
                f"{prefix}/rubrics_total": rubrics_total,
                f"{prefix}/rubrics_met": rubrics_met,
                f"{prefix}/ars": rubrics_met / rubrics_total if rubrics_total else 0.0,
            }

        metrics = summarize(scored, "audiomc")
        for axis in AXES:
            axis_rows = [row for row in scored if row.get("axis") == axis]
            if axis_rows:
                metrics.update(summarize(axis_rows, f"audiomc/{axis}"))
        duplicates = len(identities) - len(observed) - identities.count(None)
        expected = self._expected_sample_ids
        missing = (
            len(expected - observed) if expected is not None else max(0, self.config.expected_num_rows - len(observed))
        )
        unexpected = len(observed - expected) if expected is not None else 0
        judge_errors = sum(int(row.get("judge_errors") or 0) for row in scored)
        calls = [
            call
            for row in scored
            for judgment in row.get("rubric_judgments") or []
            for call in judgment.get("judge_calls") or []
        ]

        def usage_total(*path: str) -> int:
            total = 0
            for call in calls:
                value: Any = call.get("usage")
                for key in path:
                    value = value.get(key) if isinstance(value, Mapping) else None
                if isinstance(value, int) and not isinstance(value, bool):
                    total += value
            return total

        candidate_invalid = sum(row.get("candidate_valid") is not True for row in rows)
        sample_complete = (
            len(rows) == len(observed) == self.config.expected_num_rows
            and len(scored) == len(rows)
            and not duplicates
            and not missing
            and not unexpected
        )
        metrics.update(
            {
                "audiomc/count": len(rows),
                "audiomc/expected_rows": self.config.expected_num_rows,
                "audiomc/masked_rows": len(rows) - len(scored),
                "audiomc/duplicate_rows": duplicates,
                "audiomc/missing_rows": missing,
                "audiomc/unexpected_sample_ids": unexpected,
                "audiomc/judge_errors": judge_errors,
                "audiomc/judge_calls": len(calls),
                "audiomc/judge_finish_length": sum(
                    judgment.get("judge_finish_reason") == "length"
                    for row in scored
                    for judgment in row.get("rubric_judgments") or []
                ),
                "audiomc/judge_prompt_tokens": usage_total("prompt_tokens"),
                "audiomc/judge_completion_tokens": usage_total("completion_tokens"),
                "audiomc/judge_reasoning_tokens": usage_total("completion_tokens_details", "reasoning_tokens"),
                "audiomc/empty_candidates": sum(row.get("empty_candidate") is True for row in scored),
                "audiomc/candidate_invalid_rows": candidate_invalid,
                "audiomc/sample_complete": sample_complete,
                "audiomc/complete": sample_complete and not judge_errors,
            }
        )
        return metrics

    def get_key_metrics(self, agent_metrics: Mapping[str, object]) -> dict[str, object]:
        return {key: value for key, value in agent_metrics.items() if key.startswith("audiomc/")}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        aggregate = compute_aggregate_metrics(body.verify_responses)
        metrics = self.compute_metrics([body.verify_responses])
        aggregate.agent_metrics.update(metrics)
        aggregate.key_metrics.update(metrics)
        return aggregate


if __name__ == "__main__":
    AudioMCServer.run_webserver()
