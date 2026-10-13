# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BFCL v3: AST-checked single-turn function calling.

The candidate answers with ``<tool_call>`` blocks in the chat template's XML
syntax. The calls are parsed from the final answer and checked against the gold
calls with Gorilla's BFCL v3 AST checker. Irrelevance rows pass when the
candidate makes no call.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.reward_profile import compute_aggregate_metrics
from resources_servers.bfclv3.scoring import candidate_prediction, names_match, parse_prediction, score_row
from resources_servers.bfclv3.task_data import AST_CATEGORIES, CATEGORIES, BFCLv3Metadata, Category


MetricValue = int | float | bool | None


class BFCLv3Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    # "template": the chat template's <tool_call><function=...><parameter=...> syntax.
    prompt_style: Literal["template"] = "template"
    expected_num_rows: int = Field(gt=0)
    expected_sample_ids_file: str | None = None


class BFCLv3VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: BFCLv3Metadata


class BFCLv3VerifyResponse(BFCLv3VerifyRequest, BaseVerifyResponse):
    sample_id: str
    category: Category
    prompt_style: str
    candidate_answer: str
    candidate_status: str | None
    candidate_valid: bool
    candidate_incomplete_details: dict[str, str] | None
    parse_status: str
    predicted_calls: list[dict[str, Any]]
    num_pred_calls: int
    num_gold_calls: int
    name_correct: bool
    bfcl_valid: bool
    error_type: str


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


class BFCLv3Server(SimpleResourcesServer):
    ray_enabled = False
    config: BFCLv3Config
    _expected_sample_ids: set[str] | None = PrivateAttr(default=None)

    def model_post_init(self, context: object) -> None:
        if self.config.expected_sample_ids_file:
            self._expected_sample_ids = _expected_ids(self.config.expected_sample_ids_file)
            if len(self._expected_sample_ids) != self.config.expected_num_rows:
                raise ValueError("expected_sample_ids_file count differs from expected_num_rows")
        super().model_post_init(context)

    async def verify(self, body: BFCLv3VerifyRequest) -> BFCLv3VerifyResponse:
        metadata = body.verifier_metadata
        prediction = candidate_prediction(body.response)
        calls, parse_status = parse_prediction(prediction)
        valid, error_type = score_row(calls, metadata.gold_toolcall, metadata.tools, metadata.category)
        response = body.response
        return BFCLv3VerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": float(valid),
                "sample_id": metadata.id,
                "category": metadata.category,
                "prompt_style": self.config.prompt_style,
                "candidate_answer": prediction,
                "candidate_status": response.status,
                "candidate_valid": response.status == "completed" and response.incomplete_details is None,
                "candidate_incomplete_details": response.incomplete_details.model_dump()
                if response.incomplete_details is not None
                else None,
                "parse_status": parse_status,
                "predicted_calls": calls,
                "num_pred_calls": len(calls),
                "num_gold_calls": len(metadata.gold_toolcall),
                "name_correct": names_match(calls, metadata.gold_toolcall),
                "bfcl_valid": valid,
                "error_type": error_type,
            }
        )

    def compute_metrics(self, tasks: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, MetricValue]:
        """AST accuracy, irrelevance accuracy, and per-category accuracy, plus coverage.

        ``bfclv3/ast_accuracy`` is the micro average over the non-irrelevance
        rows, as in the reference headline. Irrelevance is reported on its own
        with its false-positive rate (any predicted call). Masked rows count
        toward coverage but are not scored.
        """
        rows = [row for task in tasks for row in task]
        identities = [row.get("sample_id") if isinstance(row.get("sample_id"), str) else None for row in rows]
        observed = {identity for identity in identities if identity is not None}
        scored = [row for row in rows if not row.get("mask_sample") and isinstance(row.get("bfcl_valid"), bool)]

        def rate(selected: list[Mapping[str, object]], key: str) -> float:
            return sum(row.get(key) is True for row in selected) / len(selected) if selected else 0.0

        ast_rows = [row for row in scored if row.get("category") in AST_CATEGORIES]
        irrelevance = [row for row in scored if row.get("category") == "irrelevance"]
        metrics: dict[str, MetricValue] = {
            "bfclv3/ast_accuracy": rate(ast_rows, "bfcl_valid"),
            "bfclv3/ast_rows": len(ast_rows),
            "bfclv3/name_accuracy": rate(ast_rows, "name_correct"),
            "bfclv3/irrelevance_accuracy": rate(irrelevance, "bfcl_valid"),
            "bfclv3/irrelevance_false_positive_rate": sum(
                int(row.get("num_pred_calls") or 0) > 0 for row in irrelevance
            )
            / len(irrelevance)
            if irrelevance
            else 0.0,
            "bfclv3/irrelevance_rows": len(irrelevance),
            "bfclv3/accuracy": rate(scored, "bfcl_valid"),
        }
        for category in CATEGORIES:
            selected = [row for row in scored if row.get("category") == category]
            if selected:
                metrics[f"bfclv3/{category}/accuracy"] = rate(selected, "bfcl_valid")
                metrics[f"bfclv3/{category}/count"] = len(selected)

        duplicates = len(identities) - len(observed) - identities.count(None)
        expected = self._expected_sample_ids
        missing = (
            len(expected - observed) if expected is not None else max(0, self.config.expected_num_rows - len(observed))
        )
        unexpected = len(observed - expected) if expected is not None else 0
        sample_complete = (
            len(rows) == len(observed) == self.config.expected_num_rows
            and len(scored) == len(rows)
            and not duplicates
            and not missing
            and not unexpected
        )
        metrics.update(
            {
                "bfclv3/parsed_rows": sum(row.get("parse_status") == "xml" for row in scored),
                "bfclv3/no_toolcall_rows": sum(row.get("parse_status") == "none" for row in scored),
                "bfclv3/empty_predictions": sum(row.get("parse_status") == "empty" for row in scored),
                "bfclv3/checker_exceptions": sum(
                    str(row.get("error_type") or "").startswith("checker_exception") for row in scored
                ),
                "bfclv3/count": len(rows),
                "bfclv3/expected_rows": self.config.expected_num_rows,
                "bfclv3/masked_rows": len(rows) - len(scored),
                "bfclv3/duplicate_rows": duplicates,
                "bfclv3/missing_rows": missing,
                "bfclv3/unexpected_sample_ids": unexpected,
                "bfclv3/candidate_invalid_rows": sum(row.get("candidate_valid") is not True for row in rows),
                "bfclv3/complete": sample_complete,
            }
        )
        return metrics

    def get_key_metrics(self, agent_metrics: Mapping[str, object]) -> dict[str, object]:
        return {key: value for key, value in agent_metrics.items() if key.startswith("bfclv3/")}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        aggregate = compute_aggregate_metrics(body.verify_responses)
        metrics = self.compute_metrics([body.verify_responses])
        aggregate.agent_metrics.update(metrics)
        aggregate.key_metrics.update(metrics)
        return aggregate


if __name__ == "__main__":
    BFCLv3Server.run_webserver()
