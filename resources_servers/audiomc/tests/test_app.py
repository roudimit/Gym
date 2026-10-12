# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-compatible extraction, judging, and APR/ARS aggregation."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.audiomc.app import AudioMCConfig, AudioMCServer, AudioMCVerifyRequest
from resources_servers.audiomc.scoring import candidate_prediction, conversation_history, parse_judgment
from resources_servers.audiomc.task_data import AudioMCTurn


PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "legacy_rubric_v1.yaml"


def candidate(text="Final answer.", *, reasoning="Thinking.", status="completed"):
    output = []
    if reasoning is not None:
        output.append({"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    output.append(
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    )
    return NeMoGymResponse.model_validate(
        {
            "id": "candidate",
            "created_at": 0,
            "model": "audio-model",
            "object": "response",
            "status": status,
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": output,
        }
    )


TURNS = [
    {"turn": 1, "user_transcript": "Hi there.", "assistant_transcript": "Hello!"},
    {"turn": 2, "user_transcript": "  What did I say? ", "assistant_transcript": "ignored final"},
]


def request(sample_id="t1", *, axis="VOICE_EDITING", rubric=("A", "B"), response=None):
    return AudioMCVerifyRequest.model_validate(
        {
            "id": f"audiomc/{sample_id}",
            "responses_create_params": {"input": [{"role": "user", "content": "Please answer."}]},
            "verifier_metadata": {"id": sample_id, "axis": axis, "rubric": list(rubric), "turns": TURNS},
            "response": (response or candidate()).model_dump(),
        }
    )


def judge_reply(content, finish_reason="stop"):
    payload = {
        "id": "judge",
        "created": 0,
        "model": "judge-model",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "completion_tokens_details": {"reasoning_tokens": 30},
        },
    }
    response = MagicMock()
    response.status = 200
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    response.content.read = AsyncMock(return_value=json.dumps(payload).encode())
    return response


def server(replies=(), *, expected=1, **overrides):
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(
        side_effect=[judge_reply(*reply) if isinstance(reply, tuple) else judge_reply(reply) for reply in replies]
    )
    values = {
        "name": "audiomc",
        "host": "127.0.0.1",
        "port": 8160,
        "entrypoint": "app.py",
        "expected_num_rows": expected,
        "judge_model_server": {"type": "responses_api_models", "name": "judge_model"},
        "judge_prompt_file": str(PROMPT),
    }
    return AudioMCServer(config=AudioMCConfig(**(values | overrides)), server_client=client)


MET = '```json\n{"explanation": "ok", "criteria_met": true}\n```'
NOT_MET = '{"explanation": "no", "criteria_met": "false"}'


@pytest.mark.parametrize(
    ("text", "reasoning", "expected"),
    [
        ("\n\nAnswer.", "Think.", "Answer."),
        ("Answer.", None, ""),  # reasoning mode without </think> -> empty prediction
        ("x</think>\n\ny", None, "y"),
        ("Answer.", "", ""),  # empty reasoning is treated as none
    ],
)
def test_candidate_prediction_matches_reference_stripping(text, reasoning, expected):
    assert candidate_prediction(candidate(text, reasoning=reasoning)) == expected


def test_conversation_history_puts_prediction_in_final_turn():
    turns = [AudioMCTurn.model_validate(turn) for turn in TURNS]
    assert conversation_history(turns, "Mine.") == (
        "User: Hi there.\n\nAssistant: Hello!\n\nUser: What did I say?\n\nAssistant: Mine."
    )
    assert conversation_history([], "Mine.") == "Assistant: Mine."


@pytest.mark.parametrize(("raw", "met"), [(MET, True), (NOT_MET, False), ('noise {"criteria_met": "yes"} x', True)])
def test_parse_judgment(raw, met):
    assert parse_judgment(raw)[1] is met


@pytest.mark.parametrize("raw", ["", "not json", '{"criteria_met": "maybe"}', "[1]"])
def test_parse_judgment_rejects_invalid(raw):
    with pytest.raises(ValueError):
        parse_judgment(raw)


def test_verify_judges_each_rubric_with_reference_messages():
    svc = server([MET, MET])
    result = asyncio.run(svc.verify(request()))
    assert (result.reward, result.task_pass, result.rubrics_met, result.rubrics_total) == (1.0, True, 2, 2)
    body = svc.server_client.post.call_args_list[0].kwargs["json"]
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert (
        "# Rubric item\n\nA" in body["messages"][0]["content"]
        or "# Rubric item\n\nB" in body["messages"][0]["content"]
    )
    assert "Assistant: Final answer." in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "Return the JSON judgment now."
    assert (body["temperature"], body["max_tokens"], "top_p" in body) == (0.0, 4096, False)


def test_unparseable_judgment_is_retried_then_counts_as_not_met():
    svc = server(["garbage", "garbage", "garbage", MET], judge_max_concurrency=1)
    result = asyncio.run(svc.verify(request()))
    rows = sorted(result.rubric_judgments, key=lambda row: row["rubric_index"])
    assert rows[0]["judge_error"] and rows[0]["judge_attempts"] == 3 and rows[0]["criteria_met"] is False
    assert rows[1]["criteria_met"] is True
    assert (result.task_pass, result.judge_errors, result.reward) == (False, 1, 0.0)


def test_empty_candidate_fails_every_rubric_without_judge_calls():
    svc = server()
    result = asyncio.run(svc.verify(request(response=candidate("Answer.", reasoning=None))))
    assert result.empty_candidate and result.rubrics_met == 0 and not result.task_pass
    assert {row["judge_raw"] for row in result.rubric_judgments} == {"AUTO_EMPTY_CANDIDATE"}
    svc.server_client.post.assert_not_called()


def test_metrics_report_apr_and_micro_ars_per_axis():
    svc = server([MET, MET, MET, NOT_MET, MET], expected=3)
    rows = [
        asyncio.run(svc.verify(request("a", rubric=("x", "y")))).model_dump(),
        asyncio.run(svc.verify(request("b", rubric=("x", "y")))).model_dump(),
        asyncio.run(svc.verify(request("c", axis="SELF_COHERENCE", rubric=("x",)))).model_dump(),
    ]
    metrics = svc.compute_metrics([rows])
    assert metrics["audiomc/tasks_pass"] == 2
    assert metrics["audiomc/apr"] == pytest.approx(2 / 3)
    assert (metrics["audiomc/rubrics_met"], metrics["audiomc/rubrics_total"]) == (4, 5)
    assert metrics["audiomc/ars"] == pytest.approx(4 / 5)
    assert metrics["audiomc/VOICE_EDITING/apr"] == pytest.approx(1 / 2)
    assert metrics["audiomc/SELF_COHERENCE/ars"] == 1.0
    assert metrics["audiomc/complete"] is True


def test_axis_instance_rejects_other_axes():
    svc = server(axis="INFERENCE_MEMORY")
    with pytest.raises(ValueError, match="cannot score axis"):
        asyncio.run(svc.verify(request()))


def test_prompt_requires_placeholders(tmp_path):
    prompt = tmp_path / "judge.yaml"
    prompt.write_text("user: no placeholders\n")
    with pytest.raises(ValueError, match="conversation_history"):
        server(judge_prompt_file=str(prompt))


def test_judgments_record_finish_reason_usage_and_every_attempt():
    svc = server([("", "length"), MET, MET], judge_max_concurrency=1, expected=1)
    result = asyncio.run(svc.verify(request(rubric=("only",))))
    judgment = result.rubric_judgments[0]
    assert [call["finish_reason"] for call in judgment["judge_calls"]] == ["length", "stop"]
    assert judgment["judge_calls"][0]["error"] and judgment["judge_calls"][1]["error"] is None
    assert judgment["judge_finish_reason"] == "stop"
    assert judgment["judge_usage"]["completion_tokens_details"]["reasoning_tokens"] == 30
    metrics = svc.compute_metrics([[result.model_dump()]])
    assert metrics["audiomc/judge_calls"] == 2
    assert metrics["audiomc/judge_finish_length"] == 0
    assert (metrics["audiomc/judge_completion_tokens"], metrics["audiomc/judge_reasoning_tokens"]) == (80, 60)


def test_final_length_cut_is_counted():
    svc = server([('```json\n{"criteria_met": true}\n```', "length")], expected=1)
    result = asyncio.run(svc.verify(request(rubric=("only",))))
    assert svc.compute_metrics([[result.model_dump()]])["audiomc/judge_finish_length"] == 1
