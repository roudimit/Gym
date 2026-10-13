# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-compatible prediction extraction, tool-call parsing, AST scoring, and metrics."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.bfclv3.app import BFCLv3Config, BFCLv3Server, BFCLv3VerifyRequest
from resources_servers.bfclv3.scoring import candidate_prediction, parse_prediction, score_row


EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"

AREA_TOOL = {
    "name": "calculate_triangle_area",
    "description": "Area of a triangle.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer", "description": "Base."},
            "height": {"type": "integer", "description": "Height."},
            "unit": {"type": "string", "description": "Unit, default units."},
        },
        "required": ["base", "height"],
    },
}
WEATHER_TOOL = {
    "name": "weather.get",
    "description": "Weather for a city.",
    "parameters": {
        "type": "dict",
        "properties": {"city": {"type": "string", "description": "City."}},
        "required": ["city"],
    },
}


def call(name, **params):
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def candidate(text, *, reasoning="Thinking.", status="completed"):
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
            "model": "policy",
            "object": "response",
            "status": status,
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": output,
        }
    )


def request(sample_id, category, gold, tools, text, *, reasoning="Thinking."):
    return BFCLv3VerifyRequest.model_validate(
        {
            "id": f"bfclv3/{sample_id}",
            "responses_create_params": {
                "input": [{"role": "system", "content": "tools"}, {"role": "user", "content": ""}]
            },
            "verifier_metadata": {"id": sample_id, "category": category, "tools": tools, "gold_toolcall": gold},
            "response": candidate(text, reasoning=reasoning).model_dump(),
        }
    )


def server(expected=1, **overrides):
    values = {
        "name": "bfclv3",
        "host": "127.0.0.1",
        "port": 8170,
        "entrypoint": "app.py",
        "expected_num_rows": expected,
    }
    return BFCLv3Server(config=BFCLv3Config(**(values | overrides)), server_client=MagicMock(spec=ServerClient))


AREA_GOLD = [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}]


@pytest.mark.parametrize(
    ("text", "reasoning", "expected"),
    [
        ("\n\nAnswer.", "Think.", "Answer."),
        ("Answer.", None, ""),  # reasoning mode without </think> -> empty prediction
        ("x</think>\n\ny", None, "y"),
    ],
)
def test_candidate_prediction_matches_reference(text, reasoning, expected):
    assert candidate_prediction(candidate(text, reasoning=reasoning)) == expected


def test_parse_literal_evaluates_values_and_merges_blocks():
    text = "Let me call it.\n" + call("a.b", n="5", xs='["x", 1]', s="Houston, TX") + call("c", flag="True")
    calls, status = parse_prediction(text)
    assert status == "xml"
    assert calls == [
        {"name": "a.b", "arguments": {"n": 5, "xs": ["x", 1], "s": "Houston, TX"}},
        {"name": "c", "arguments": {"flag": True}},
    ]


def test_parse_accepts_missing_wrapper_and_drops_think_blocks():
    text = "<think>" + call("ignored", a="1") + "</think><function=f>\n<parameter=a>\n2\n</parameter>\n</function>"
    assert parse_prediction(text) == ([{"name": "f", "arguments": {"a": 2}}], "xml")
    assert parse_prediction("I can't help with that.") == ([], "none")
    assert parse_prediction("  ") == ([], "empty")


@pytest.mark.parametrize(
    ("pred", "valid", "error"),
    [
        ([{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}], True, ""),  # optional omitted
        ([{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}], True, ""),
        ([{"name": "calculate_triangle_area", "arguments": {"base": "10", "height": 5}}], False, "type_error:simple"),
        (
            [{"name": "calculate_triangle_area", "arguments": {"base": 10}}],
            False,
            "simple_function_checker:missing_required",
        ),
        ([], False, "no_prediction"),
    ],
)
def test_simple_rows_use_gorilla_checker(pred, valid, error):
    assert score_row(pred, AREA_GOLD, [AREA_TOOL], "simple") == (valid, error)


def test_parallel_categories_ignore_call_order():
    gold = [
        {"name": "weather.get", "arguments": {"city": "Paris"}},
        {"name": "weather.get", "arguments": {"city": "Rome"}},
    ]
    pred = list(reversed(gold))
    assert score_row(pred, gold, [WEATHER_TOOL], "parallel") == (True, "")
    assert score_row(pred, gold, [WEATHER_TOOL], "parallel_multiple") == (True, "")
    assert score_row(pred[:1], gold, [WEATHER_TOOL], "parallel")[0] is False


def test_irrelevance_passes_only_on_abstention():
    assert score_row([], [], [AREA_TOOL], "irrelevance") == (True, "")
    assert score_row(AREA_GOLD, [], [AREA_TOOL], "irrelevance") == (
        False,
        "irrelevance_function_checker:should_abstain",
    )


def test_verify_scores_the_final_answer():
    svc = server()
    text = call("calculate_triangle_area", base="10", height="5")
    result = asyncio.run(svc.verify(request("simple_0", "simple", AREA_GOLD, [AREA_TOOL], text)))
    assert (result.reward, result.parse_status, result.num_pred_calls, result.name_correct) == (1.0, "xml", 1, True)
    # Calls made only inside the reasoning do not count.
    hidden = asyncio.run(svc.verify(request("simple_0", "simple", AREA_GOLD, [AREA_TOOL], "Done.", reasoning=text)))
    assert (hidden.reward, hidden.parse_status, hidden.error_type) == (0.0, "none", "no_prediction")


def test_example_rows_validate():
    rows = [json.loads(line) for line in EXAMPLES.read_text().splitlines()]
    categories = [row["verifier_metadata"]["category"] for row in rows]
    assert categories == ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
    for row in rows:
        BFCLv3VerifyRequest.model_validate(row | {"response": candidate("x").model_dump()})


def test_metadata_rejects_gold_that_contradicts_category():
    with pytest.raises(ValueError, match="gold_toolcall"):
        request("irrelevance_0", "irrelevance", AREA_GOLD, [AREA_TOOL], "x")


def test_metrics_report_ast_micro_and_irrelevance_separately():
    svc = server(expected=4)
    good = call("calculate_triangle_area", base="10", height="5")
    rows = [
        asyncio.run(svc.verify(request("simple_0", "simple", AREA_GOLD, [AREA_TOOL], good))),
        asyncio.run(svc.verify(request("simple_1", "simple", AREA_GOLD, [AREA_TOOL], "No call."))),
        asyncio.run(svc.verify(request("irrelevance_0", "irrelevance", [], [AREA_TOOL], "No tool fits."))),
        asyncio.run(svc.verify(request("irrelevance_1", "irrelevance", [], [AREA_TOOL], good))),
    ]
    metrics = svc.compute_metrics([[row.model_dump() for row in rows]])
    assert (metrics["bfclv3/ast_accuracy"], metrics["bfclv3/ast_rows"]) == (0.5, 2)
    assert metrics["bfclv3/irrelevance_accuracy"] == 0.5
    assert metrics["bfclv3/irrelevance_false_positive_rate"] == 0.5
    assert metrics["bfclv3/simple/accuracy"] == 0.5
    assert (metrics["bfclv3/parsed_rows"], metrics["bfclv3/no_toolcall_rows"]) == (2, 2)
    assert metrics["bfclv3/complete"] is True
    partial = svc.compute_metrics([[rows[0].model_dump()]])
    assert (partial["bfclv3/missing_rows"], partial["bfclv3/complete"]) == (3, False)
