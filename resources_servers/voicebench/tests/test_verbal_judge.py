# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from resources_servers.voicebench.scorers.verbal_judge import (
    VerbalJudgeScorer,
    parse_verbal_grade,
    score_verbal_outputs,
)
from resources_servers.voicebench.tests.test_app import request, server
from resources_servers.voicebench.tests.test_judge import client, judge_settings


def grade(content=4, delivery=3, overall=4):
    return json.dumps(
        {
            "helpfulness_completeness": {"evidence": "covers the essentials", "score": content},
            "conciseness_spoken_readiness": {"evidence": "one list read aloud", "score": delivery},
            "overall": {"evidence": "useful, slightly long", "score": overall},
        }
    )


def verbal_server(tmp_path, **overrides):
    return server(tmp_path, scoring_version="verbal_v1", **overrides)


@pytest.mark.parametrize(
    "text",
    [
        grade(),
        f"```json\n{grade()}\n```",
        f"Here is my grade: {grade()} done",
        grade().replace(": 4}", ": 4.0}"),
        grade()[:-1] + "\n```",
        grade().replace('"one list read aloud"', '"one list "read" aloud"'),
        f'<think>use {{score}} keys, e.g. {{"a": 1}}</think>{grade()}',
    ],
)
def test_parses_fenced_embedded_and_integral_float_grades(text):
    assert parse_verbal_grade(text) == {
        "helpfulness_completeness": 4,
        "conciseness_spoken_readiness": 3,
        "overall": 4,
    }


@pytest.mark.parametrize(
    "text",
    [
        None,
        "4",
        "{not json",
        "[1, 2]",
        grade(content=6),
        grade(delivery=-1),
        grade(overall=3.5),
        grade(overall=True),
        grade(content="4"),
        grade().replace('"overall"', '"total"'),
        '{"helpfulness_completeness": {"score": Infinity}}',
        grade(overall=10**400),
        # Near-JSON whose first axis lacks a score must not borrow the next axis's score.
        '{"helpfulness_completeness": {"evidence": "good"}, '
        '"conciseness_spoken_readiness": {"evidence": "x", "score": 1}, "overall": {"evidence": "y", "score": 2}',
    ],
)
def test_rejects_malformed_or_out_of_range_grades(text):
    with pytest.raises(ValueError):
        parse_verbal_grade(text)


def test_scores_average_parsed_choices_and_record_dropped_ones():
    scored = score_verbal_outputs([grade(5, 4, 5), "garbage", grade(3, 2, 3)])
    assert scored.valid
    assert scored.reward == pytest.approx(0.8)
    assert scored.metrics == {
        "score_sum": 8.0,
        "score_count": 2.0,
        "helpfulness_completeness_sum": 8.0,
        "conciseness_spoken_readiness_sum": 6.0,
    }
    assert scored.details["verbal_scores"] == {
        "helpfulness_completeness": 4.0,
        "conciseness_spoken_readiness": 3.0,
        "overall": 4.0,
    }
    assert scored.details["judge_parsed_n"] == 2 and scored.details["judge_scores"][1] is None
    assert scored.details["judge_parse_errors"][0].startswith("choice 1:")


def test_row_is_invalid_only_when_no_choice_parses():
    scored = score_verbal_outputs(["bad", None])
    assert not scored.valid and scored.reward == 0.0
    assert scored.details["voicebench_score"] is None and len(scored.details["judge_parse_errors"]) == 2
    empty = score_verbal_outputs([], requested_n=3)
    assert not empty.valid and empty.details["judge_parse_errors"] == ["judge returned no choices"]


def test_word_count_placeholder_is_filled_from_the_answer_only(tmp_path):
    settings = judge_settings(tmp_path)
    prompt = tmp_path / "verbal.yaml"
    prompt.write_text(yaml.safe_dump({"system": "s", "user": "{prompt}|{response}|{word_count} words"}))
    settings = settings.model_copy(update={"judge_prompt_file": str(prompt)})
    network = client([grade()])
    scorer = VerbalJudgeScorer(settings, network)
    result = asyncio.run(scorer.score("three short words", {"prompt": "say {word_count}"}))
    messages = network.post.call_args.kwargs["json"].model_dump(exclude_unset=True)["messages"]
    assert messages[-1]["content"] == "say {word_count}|three short words|3 words"
    assert result.valid and result.reward == pytest.approx(0.8)


def test_verbal_version_requires_open_scoring(tmp_path):
    with pytest.raises(ValidationError, match="verbal_v1 requires scoring_type=open"):
        server(tmp_path, subset="sd-qa", scoring_type="reference_qa", scoring_version="verbal_v1")
    with pytest.raises(ValidationError):
        server(tmp_path, scoring_version="verbal_v2")


def test_verbal_server_verifies_and_aggregates_axis_scores(tmp_path):
    svc = verbal_server(tmp_path, expected=3, ids=["test_0", "test_1", "test_2"])
    svc.server_client.post.return_value = client([grade(5, 4, 5), grade(3, 4, 3)]).post.return_value
    first = asyncio.run(svc.verify(request("test_0"))).model_dump(mode="json")
    assert first["scoring_version"] == "verbal_v1" and first["reward"] == pytest.approx(0.8)
    assert first["verbal_scores"]["conciseness_spoken_readiness"] == 4.0

    svc.server_client.post.return_value = client([grade(1, 5, 1)]).post.return_value
    second = asyncio.run(svc.verify(request("test_1"))).model_dump(mode="json")
    svc.server_client.post.return_value = client(["not a grade"]).post.return_value
    third = asyncio.run(svc.verify(request("test_2"))).model_dump(mode="json")
    assert not third["score_valid"] and third["failure_kind"] is not None

    rows = [
        row | {"_ng_task_index": index, "_ng_rollout_index": 0} for index, row in enumerate((first, second, third))
    ]
    metrics = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))).key_metrics
    # Valid means pool the three parsed choices; status-safe averages per row over all three rows.
    assert metrics["voicebench/normalized_mean_valid"] == pytest.approx(60)
    assert metrics["voicebench/verbal/overall/normalized_mean_valid"] == pytest.approx(60)
    assert metrics["voicebench/verbal/helpfulness_completeness/normalized_mean_valid"] == pytest.approx(60)
    assert metrics["voicebench/verbal/conciseness_spoken_readiness/normalized_mean_valid"] == pytest.approx(260 / 3)
    assert metrics["voicebench/verbal/helpfulness_completeness/status_safe_normalized_score"] == pytest.approx(
        100 * (4 + 1 + 0) / 5 / 3
    )
    assert metrics["voicebench/verbal/conciseness_spoken_readiness/status_safe_normalized_score"] == pytest.approx(
        100 * (4 + 5 + 0) / 5 / 3
    )
    assert metrics["voicebench/status_safe_normalized_score"] == pytest.approx(100 * (0.8 + 0.2) / 3)
    assert metrics["voicebench/invalid_scores"] == 1 and not metrics["voicebench/complete"]


def test_legacy_metrics_have_no_verbal_keys(tmp_path):
    from resources_servers.voicebench.scorers.open_judge import score_outputs

    svc = server(tmp_path)
    scored = score_outputs(["4"])
    row = scored.details | {
        "sample_id": "test_0",
        "score_valid": True,
        "scoring_metrics": scored.metrics,
        "candidate_valid": True,
        "reward": scored.reward,
        "status_safe_reward": scored.reward,
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
    }
    metrics = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=[row]))).key_metrics
    assert not any(key.startswith("voicebench/verbal/") for key in metrics)


def test_null_top_p_is_omitted_from_the_judge_request(tmp_path):
    network = client([grade()])
    scorer = VerbalJudgeScorer(judge_settings(tmp_path, judge_top_p=None, judge_n=1, judge_temperature=0.0), network)
    asyncio.run(scorer.score("answer", {"prompt": "question"}))
    params = network.post.call_args.kwargs["json"].model_dump(exclude_unset=True)
    assert "top_p" not in params and params["temperature"] == 0.0 and params["n"] == 1


def test_lenient_fallback_reads_only_the_final_grade_and_each_entrys_own_score():
    # Unescaped quotes break strict JSON; a scratch score precedes the real grade.
    text = (
        '<think>maybe "overall": {"score": 5}</think>'
        '{"helpfulness_completeness": {"evidence": "it says "score": 5 here", "score": 1}, '
        '"conciseness_spoken_readiness": {"evidence": "ok", "score": 4}, '
        '"overall": {"evidence": "y "q"", "score": 0}}'
    )
    assert parse_verbal_grade(text) == {
        "helpfulness_completeness": 1,
        "conciseness_spoken_readiness": 4,
        "overall": 0,
    }
