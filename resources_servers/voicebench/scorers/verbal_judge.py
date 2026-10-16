# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verbal-channel open-response judge (scoring_version verbal_v1).

The judge returns one JSON object per choice with two absolute axes and a
holistic overall score, each an integer from 0 to 5:

    {"helpfulness_completeness": {"evidence": "...", "score": 4},
     "conciseness_spoken_readiness": {"evidence": "...", "score": 3},
     "overall": {"evidence": "...", "score": 4}}

Unlike legacy_v1, an unparseable choice is dropped and recorded rather than
failing the run; a row is invalid only when no choice parses.
"""

import json
import re

from pydantic import JsonValue

from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.scorers.open_judge import OpenJudgeScorer


AXES = ("helpfulness_completeness", "conciseness_spoken_readiness")
SCORE_KEYS = (*AXES, "overall")
MAX_SCORE = 5
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _first_grade_object(text: str) -> dict:
    """Decode the first JSON object holding an overall entry; text such as a leaked
    think block may contain braces before it."""
    decoder = json.JSONDecoder()
    error = "judge output contains no JSON object"
    start = text.find("{")
    while start >= 0:
        try:
            document, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as decode_error:
            error = f"judge output is not valid JSON: {decode_error}"
        else:
            if isinstance(document, dict) and "overall" in document:
                return document
            error = "judge output has no JSON object with an overall score"
        start = text.find("{", start + 1)
    raise ValueError(error)


def _lenient_grade_object(text: str) -> dict:
    """Recover each key's score from near-JSON, e.g. a missing outer closing brace.

    Only the last grade is read (text such as a leaked think block can come
    before it), and each score must close its key's own brace-free entry, so a
    key without a score is rejected rather than borrowing a later key's score.
    """
    start = text.rfind(f'"{SCORE_KEYS[0]}"')
    if start < 0:
        raise ValueError(f"judge output has no {SCORE_KEYS[0]}")
    document = {}
    for key in SCORE_KEYS:
        # Greedy [^{}]* takes the entry's last "score", after any quoted in evidence.
        match = re.search(rf'"{key}"\s*:\s*\{{[^{{}}]*"score"\s*:\s*(-?\d+(?:\.\d+)?)\s*\}}', text[start:])
        if match is None:
            raise ValueError(f"judge output has no {key}.score")
        document[key] = {"score": json.loads(match.group(1))}
    return document


def parse_verbal_grade(text: str | None) -> dict[str, int]:
    """Return the integer 0-5 score for each axis and the overall score."""
    if text is None:
        raise ValueError("judge content is None")
    stripped = _FENCE.sub("", text.strip())
    try:
        document = _first_grade_object(stripped)
    except ValueError as error:
        try:
            document = _lenient_grade_object(stripped)
        except ValueError:
            raise error from None
    scores: dict[str, int] = {}
    for key in SCORE_KEYS:
        entry = document.get(key)
        score = entry.get("score") if isinstance(entry, dict) else None
        # bool is an int subclass; an integral float such as 4.0 is accepted.
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{key}.score must be an integer")
        # Range first: a huge integer cannot be converted to a float.
        if not 0 <= score <= MAX_SCORE:
            raise ValueError(f"{key}.score must be between 0 and {MAX_SCORE}")
        if score != int(score):
            raise ValueError(f"{key}.score must be an integer")
        scores[key] = int(score)
    return scores


def score_verbal_outputs(
    outputs: list[str | None], *, finish_reasons: list[str] | None = None, requested_n: int = 3
) -> ScoringResult:
    """Average every parsed choice; reward is the mean overall score divided by 5."""
    finish_reasons = finish_reasons if finish_reasons is not None else ["stop"] * len(outputs)
    if len(finish_reasons) != len(outputs):
        raise ValueError("finish_reasons must match judge output count")
    grades: list[dict[str, int] | None] = []
    errors: list[str] = []
    for index, output in enumerate(outputs):
        try:
            grades.append(parse_verbal_grade(output))
        except ValueError as error:
            grades.append(None)
            errors.append(f"choice {index}: {error}")
    if not outputs:
        errors.append("judge returned no choices")
    parsed = [grade for grade in grades if grade is not None]
    count = len(parsed)
    sums = {key: float(sum(grade[key] for grade in parsed)) for key in SCORE_KEYS}
    means: dict[str, float | None] = {key: sums[key] / count if count else None for key in SCORE_KEYS}
    overall = means["overall"]
    details: dict[str, JsonValue] = {
        "judge_valid": bool(count),
        "judge_outputs": outputs,
        "judge_scores": grades,
        "judge_finish_reasons": finish_reasons,
        "judge_requested_n": requested_n,
        "judge_actual_n": len(outputs),
        "judge_choice_count_mismatch": len(outputs) != requested_n,
        "judge_parse_errors": errors,
        "judge_parsed_n": count,
        "judge_numeric_valid": bool(count),
        "verbal_scores": means,
        "voicebench_score": overall,
        "voicebench_score_sum": sums["overall"],
        "voicebench_score_count": count,
    }
    # score_sum/score_count keep the overall score on the legacy aggregation path.
    metrics = {"score_sum": sums["overall"], "score_count": float(count)} | {
        f"{axis}_sum": sums[axis] for axis in AXES
    }
    return ScoringResult(overall / MAX_SCORE if overall is not None else 0.0, metrics, details, bool(count))


class VerbalJudgeScorer(OpenJudgeScorer):
    """Same transport and prompt templating as legacy_v1, with a JSON multi-axis grade."""

    async def score(self, answer: str, metadata: dict[str, JsonValue]) -> ScoringResult:
        judged, timings = await self.collect(answer, metadata)
        result = score_verbal_outputs(
            [choice.message.content for choice in judged.choices],
            finish_reasons=[choice.finish_reason for choice in judged.choices],
            requested_n=self.settings.judge_n,
        )
        return ScoringResult(
            result.reward, result.metrics, self.diagnostics(judged) | result.details | timings, result.valid
        )
