# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-compatible AudioMC prediction extraction, judge prompt, and judge parsing.

Each function mirrors VLMEvalKitMcore (``vlmeval/dataset/avlm/audiomc.py`` and
``vlmeval/inference.py`` at commit df7a6995) so scores stay comparable.
"""

import json
import re
from collections.abc import Sequence

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.audiomc.task_data import AudioMCTurn


def candidate_prediction(response: NeMoGymResponse) -> str:
    """Return the text VLMEvalKit would save as the prediction.

    The reference client formats ``<think>\\n{reasoning}\\n</think>\\n{content}``
    when the server returned reasoning, else the bare content. Its inference
    loop then keeps the text after the last ``</think>``; with reasoning
    enabled, a response without ``</think>`` becomes an empty prediction.
    """
    content = ""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            content = "\n".join(part.text for part in item.content if part.type == "output_text")
            break
    reasoning = ""
    for item in reversed(response.output):
        if item.type == "reasoning":
            reasoning = "".join(part.text for part in (item.summary or []))
            break
    raw = f"<think>\n{reasoning}\n</think>\n{content}" if reasoning else content
    end = raw.rfind("</think>")
    if end == -1:
        return ""
    return raw[end + len("</think>") :].lstrip("\n")


def conversation_history(turns: Sequence[AudioMCTurn], prediction: str) -> str:
    """Transcript history with the candidate's answer as the final assistant turn."""
    spoken = [turn for turn in turns if turn.user_transcript]
    if not spoken:
        return f"Assistant: {prediction}"
    last_turn = max(turn.turn for turn in spoken)
    parts: list[str] = []
    for turn in sorted(spoken, key=lambda item: item.turn):
        user_text = str(turn.user_transcript or "").strip()
        if user_text:
            parts.append(f"User: {user_text}")
        if turn.turn < last_turn:
            assistant_text = str(turn.assistant_transcript or "").strip()
            if assistant_text:
                parts.append(f"Assistant: {assistant_text}")
        else:
            parts.append(f"Assistant: {prediction}")
    return "\n\n".join(parts)


def _normalize_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def parse_judgment(raw: str | None) -> tuple[str, bool]:
    """Parse ``{"explanation", "criteria_met"}`` from a fenced or bare JSON reply."""
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("judge JSON is not an object")
    criteria_met = _normalize_bool(parsed.get("criteria_met"))
    if criteria_met is None:
        raise ValueError(f"criteria_met is not a boolean: {parsed.get('criteria_met')!r}")
    explanation = parsed.get("explanation", "")
    return ("" if explanation is None else str(explanation)), criteria_met
