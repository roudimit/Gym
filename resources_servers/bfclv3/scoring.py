# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-compatible BFCL v3 prediction extraction, tool-call parsing, and AST scoring.

Each function mirrors VLMEvalKitMcore (``vlmeval/dataset/avlm/bfclv3.py`` and
``vlmeval/inference.py`` at commit df7a6995, prompt style ``template``) so
scores stay comparable. The AST check itself is Gorilla's, vendored verbatim in
``gorilla/ast_checker.py``.
"""

import ast
import re
from collections.abc import Sequence
from typing import Any

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.bfclv3.gorilla.ast_checker import ast_checker


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# <tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>
_XML_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_XML_FUNC_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


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
    if not reasoning and "</think>" not in content:
        # Thinking disabled (chat_template_kwargs.enable_thinking=false): the server returned no reasoning item and the
        # model never opened a think block, so the whole content is the answer. VLMEvalKitMcore's --no-reasoning path
        # formats such replies as "<think>\n<think></think>{content}" so that its strip-after-last-</think> keeps them.
        return content.lstrip("\n")
    raw = f"<think>\n{reasoning}\n</think>\n{content}" if reasoning else content
    end = raw.rfind("</think>")
    if end == -1:
        return ""
    return raw[end + len("</think>") :].lstrip("\n")


def parse_xml_calls(text: str) -> list[dict[str, Any]]:
    """Parse the XML tool syntax into BFCL's ``{name, arguments}`` shape.

    Values are literal-evaluated where possible, because the AST checker
    distinguishes 5 from "5". The ``<tool_call>`` wrapper is optional.
    """
    calls = []
    for block in _XML_CALL_RE.findall(text) or [text]:
        for name, body in _XML_FUNC_RE.findall(block):
            arguments: dict[str, Any] = {}
            for key, raw in _XML_PARAM_RE.findall(body):
                raw = raw.strip()
                try:
                    arguments[key.strip()] = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    arguments[key.strip()] = raw
            calls.append({"name": name.strip(), "arguments": arguments})
    return calls


def parse_prediction(text: str) -> tuple[list[dict[str, Any]], str]:
    """Return ``(calls, parse_status)``; status is ``empty``, ``xml``, or ``none`` (abstained)."""
    cleaned = _THINK_RE.sub("", text or "").strip()
    if not cleaned:
        return [], "empty"
    calls = parse_xml_calls(cleaned)
    if calls:
        return calls, "xml"
    return [], "none"


def normalize_tool_schema(func: dict[str, Any]) -> dict[str, Any]:
    """BFCL writes ``"type": "dict"``; the checker expects ``"object"``."""
    func = dict(func)
    if "parameters" in func:
        params = dict(func["parameters"])
        if params.get("type") == "dict":
            params["type"] = "object"
        params.setdefault("required", [])
        if "properties" in params:
            properties = dict(params["properties"])
            for name, value in properties.items():
                if isinstance(value, dict) and value.get("type") == "dict":
                    properties[name] = {**value, "type": "object"}
            params["properties"] = properties
        func["parameters"] = params
    return func


def unwrap_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unwrap OpenAI-style ``{"type": "function", "function": {...}}`` entries."""
    return [
        normalize_tool_schema(tool["function"] if ("function" in tool and "type" in tool) else tool) for tool in tools
    ]


def to_bfcl_output(calls: Sequence[Any]) -> list[dict[str, Any]]:
    """``[{"name": f, "arguments": {...}}]`` -> ``[{f: {...}}]``."""
    out = []
    for call in calls:
        if not isinstance(call, dict) or "name" not in call:
            continue
        args = call.get("arguments", call.get("parameters", {}))
        out.append({call["name"]: args if isinstance(args, dict) else {}})
    return out


def to_bfcl_possible_answer(calls: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gold calls -> BFCL possible answers; optional parameters also accept ``""`` (omitted)."""
    required_by_func = {}
    for tool in tools:
        inner = tool.get("function", tool) if "function" in tool else tool
        required_by_func[inner.get("name", "")] = set(inner.get("parameters", {}).get("required", []))
    result = []
    for call in calls:
        name = call["name"]
        required = required_by_func.get(name, set())
        result.append(
            {
                name: {
                    key: [value] if key in required else [value, ""]
                    for key, value in call.get("arguments", {}).items()
                }
            }
        )
    return result


def score_row(
    pred_calls: Sequence[dict[str, Any]],
    gold_calls: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    category: str,
) -> tuple[bool, str]:
    """Return ``(valid, error_type)`` from Gorilla's checker; ``error_type`` is empty on a pass."""
    func_descriptions = unwrap_tools(tools)
    model_output = to_bfcl_output(pred_calls)
    if category == "irrelevance":
        try:
            result = ast_checker(func_descriptions, model_output, [], category)
        except Exception as exc:  # noqa: BLE001 - the checker raises bare Exception
            return False, f"checker_exception: {exc}"
        valid = bool(result.get("valid", False))
        return valid, "" if valid else result.get("error_type", "")
    if not pred_calls:
        return False, "no_prediction"
    if not gold_calls:
        return False, "no_ground_truth"
    possible_answer = to_bfcl_possible_answer(gold_calls, tools)
    # The checker uses "parallel" for both parallel categories.
    checker_category = "parallel" if "parallel" in category else category
    try:
        result = ast_checker(func_descriptions, model_output, possible_answer, checker_category)
    except Exception as exc:  # noqa: BLE001
        return False, f"checker_exception: {exc}"
    valid = bool(result.get("valid", False))
    return valid, "" if valid else result.get("error_type", "")


def names_match(pred_calls: Sequence[Any], gold_calls: Sequence[dict[str, Any]]) -> bool:
    """Whether the predicted function names equal the gold names as a multiset."""
    predicted = sorted(call.get("name", "") for call in pred_calls if isinstance(call, dict))
    return predicted == sorted(call.get("name", "") for call in gold_calls)
