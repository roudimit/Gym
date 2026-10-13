# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier metadata for BFCL v3 single-turn function calling."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Category = Literal["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
CATEGORIES: tuple[Category, ...] = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")
AST_CATEGORIES: tuple[Category, ...] = ("simple", "multiple", "parallel", "parallel_multiple")


class BFCLv3Metadata(BaseModel):
    """Tools and gold calls are verifier-only; the candidate sees the tools only through its prompt."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    category: Category
    tools: list[dict[str, Any]] = Field(min_length=1)
    # [{"name": ..., "arguments": {...}}]; empty exactly for irrelevance rows.
    gold_toolcall: list[dict[str, Any]]
    user_text: str | None = None
    # Audio clip length in seconds; provenance only, not read by verify().
    duration: float | None = None

    @model_validator(mode="after")
    def gold_matches_category(self) -> "BFCLv3Metadata":
        if (self.category == "irrelevance") != (not self.gold_toolcall):
            raise ValueError("gold_toolcall must be empty for irrelevance rows and nonempty otherwise")
        return self


# Rows still nest these fields under verifier_metadata; validation splices them up to this flat schema.
TaskData = BFCLv3Metadata
