# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier metadata for the AudioMC multi-turn audio benchmark."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Axis = Literal["INFERENCE_MEMORY", "INSTRUCTION_RETENTION", "SELF_COHERENCE", "VOICE_EDITING"]
AXES: tuple[Axis, ...] = ("INFERENCE_MEMORY", "INSTRUCTION_RETENTION", "SELF_COHERENCE", "VOICE_EDITING")


class AudioMCTurn(BaseModel):
    """One conversation turn; transcripts are judge-only and never sent to the candidate."""

    model_config = ConfigDict(extra="allow")

    turn: int
    user_transcript: str | None = None
    assistant_transcript: str | None = None


class AudioMCMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    axis: Axis
    rubric: list[str]
    turns: list[AudioMCTurn]


# Rows still nest these fields under verifier_metadata; validation splices them up to this flat schema.
TaskData = AudioMCMetadata
