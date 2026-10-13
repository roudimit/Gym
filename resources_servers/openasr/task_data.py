# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Labels for the byte-preserving OpenASR datasets prepared in text-evals."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Subset = Literal[
    "ami",
    "earnings22",
    "gigaspeech",
    "librispeech_test_clean",
    "librispeech_test_other",
    "spgispeech",
    "tedlium",
    "voxpopuli",
]


class OpenASRMetadata(BaseModel):
    """Reference text and source identity stay outside candidate-visible input."""

    model_config = ConfigDict(extra="allow")

    subset: Subset
    id: str = Field(min_length=1)
    text: str | None

    # Source and audio provenance written by the dataset preparer; verify() does not read these.
    dataset: str | None = None
    hf_config: str | None = None
    hf_split: str | None = None
    row_index: int | None = None
    audio_format: str | None = None
    audio_length_s: float | None = None
    audio_sha256: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bits_per_sample: int | None = None
    num_frames: int | None = None
    size_bytes: int | None = None
    offset_data: int | None = None
    wav_format: int | None = None


# Rows still nest these fields under verifier_metadata; validation splices them up to this flat schema.
TaskData = OpenASRMetadata
