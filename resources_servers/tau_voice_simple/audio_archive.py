# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One uncompressed tar per run holding every episode's synthesized user audio."""

import io
import tarfile
import threading
from pathlib import Path


class AudioArchive:
    """Append each finished episode's members to a single tar.

    Members are ``<episode>/user_turn_<NNNN>.wav`` plus ``<episode>/simulation.json``.
    Appends are serialized, so concurrent verifies never interleave tar blocks.
    A resumed run appends to the existing tar; a repeated episode name is
    stored again and readers take the last copy, as tar extraction does.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, episode: str, members: list[tuple[str, bytes]]) -> list[str]:
        names = [f"{episode}/{name}" for name, _ in members]
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self.path.exists() and self.path.stat().st_size else "w"
            with tarfile.open(self.path, mode, format=tarfile.PAX_FORMAT) as archive:
                for name, (_, data) in zip(names, members):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
        return names
