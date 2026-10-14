# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import os
import tarfile
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.tau_voice_simple.audio_archive import AudioArchive
from resources_servers.tau_voice_simple.task_data import TauVoiceSimpleMetadata
from resources_servers.tau_voice_simple.tts import (
    GPT4O_MINI_TTS_VOICES,
    TTSClient,
    TTSConfig,
    decode_pcm16_wav,
    pcm16_wav_bytes,
    voice_for_session,
)


HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent


def tts_config(**overrides) -> TTSConfig:
    return TTSConfig(
        **{"endpoint": "http://tts/v1/audio/speech", "api_key": "secret", "model": "tts", "voice": "onyx"} | overrides
    )


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestTTS:
    def test_voice_rotation_matches_reference(self) -> None:
        assert len(GPT4O_MINI_TTS_VOICES) == 11
        assert voice_for_session(40) == "onyx"  # retail__40 used onyx in the reference run
        assert voice_for_session(11) == "alloy"

    def test_request_payload_and_decoding(self) -> None:
        wav = pcm16_wav_bytes(b"\x01\x00" * 240, 24000)
        seen = {}

        def urlopen(request, timeout):
            seen["payload"] = json.loads(request.data)
            seen["auth"] = request.get_header("Authorization")
            return _Response(wav)

        result = TTSClient(tts_config(), urlopen=urlopen).synthesize("  Hello there. ")
        assert seen["payload"]["input"] == "Hello there."
        assert seen["payload"]["response_format"] == "wav" and seen["payload"]["voice"] == "onyx"
        assert "Preserve the supplied wording exactly" in seen["payload"]["instructions"]
        assert seen["auth"] == "Bearer secret"
        assert (result.sample_rate, len(result.pcm_s16le)) == (24000, 480)
        assert decode_pcm16_wav(result.wav_bytes()) == (b"\x01\x00" * 240, 24000)

    def test_retries_server_errors_then_redacts_key(self, monkeypatch) -> None:
        monkeypatch.setattr("resources_servers.tau_voice_simple.tts.sleep", lambda seconds: None)
        attempts = []

        def urlopen(request, timeout):
            attempts.append(1)
            raise urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"key secret leaked"))

        with pytest.raises(RuntimeError, match="HTTP 503") as error:
            TTSClient(tts_config(max_retries=2), urlopen=urlopen).synthesize("Hi")
        assert len(attempts) == 3
        assert "secret" not in str(error.value)

    def test_rejects_unknown_voice(self) -> None:
        with pytest.raises(ValueError, match="Unsupported TTS voice"):
            TTSClient(tts_config(voice="robot"))


def test_audio_archive_appends_episodes(tmp_path) -> None:
    archive = AudioArchive(tmp_path / "run" / "user_audio.tar")
    first = archive.append("airline__0", [("user_turn_0000.wav", b"a"), ("simulation.json", b"{}")])
    archive.append("retail__1", [("user_turn_0000.wav", b"bb")])
    assert first == ["airline__0/user_turn_0000.wav", "airline__0/simulation.json"]
    with tarfile.open(tmp_path / "run" / "user_audio.tar") as tar:
        assert tar.getnames() == first + ["retail__1/user_turn_0000.wav"]
        assert tar.extractfile("retail__1/user_turn_0000.wav").read() == b"bb"


def test_example_rows_validate() -> None:
    rows = [json.loads(line) for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    metadata = [TauVoiceSimpleMetadata.model_validate(row["verifier_metadata"]) for row in rows]
    assert {row.domain for row in metadata} == {"airline", "retail", "telecom"}
    assert all(row.tts_voice in GPT4O_MINI_TTS_VOICES for row in metadata)


tau2 = pytest.importorskip("tau2", reason="tau2 is installed in the server venv")


def test_stop_guard_keeps_turn_open_for_requested_action() -> None:
    from resources_servers.tau_voice_simple.user import suppress_stop_after_pending_action

    writes = ("cancel_reservation",)
    assert suppress_stop_after_pending_action("Yes, please cancel it. ###STOP###", writes) == "Yes, please cancel it."
    assert suppress_stop_after_pending_action("Thanks, bye! ###STOP###", writes) == "Thanks, bye! ###STOP###"
    assert (
        suppress_stop_after_pending_action("Yes, please cancel it. ###STOP###", ())
        == "Yes, please cancel it. ###STOP###"
    )


@pytest.fixture(scope="module")
def tau2_episode_module():
    data_dir = os.environ.get("TAU2_DATA_DIR")
    if not data_dir or not (Path(data_dir) / "tau2" / "domains" / "airline").is_dir():
        pytest.skip("set TAU2_DATA_DIR to Tau's data/ tree to run episode tests")
    from resources_servers.tau_voice_simple import episode

    return episode


def test_relay_episode_runs_tau_to_user_stop(tau2_episode_module, monkeypatch) -> None:
    """A scripted user and candidate drive Tau's orchestrator through a tool call to user_stop."""
    E = tau2_episode_module
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator import UserSimulator

    script = iter(["I want to cancel reservation EHGLP3.", "Okay, thanks. ###STOP###"])

    def fake_generate(self, message, state):
        return UserMessage(role="user", content=next(script))

    monkeypatch.setattr(UserSimulator, "_generate_next_message", fake_generate)

    class FakeTTS:
        def synthesize(self, text):
            return SimpleNamespace(
                pcm_s16le=b"\x00\x00" * 160,
                sample_rate=16000,
                model="tts",
                voice="onyx",
                wav_bytes=lambda: pcm16_wav_bytes(b"\x00\x00" * 160, 16000),
            )

    task = E.load_tasks("airline", "base")["0"]
    assert (
        "Agent tells the user that reservation EHGLP3 cannot be cancelled." in task.evaluation_criteria.nl_assertions
    )
    settings = E.EpisodeSettings("s2t", 100, 10, 1200.0, "dummy", {})
    episode = E.Episode(
        domain="airline", task=task, seed=42, simulation_id="t", settings=settings, tts_client=FakeTTS()
    )

    first = episode.start()
    assert first["observation"]["type"] == "user_audio" and first["observation"]["turn_index"] == 0
    tool = episode.submit(
        E.assistant_message(
            None, {"id": "c1", "name": "get_reservation_details", "arguments": {"reservation_id": "EHGLP3"}}, None
        )
    )
    assert tool["observation"]["type"] == "tool_results"
    assert tool["observation"]["results"][0]["tool_call_id"] == "c1"
    final = episode.submit(E.assistant_message("Sorry, it cannot be cancelled.", None, {"reasoning_content": "r"}))
    assert final == {"done": True, "termination_reason": "user_stop"}

    simulation = episode.finalize({"interaction_mode": "s2t"})
    summary = E.trajectory_summary(simulation)
    assert summary["num_user_audio_turns"] == 1 and summary["num_tool_calls"] == 1
    assert not any(message.get("audio_content") for message in json.loads(E.simulation_json(simulation))["messages"])
    with pytest.raises(RuntimeError, match="already ended"):
        episode.submit(E.assistant_message("Hello?", None, None))
