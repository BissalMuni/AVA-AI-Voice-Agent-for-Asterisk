"""Recorded greeting (provider `greeting_file`) lifecycle tests."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.config import OpenAIRealtimeProviderConfig
from src.core.models import CallSession
from src.engine import Engine


def _engine(play_ok=True):
    engine = Engine.__new__(Engine)
    engine.ari_client = SimpleNamespace(
        play_media_on_channel_with_id=AsyncMock(return_value=play_ok),
        stop_playback=AsyncMock(return_value=True),
    )
    engine._save_session = AsyncMock()
    engine._connection_audio_handoff_tasks = {}
    return engine


def test_provider_greeting_file_reads_dict_and_model_configs():
    assert Engine._provider_greeting_file(None) is None
    assert Engine._provider_greeting_file({"greeting_file": " custom/ava-greeting "}) == "custom/ava-greeting"
    assert Engine._provider_greeting_file({"greeting_file": ""}) is None
    cfg = OpenAIRealtimeProviderConfig(greeting="hi", greeting_file="custom/ava-greeting")
    assert Engine._provider_greeting_file(cfg) == "custom/ava-greeting"
    assert Engine._provider_greeting_file(OpenAIRealtimeProviderConfig()) is None


@pytest.mark.asyncio
async def test_greeting_file_starts_on_caller_and_replaces_setup_ringback():
    engine = _engine()
    session = CallSession(call_id="call-gf", caller_channel_id="caller-gf")
    session.connection_audio_playback_id = "connection-audio-1"
    session.connection_audio_media_uri = "tone:ring"

    assert await engine._start_greeting_file(session, "custom/ava-greeting") is True

    playback_id = session.greeting_file_playback_id
    assert playback_id and playback_id.startswith("greeting-file-")
    assert session.greeting_file_media_uri == "sound:custom/ava-greeting"
    assert session.greeting_file_started_ts > 0
    engine.ari_client.play_media_on_channel_with_id.assert_awaited_once_with(
        "caller-gf", "sound:custom/ava-greeting", playback_id
    )
    # Setup ringback is stopped so it does not overlap the recorded greeting.
    engine.ari_client.stop_playback.assert_awaited_once_with("connection-audio-1")
    assert session.connection_audio_playback_id is None
    assert engine._greeting_file_playbacks == {playback_id: "call-gf"}

    # Second start is a no-op.
    assert await engine._start_greeting_file(session, "custom/other") is True
    assert engine.ari_client.play_media_on_channel_with_id.await_count == 1


@pytest.mark.asyncio
async def test_greeting_file_start_failure_leaves_provider_greeting_alone():
    engine = _engine(play_ok=False)
    session = CallSession(call_id="call-gf-fail", caller_channel_id="caller-gf-fail")

    assert await engine._start_greeting_file(session, "custom/ava-greeting") is False
    assert session.greeting_file_playback_id is None
    engine._save_session.assert_not_awaited()

    # Unsafe/remote media is rejected before any ARI call.
    engine2 = _engine()
    assert await engine2._start_greeting_file(session, "sound:https://example.test/x.wav") is False
    engine2.ari_client.play_media_on_channel_with_id.assert_not_awaited()


def test_suppress_provider_greeting_clears_text_and_flags_provider():
    provider = SimpleNamespace(config=OpenAIRealtimeProviderConfig(greeting="안녕하세요. 당직 봇입니다."))
    cleared = Engine._suppress_provider_greeting(provider)
    assert cleared == "안녕하세요. 당직 봇입니다."
    assert provider.config.greeting == ""
    assert provider._greeting_file_active is True
    assert provider._greeting_file_text == "안녕하세요. 당직 봇입니다."

    dict_provider = SimpleNamespace(config={"greeting": "hello"})
    assert Engine._suppress_provider_greeting(dict_provider) == "hello"
    assert dict_provider.config["greeting"] == ""
    assert dict_provider._greeting_file_active is True


@pytest.mark.asyncio
async def test_playback_finished_clears_greeting_file_without_ari_stop():
    engine = _engine()
    session = CallSession(call_id="call-gf-done", caller_channel_id="caller-gf-done")
    await engine._start_greeting_file(session, "custom/ava-greeting")
    playback_id = session.greeting_file_playback_id
    engine.session_store = SimpleNamespace(get_by_call_id=AsyncMock(return_value=session))

    assert await engine._on_greeting_file_playback_finished("unrelated-playback") is False
    assert await engine._on_greeting_file_playback_finished(playback_id) is True

    assert session.greeting_file_playback_id is None
    assert session.greeting_file_media_uri is None
    assert session.greeting_file_started_ts == 0.0
    assert playback_id not in engine._greeting_file_playbacks
    # Natural completion must not issue an ARI stop for the finished playback.
    engine.ari_client.stop_playback.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_session_update_tells_model_greeting_already_played():
    from src.providers.openai_realtime import OpenAIRealtimeProvider

    cfg = OpenAIRealtimeProviderConfig(
        api_key="test-key",
        instructions="당직실 응대 봇입니다.",
        greeting="안녕하세요. 당직 봇입니다.",
        greeting_file="custom/ava-greeting",
    )
    provider = OpenAIRealtimeProvider(cfg, on_event=AsyncMock())
    provider._call_id = "call-gf-note"
    sent = []

    async def _send_json(payload):
        sent.append(payload)

    provider._send_json = _send_json

    # Without a recorded greeting the instructions are untouched.
    await provider._send_session_update()
    plain = sent[-1]["session"]["instructions"]
    assert "recorded greeting" not in plain

    Engine._suppress_provider_greeting(provider)
    await provider._send_session_update()
    noted = sent[-1]["session"]["instructions"]
    assert noted.startswith(plain)
    assert "recorded greeting has already been played" in noted
    assert "안녕하세요. 당직 봇입니다." in noted
    assert provider.config.greeting == ""


@pytest.mark.asyncio
async def test_cleanup_stop_issues_ari_stop_while_greeting_still_playing():
    engine = _engine()
    session = CallSession(call_id="call-gf-cleanup", caller_channel_id="caller-gf-cleanup")
    await engine._start_greeting_file(session, "custom/ava-greeting")
    playback_id = session.greeting_file_playback_id

    await engine._stop_greeting_file(session, reason="call-cleanup")
    await engine._stop_greeting_file(session, reason="call-cleanup-again")

    engine.ari_client.stop_playback.assert_awaited_once_with(playback_id)
    assert session.greeting_file_playback_id is None
    assert playback_id not in engine._greeting_file_playbacks
