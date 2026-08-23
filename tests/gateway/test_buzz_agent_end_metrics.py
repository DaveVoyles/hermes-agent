"""Buzz agent:end hook context carries provider totals and the inbound event id."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


INBOUND_EVENT_ID = "aa" * 32


def _source():
    return SessionSource(
        platform=Platform("buzz"),
        chat_id="ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd",
        chat_type="group",
        user_id="npub1testowner",
    )


def _event():
    return MessageEvent(
        text="hello from buzz",
        source=_source(),
        message_id=INBOUND_EVENT_ID,
    )


def _runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:buzz:group:channel:user",
        session_id="sess-buzz",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform("buzz"),
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


@pytest.mark.asyncio
async def test_buzz_agent_end_preserves_provider_total_and_inbound_turn_id(
    monkeypatch, tmp_path
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "done",
            "messages": [
                {"role": "user", "content": "hello from buzz"},
                {"role": "assistant", "content": "done"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "api_calls": 1,
            "failed": False,
            "model": "stealth/ox-alpha",
            "provider": "openrouter",
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 125,
            "turn_input_tokens": 100,
            "turn_output_tokens": 20,
            "turn_total_tokens": 125,
            "token_delta_reliable": True,
            "session_id": "sess-buzz",
        }
    )

    await runner._handle_message_with_agent(
        _event(),
        _source(),
        "agent:main:buzz:group:channel:user",
        1,
    )

    end_context = next(
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    )
    assert end_context["platform"] == "buzz"
    assert end_context["turn_id"] == INBOUND_EVENT_ID
    assert end_context["total_tokens"] == 125
    assert end_context["total_tokens"] != (
        end_context["input_tokens"] + end_context["output_tokens"]
    )
    assert end_context["turn_total_tokens"] == 125
    assert end_context["turn_total_tokens"] != (
        end_context["turn_input_tokens"] + end_context["turn_output_tokens"]
    )
    assert end_context["token_delta_reliable"] is True
    assert end_context["session_id"] == "sess-buzz"
