import os
import queue
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_module
from cli import HermesCLI
from hermes_cli.callbacks import approval_callback, prompt_for_secret


class _FakeStdout:
    def __init__(self, is_tty=True, fileno_value=1):
        self._is_tty = is_tty
        self._fileno_value = fileno_value
        self.writes = []
        self.flushed = False

    def isatty(self):
        return self._is_tty

    def fileno(self):
        return self._fileno_value

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        self.flushed = True


class _FakeBuffer:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class _FakeApp:
    def __init__(self):
        self.invalidated = False
        self.current_buffer = _FakeBuffer()

    def invalidate(self):
        self.invalidated = True


def _make_cli_stub(with_app=False):
    cli = HermesCLI.__new__(HermesCLI)
    cli.osc9_ready_on_waiting = True
    cli._last_waiting_for_user_input = None
    cli._agent_running = False
    cli._command_running = False
    cli._should_exit = False
    cli._sudo_state = None
    cli._secret_state = None
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._voice_processing = False
    cli._voice_recording = False
    cli._secret_deadline = 0
    cli._last_invalidate = 0.0
    cli._command_status = ""
    cli._invalidate = MagicMock()
    cli._app = _FakeApp() if with_app else None
    return cli


def test_emit_terminal_escape_writes_raw_sequence_to_real_tty():
    cli = _make_cli_stub()
    fake_stdout = _FakeStdout(is_tty=True, fileno_value=7)

    with patch.object(sys, "__stdout__", fake_stdout), patch("os.write") as mock_write:
        cli._emit_terminal_escape("\x1b]9;Hermes ready\x07")

    mock_write.assert_called_once_with(7, b"\x1b]9;Hermes ready\x07")
    assert fake_stdout.writes == []


def test_sync_ready_notification_emits_once_per_busy_to_waiting_transition():
    cli = _make_cli_stub()
    cli._emit_ready_osc9 = MagicMock()

    cli._sync_ready_notification()
    cli._sync_ready_notification()
    assert cli._emit_ready_osc9.call_count == 1

    cli._agent_running = True
    cli._sync_ready_notification()
    assert cli._emit_ready_osc9.call_count == 1

    cli._agent_running = False
    cli._sync_ready_notification()
    assert cli._emit_ready_osc9.call_count == 2


def test_busy_command_syncs_ready_notification_on_entry_and_exit():
    cli = _make_cli_stub()
    cli._sync_ready_notification = MagicMock()

    with patch("builtins.print"):
        with cli._busy_command("Reloading"):
            assert cli._command_running is True

    assert cli._command_running is False
    assert cli._sync_ready_notification.call_count == 2


def test_prompt_for_secret_syncs_ready_notification_on_open_and_timeout():
    cli = _make_cli_stub(with_app=True)
    cli._sync_ready_notification = MagicMock()
    cleared = {"value": False}

    def clear_buffer():
        cleared["value"] = True

    cli._clear_secret_input_buffer = clear_buffer

    with patch("hermes_cli.callbacks.queue.Queue.get", side_effect=queue.Empty), patch(
        "hermes_cli.callbacks._time.monotonic",
        side_effect=[0, 121],
    ):
        result = prompt_for_secret(cli, "TENOR_API_KEY", "Tenor API key")

    assert result["success"] is True
    assert result["reason"] == "timeout"
    assert cleared["value"] is True
    assert cli._sync_ready_notification.call_count == 2


def test_approval_callback_syncs_ready_notification_on_open_and_close():
    cli = _make_cli_stub(with_app=True)
    cli._sync_ready_notification = MagicMock()
    result = {}

    def _run_callback():
        result["value"] = approval_callback(cli, "rm -rf /tmp/demo", "danger")

    thread = threading.Thread(target=_run_callback, daemon=True)
    thread.start()

    deadline = time.time() + 2
    while cli._approval_state is None and time.time() < deadline:
        time.sleep(0.01)

    assert cli._approval_state is not None
    cli._approval_state["response_queue"].put("deny")
    thread.join(timeout=2)

    assert result["value"] == "deny"
    assert cli._sync_ready_notification.call_count == 2


def test_init_reads_osc9_ready_on_waiting_from_config():
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {
            "compact": False,
            "tool_progress": "all",
            "osc9_ready_on_waiting": True,
        },
        "agent": {},
        "terminal": {"env_type": "local"},
    }

    with patch.object(cli_module, "get_tool_definitions", return_value=[]), patch.dict(
        "os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False
    ), patch.dict(cli_module.__dict__, {"CLI_CONFIG": clean_config}):
        cli = HermesCLI()

    assert cli.osc9_ready_on_waiting is True
    assert cli._last_waiting_for_user_input is None
