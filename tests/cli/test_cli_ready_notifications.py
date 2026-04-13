import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_module
from cli import HermesCLI
from hermes_cli.callbacks import prompt_for_secret
from hermes_cli.config import DEFAULT_CONFIG


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


def _make_cli_stub(with_app=True):
    cli = HermesCLI.__new__(HermesCLI)
    cli.osc9_ready_on_waiting = True
    cli._agent_running = False
    cli._command_running = False
    cli._sudo_state = None
    cli._secret_state = None
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._voice_processing = False
    cli._voice_recording = False
    cli._should_exit = False
    cli._last_waiting_for_user_input = None
    cli._last_invalidate = 0.0
    cli._secret_deadline = 0
    cli._modal_input_snapshot = None
    cli._invalidate = MagicMock()
    cli._app = _FakeApp() if with_app else None
    return cli


def test_display_config_defaults_osc9_ready_disabled():
    assert DEFAULT_CONFIG["display"]["osc9_ready_on_waiting"] is False


def test_emit_terminal_escape_uses_real_tty_stdout_and_os_write():
    cli = _make_cli_stub()
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = True
    fake_stdout.fileno.return_value = 123

    with patch.object(cli_module.sys, "__stdout__", fake_stdout), patch.object(cli_module.os, "write") as mock_write:
        HermesCLI._emit_terminal_escape(cli, "\x1b]9;Hermes ready\x07")

    mock_write.assert_called_once_with(123, b"\x1b]9;Hermes ready\x07")


def test_sync_ready_notification_emits_once_per_busy_to_waiting_transition():
    cli = _make_cli_stub(with_app=False)
    cli._emit_ready_osc9 = MagicMock()

    HermesCLI._sync_ready_notification(cli)
    HermesCLI._sync_ready_notification(cli)

    cli._emit_ready_osc9.assert_called_once()

    cli._agent_running = True
    HermesCLI._sync_ready_notification(cli)
    cli._agent_running = False
    HermesCLI._sync_ready_notification(cli)

    assert cli._emit_ready_osc9.call_count == 2


def test_busy_command_syncs_ready_notification_on_entry_and_exit():
    cli = _make_cli_stub(with_app=False)
    cli._sync_ready_notification = MagicMock()

    with HermesCLI._busy_command(cli, "Testing notifications"):
        pass

    assert cli._sync_ready_notification.call_count == 2


def test_submit_secret_response_syncs_ready_notification():
    cli = _make_cli_stub()
    cli._sync_ready_notification = MagicMock()
    cli._secret_state = {
        "response_queue": queue.Queue(),
        "var_name": "TENOR_API_KEY",
        "prompt": "Tenor API key",
        "metadata": {},
    }
    cli._secret_deadline = 123

    HermesCLI._submit_secret_response(cli, "super-secret-value")

    assert cli._secret_state is None
    assert cli._secret_deadline == 0
    cli._sync_ready_notification.assert_called_once_with()


def test_prompt_for_secret_syncs_ready_notification_on_open_and_timeout():
    cli = _make_cli_stub(with_app=True)
    cli._sync_ready_notification = MagicMock()

    with patch("hermes_cli.callbacks.queue.Queue.get", side_effect=queue.Empty), patch(
        "hermes_cli.callbacks._time.monotonic",
        side_effect=[0, 121],
    ):
        result = prompt_for_secret(cli, "TENOR_API_KEY", "Tenor API key")

    assert result["success"] is True
    assert result["reason"] == "timeout"
    assert cli._sync_ready_notification.call_count == 2


def test_handle_approval_selection_syncs_ready_notification():
    cli = _make_cli_stub()
    cli._sync_ready_notification = MagicMock()
    response_queue = queue.Queue()
    cli._approval_state = {
        "command": "rm -rf /tmp/demo",
        "description": "delete temp files",
        "choices": ["once", "session", "always", "deny"],
        "selected": 0,
        "response_queue": response_queue,
    }

    cli._handle_approval_selection()

    assert cli._approval_state is None
    assert response_queue.get_nowait() == "once"
    cli._sync_ready_notification.assert_called_once_with()
