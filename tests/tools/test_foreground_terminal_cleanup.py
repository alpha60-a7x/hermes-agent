import threading
import time

from tools.environments.local import LocalEnvironment
from tools.interrupt import set_interrupt
from tools.terminal_tool import _active_environments, _env_lock, _last_activity, cleanup_vm


def _run_command_in_thread(env: LocalEnvironment, command: str, timeout: int = 120):
    result_holder: dict[str, object] = {"value": None}

    def _runner() -> None:
        result_holder["value"] = env.execute(command, timeout=timeout)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return thread, result_holder


def _force_interrupt_thread(thread: threading.Thread) -> None:
    if not thread.is_alive():
        return
    set_interrupt(True, thread_id=thread.ident)
    thread.join(timeout=5)
    set_interrupt(False, thread_id=thread.ident)


def test_local_environment_cleanup_kills_active_foreground_process():
    env = LocalEnvironment(cwd="/tmp", timeout=120)
    thread, result_holder = _run_command_in_thread(env, "sleep 60")

    try:
        time.sleep(0.5)
        env.cleanup()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert result_holder["value"] is not None
        assert result_holder["value"]["returncode"] != 0
    finally:
        _force_interrupt_thread(thread)
        env.cleanup()


def test_cleanup_vm_kills_active_foreground_local_process():
    task_id = "test-foreground-cleanup"
    env = LocalEnvironment(cwd="/tmp", timeout=120)
    thread, result_holder = _run_command_in_thread(env, "sleep 60")

    try:
        with _env_lock:
            _active_environments[task_id] = env
            _last_activity[task_id] = time.time()

        time.sleep(0.5)
        cleanup_vm(task_id)
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert result_holder["value"] is not None
        assert result_holder["value"]["returncode"] != 0
        with _env_lock:
            assert task_id not in _active_environments
    finally:
        _force_interrupt_thread(thread)
        cleanup_vm(task_id)
