import argparse
import errno
import os
import pty
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import requests


DEFAULT_PROMPT = "Call the get_current_directory tool exactly once, then reply with only the cwd path."
PROMPT_MARKER = b">>>"
CHAT_LOCK_FILE = ".ollama_chat_lock"


@dataclass
class SessionResult:
    mode: str
    transcript: str
    used_tool: bool
    contains_cwd: bool
    contains_error: bool
    returncode: int

    @property
    def passed(self) -> bool:
        return self.used_tool and self.contains_cwd and not self.contains_error and self.returncode == 0


def _read_until(fd: int, marker: bytes, timeout: float) -> bytes:
    data = bytearray()
    end = time.time() + timeout
    while time.time() < end:
        ready, _, _ = select.select([fd], [], [], 1)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError as exc:
            if exc.errno == errno.EIO:
                return bytes(data)
            raise
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if marker in data:
            return bytes(data)
    return bytes(data)


def _cleanup_stale_lock(root: str) -> None:
    lock_path = os.path.join(root, CHAT_LOCK_FILE)
    if not os.path.exists(lock_path):
        return
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        pid = int(raw) if raw else None
    except Exception:
        pid = None

    if pid is not None:
        try:
            os.kill(pid, 0)
            return
        except OSError:
            pass

    try:
        os.remove(lock_path)
    except OSError:
        pass


def _wait_for_proxy(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=10)
            if response.ok:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Proxy did not become ready at {base_url}: {last_error}")


def _start_proxy(root: str, base_url: str, proxy_command: str, startup_timeout: float) -> subprocess.Popen:
    process = subprocess.Popen(
        proxy_command,
        cwd=root,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        _wait_for_proxy(base_url, startup_timeout)
    except Exception:
        _stop_proxy(process)
        raise
    return process


def _stop_proxy(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait(timeout=5)


def run_chat_session(root: str, python_path: str, model: str, openai_mode: bool, prompt: str, timeout: float) -> SessionResult:
    _cleanup_stale_lock(root)

    command = [python_path, "tools/ollama_chat.py", model, "-c"]
    if openai_mode:
        command.append("-o")

    master, slave = pty.openpty()
    env = os.environ.copy()
    env["PROMPT_TOOLKIT_NO_CPR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["TERM"] = "xterm"

    process = subprocess.Popen(
        command,
        cwd=root,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)

    transcript = bytearray()
    try:
        transcript.extend(_read_until(master, PROMPT_MARKER, 30))
        os.write(master, f"{prompt}\r".encode())
        transcript.extend(_read_until(master, PROMPT_MARKER, timeout))
        os.write(master, b"/exit\r")
        transcript.extend(_read_until(master, b"Goodbye!", 20))
        process.wait(timeout=10)
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    text = transcript.decode("utf-8", errors="replace")
    return SessionResult(
        mode="openai" if openai_mode else "native",
        transcript=text,
        used_tool="[tool] get_current_directory(" in text,
        contains_cwd=root in text,
        contains_error="❌ Error:" in text,
        returncode=process.returncode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end regression test for ollama_chat tool calling.")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name to test.")
    parser.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), help="Repository root.")
    parser.add_argument("--python", dest="python_path", default=sys.executable, help="Python interpreter to use for the chat CLI.")
    parser.add_argument("--host", default="127.0.0.1", help="Proxy host.")
    parser.add_argument("--port", default="11434", help="Proxy port.")
    parser.add_argument("--mode", choices=["native", "openai", "both"], default="both", help="Which chat transport(s) to test.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the model.")
    parser.add_argument("--timeout", type=float, default=240.0, help="Seconds to wait for a chat turn to finish.")
    parser.add_argument("--proxy-command", help="Optional shell command used to start a temporary proxy for the test run.")
    parser.add_argument("--proxy-startup-timeout", type=float, default=60.0, help="Seconds to wait for a started proxy to become ready.")
    parser.add_argument("--show-transcript", action="store_true", help="Always print the full transcript for each mode.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    base_url = f"http://{args.host}:{args.port}"

    started_proxy = None
    if args.proxy_command:
        started_proxy = _start_proxy(root, base_url, args.proxy_command, args.proxy_startup_timeout)
    else:
        _wait_for_proxy(base_url, args.proxy_startup_timeout)

    try:
        modes = [args.mode] if args.mode != "both" else ["native", "openai"]
        results = [
            run_chat_session(
                root=root,
                python_path=args.python_path,
                model=args.model,
                openai_mode=(mode == "openai"),
                prompt=args.prompt,
                timeout=args.timeout,
            )
            for mode in modes
        ]
    finally:
        if started_proxy is not None:
            _stop_proxy(started_proxy)

    overall_ok = True
    for result in results:
        print(f"[{result.mode}] passed={result.passed} used_tool={result.used_tool} contains_cwd={result.contains_cwd} contains_error={result.contains_error} returncode={result.returncode}")
        if args.show_transcript or not result.passed:
            print(f"--- transcript:{result.mode} ---")
            print(result.transcript)
        overall_ok = overall_ok and result.passed

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())