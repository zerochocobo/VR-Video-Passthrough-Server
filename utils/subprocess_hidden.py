from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Sequence


def hidden_subprocess_kwargs() -> dict:
    """Return subprocess kwargs that suppress transient console windows."""
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    if os.name == "nt" and hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _forward_pipe(pipe, target) -> None:
    if pipe is None:
        return
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            data = bytes(chunk)
            binary_target = getattr(target, "buffer", None)
            if binary_target is not None:
                binary_target.write(data)
                binary_target.flush()
            else:
                target.write(data.decode("utf-8", "replace"))
                target.flush()
    except Exception as exc:
        print(f"[process] warning: child output forwarding failed: {type(exc).__name__}: {exc}", flush=True)


def run_hidden_streaming(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    check: bool = False,
    exit_label: str = "process",
) -> int:
    """Run a hidden child process while explicitly forwarding stdout/stderr.

    This avoids relying on inherited stdio handles from a hidden parent process,
    which is fragile when the parent itself is already connected to pipes.
    """
    cmd_list = [str(part) for part in cmd]
    try:
        proc = subprocess.Popen(
            cmd_list,
            cwd=str(Path(cwd)) if cwd is not None else None,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            **hidden_subprocess_kwargs(),
        )
    except Exception as exc:
        print(f"[{exit_label}] ERROR: failed to start child process: {type(exc).__name__}: {exc}", flush=True)
        if check:
            raise subprocess.CalledProcessError(127, cmd_list) from exc
        return 127

    stdout_thread = threading.Thread(target=_forward_pipe, args=(proc.stdout, sys.stdout), name=f"{exit_label}-stdout", daemon=True)
    stderr_thread = threading.Thread(target=_forward_pipe, args=(proc.stderr, sys.stderr), name=f"{exit_label}-stderr", daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    rc = int(proc.wait())
    stdout_thread.join(timeout=5.0)
    stderr_thread.join(timeout=5.0)
    print(f"[{exit_label}] child process exited rc={rc}", flush=True)
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd_list)
    return rc
