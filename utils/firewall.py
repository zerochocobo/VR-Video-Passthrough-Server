"""Windows Firewall helper for DLNA discovery and HTTP playback.

The server needs inbound TCP on the HTTP port and inbound UDP/1900 for SSDP.
When the process is already elevated, rules are added directly with netsh.
Otherwise a temporary batch file is launched through ShellExecuteW with the
"runas" verb so Windows can show a UAC prompt.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from config import HTTP_PORT
from utils.logger import get
from utils.subprocess_hidden import hidden_subprocess_kwargs

log = get("firewall")

OLD_RULE_HTTP = "PTServer HTTP"
OLD_RULE_SSDP = "PTServer SSDP"
RULE_HTTP = "PTServer HTTP Private"
RULE_SSDP = "PTServer SSDP Private"


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore[attr-defined]
    except Exception:
        return False


def _rule_exists(name: str) -> bool:
    try:
        r = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
            capture_output=True,
            text=True,
            **hidden_subprocess_kwargs(),
        )
    except FileNotFoundError:
        return False
    return r.returncode == 0 and "No rules match" not in (r.stdout or "")


def _netsh_add(name: str, proto: str, port: int) -> bool:
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name={name}",
        "dir=in", "action=allow",
        f"protocol={proto}",
        f"localport={port}",
        "profile=private",
        "edge=no",
        "enable=yes",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, **hidden_subprocess_kwargs())
    if r.returncode != 0:
        log.warning("netsh add failed (%s): %s", name, (r.stderr or r.stdout).strip())
        return False
    return True


def _netsh_delete(name: str) -> None:
    subprocess.run(
        ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={name}"],
        capture_output=True,
        text=True,
        **hidden_subprocess_kwargs(),
    )


def _add_rules_direct() -> bool:
    _netsh_delete(OLD_RULE_HTTP)
    _netsh_delete(OLD_RULE_SSDP)
    ok1 = _rule_exists(RULE_HTTP) or _netsh_add(RULE_HTTP, "TCP", HTTP_PORT)
    ok2 = _rule_exists(RULE_SSDP) or _netsh_add(RULE_SSDP, "UDP", 1900)
    return ok1 and ok2


def _build_bat() -> Path:
    """Build a temporary elevated netsh script and return its path."""
    p = Path(tempfile.gettempdir()) / "ptserver_firewall_setup.bat"
    lines = [
        "@echo off",
        "setlocal",
        f'netsh advfirewall firewall delete rule name="{OLD_RULE_HTTP}" >nul 2>nul',
        f'netsh advfirewall firewall delete rule name="{OLD_RULE_SSDP}" >nul 2>nul',
        # The batch file deletes itself after netsh has finished.
        f'netsh advfirewall firewall add rule name="{RULE_HTTP}" dir=in '
        f'action=allow protocol=TCP localport={HTTP_PORT} profile=private edge=no enable=yes',
        f'netsh advfirewall firewall add rule name="{RULE_SSDP}" dir=in '
        f'action=allow protocol=UDP localport=1900 profile=private edge=no enable=yes',
        # Internal note.
        f'(goto) 2>nul & del "{p}"',
    ]
    p.write_text("\r\n".join(lines), encoding="utf-8")
    return p


def _elevate_run(bat: Path) -> bool:
    SW_HIDE = 0
    rc = ctypes.windll.shell32.ShellExecuteW(  # type: ignore[attr-defined]
        None, "runas", "cmd.exe", f'/c "{bat}"', None, SW_HIDE
    )
    # ShellExecuteW returns values greater than 32 for successful launches.
    return int(rc) > 32


def ensure_rules() -> bool:
    """Ensure firewall rules exist; return False only when setup is rejected."""
    if not _is_windows():
        return True

    try:
        if _rule_exists(RULE_HTTP) and _rule_exists(RULE_SSDP):
            log.info("firewall rules ok")
            return True
    except Exception as e:
        log.warning("rule check error: %s", e)

    if _is_admin():
        log.info("admin detected, adding firewall rules directly")
        return _add_rules_direct()

    log.info("requesting UAC to add firewall rules (one-time)")
    bat = _build_bat()
    if not _elevate_run(bat):
        log.warning("user denied UAC; HTTP/SSDP may be blocked")
        try:
            bat.unlink(missing_ok=True)  # type: ignore[arg-type]
        except OSError:
            pass
        return False

    # Internal note.
    for _ in range(20):
        time.sleep(0.3)
        if _rule_exists(RULE_HTTP) and _rule_exists(RULE_SSDP):
            log.info("firewall rules added")
            return True
    log.warning("firewall rule verification timed out (rules may still be applied)")
    return False
