from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "debug_output" / "ui_startup.log"
_LOCK = threading.Lock()
_MAX_BYTES = 1024 * 1024


def log_startup_event(event: str, **fields: Any) -> None:
    """Append one UI startup diagnostic event.

    This is intentionally tiny and dependency-free so it can be called from UI
    code and poller worker threads without coupling startup diagnostics to the
    main logging setup.
    """
    payload = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "event": event,
        **fields,
    }
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        with _LOCK:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if LOG_PATH.exists() and LOG_PATH.stat().st_size > _MAX_BYTES:
                LOG_PATH.replace(LOG_PATH.with_suffix(".log.1"))
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
    except Exception:
        pass
