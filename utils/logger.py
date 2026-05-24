"""Shared logging setup for console and debug_output/server.log."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_TEE_FILE = None


class _LoggerNameAliasFilter(logging.Filter):
    def __init__(self, aliases: dict[str, str]) -> None:
        super().__init__()
        self.aliases = aliases

    def filter(self, record: logging.LogRecord) -> bool:
        alias = self.aliases.get(record.name)
        if alias:
            record.name = alias
        return True


class _UvicornSocketSendNoiseFilter(logging.Filter):
    _NOISE_MESSAGE = "socket.send() raised exception"

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("uvicorn.") and record.getMessage() == self._NOISE_MESSAGE:
            return False
        return True


def _rotate_server_logs(log_dir: Path, keep: int = 5) -> None:
    """Rotate server.log to server_1.log..server_5.log before a new run."""
    current = log_dir / "server.log"
    if not current.exists():
        return
    last = log_dir / f"server_{keep}.log"
    try:
        if last.exists():
            last.unlink()
        for idx in range(keep - 1, 0, -1):
            src = log_dir / f"server_{idx}.log"
            if src.exists():
                src.replace(log_dir / f"server_{idx + 1}.log")
        current.replace(log_dir / "server_1.log")
    except OSError:
        # Logging must not prevent the server from starting.
        pass


class _Tee:
    def __init__(self, primary, secondary) -> None:
        self.primary = primary
        self.secondary = secondary
        self.encoding = getattr(primary, "encoding", "utf-8")
        self.errors = getattr(primary, "errors", "replace")

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())


def setup(level: int = logging.INFO) -> None:
    """Install stdout and file handlers on the root logger."""
    global _TEE_FILE
    formatter = logging.Formatter(_FMT)
    root = logging.getLogger()
    root.handlers.clear()
    try:
        base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
        log_dir = base / "debug_output"
        log_dir.mkdir(parents=True, exist_ok=True)
        _rotate_server_logs(log_dir)
        _TEE_FILE = open(log_dir / "server.log", "w", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(_ORIGINAL_STDOUT, _TEE_FILE)
        sys.stderr = _Tee(_ORIGINAL_STDERR, _TEE_FILE)
    except OSError:
        pass
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(formatter)
    h.addFilter(_UvicornSocketSendNoiseFilter())
    h.addFilter(_LoggerNameAliasFilter({"uvicorn.error": "uvicorn.server"}))
    root.addHandler(h)
    root.setLevel(level)


def get(name: str) -> logging.Logger:
    """Return a named logger using the shared project configuration."""
    return logging.getLogger(name)


def warmup_event(logger: logging.Logger, **fields: Any) -> None:
    """Write a structured warmup event while keeping the grep-friendly prefix."""
    logger.info(
        "[WARMUP] %s",
        json.dumps(fields, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str),
    )
