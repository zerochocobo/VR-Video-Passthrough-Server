"""Capture a short live passthrough MPEG-TS sample and optionally check A/V sync."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "debug_output" / "live_capture.ts"


def _build_url(host: str, port: int, name: str, start: float, mode: str) -> str:
    path = quote(name.replace("\\", "/"), safe="/")
    return f"http://{host}:{port}/passthrough_live/{path}?t={start:.3f}&mode={mode}"


def _capture(url: str, out: Path, seconds: float, max_bytes: int, user_agent: str) -> tuple[int, int, float]:
    out.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": user_agent, "Accept": "*/*", "Range": "bytes=0-"})
    started = time.monotonic()
    chunks = 0
    written = 0
    with urlopen(req, timeout=20) as resp, out.open("wb") as f:
        while True:
            if seconds > 0 and time.monotonic() - started >= seconds:
                break
            if max_bytes > 0 and written >= max_bytes:
                break
            data = resp.read(256 * 1024)
            if not data:
                break
            if max_bytes > 0 and written + len(data) > max_bytes:
                data = data[: max_bytes - written]
            f.write(data)
            written += len(data)
            chunks += 1
            if max_bytes > 0 and written >= max_bytes:
                break
    return written, chunks, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture live MPEG-TS and optionally run check_mpegts_sync.py.")
    parser.add_argument("--url", default="", help="full passthrough_live URL; overrides --name/--host/--port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--name", default="urvrsp00566_1_8k.mp4", help="video name/path as accepted by /passthrough_live")
    parser.add_argument("--t", type=float, default=180.0, help="start time in seconds")
    parser.add_argument("--mode", choices=("green", "alpha"), default="green")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--max-bytes", type=int, default=120 * 1024 * 1024)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--user-agent", default="SKYBOX/2.0.2")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--max-delta", type=float, default=0.10)
    args = parser.parse_args()

    url = args.url or _build_url(args.host, args.port, args.name, args.t, args.mode)
    print(f"capture url: {url}")
    written, chunks, elapsed = _capture(url, args.out, args.seconds, args.max_bytes, args.user_agent)
    print(f"wrote={written} chunks={chunks} elapsed={elapsed:.3f}s out={args.out}")
    if written <= 0:
        return 2

    if not args.check:
        return 0
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "check_mpegts_sync.py"),
        str(args.out),
        "--max-delta",
        f"{args.max_delta:.6f}",
        "--json",
    ]
    proc = subprocess.run(cmd, text=True, errors="replace")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
