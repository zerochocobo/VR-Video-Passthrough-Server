"""Replay a player's concurrent range pattern against /passthrough_seek.

The frames backend runs ONE persistent filler (one decoder/encoder/matter setup)
but players open several range connections at different offsets at once. The
2026-06-20 failure was those connections fighting over it: each request's
ensure_filler killed and restarted the single filler at its own offset, so it
rebuilt every few seconds and never produced a frame.

This drives that pattern locally - one sequential playback connection plus header
and tail probes plus scattered seeks, in parallel, repeated - and reports what
each connection got along with the counters that show whether the filler settled:

    uv run python -m tools.vmp4_frames_concurrency_probe videos/movie.mp4 3 conc.log

The number to watch is ``stream begin``: one per genuine reposition is healthy,
one every few seconds with little output is the 2026-06-20 thrash. Clear
``runtime_cache/vmp4_frames/<digest>`` first or the frames are already cached and
nothing has to be encoded.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PT_PASSTHROUGH_SEEK_ENABLED", "1")
os.environ.setdefault("PT_PASSTHROUGH_SEEK_ROUTE_POLICY", "all")
os.environ.setdefault("PT_PASSTHROUGH_SEEK_CONTAINER", "mp4")
os.environ.setdefault("PT_PASSTHROUGH_SEEK_VMP4", "1")
os.environ.setdefault("PT_PASSTHROUGH_SEEK_VMP4_BACKEND", "slot_frames")
os.environ.setdefault("PT_PASSTHROUGH_OUTPUT_MODE", "green")

from utils.runtime_dll_paths import apply_runtime_dll_paths  # noqa: E402

apply_runtime_dll_paths()

SRC = Path(sys.argv[1])
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LOG = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("vmp4_concurrency.log")

handler = logging.FileHandler(LOG, mode="w", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from http_app.routes_media import router  # noqa: E402

app = FastAPI()
app.include_router(router)
UA = {"User-Agent": "nPlayer/3.0"}
size = SRC.stat().st_size
url = f"/passthrough_seek/{SRC.name}.seek.mp4?mode=green"

results: list[tuple] = []
lock = threading.Lock()


def worker(tag: str, rng: str | None, want_bytes: int) -> None:
    client = TestClient(app)
    started = time.time()
    got, status = 0, None
    try:
        headers = {**UA, **({"Range": rng} if rng else {})}
        with client.stream("GET", url, headers=headers) as r:
            status = r.status_code
            if status in (200, 206):
                for chunk in r.iter_bytes(chunk_size=256 * 1024):
                    got += len(chunk)
                    if got >= want_bytes:
                        break
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        status = type(exc).__name__
    with lock:
        results.append((tag, rng or "(none)", status, got, round(time.time() - started, 1)))


# What a player actually does: one sequential read, two structural probes, and
# several scattered seeks, all at the same time.
plan = [
    ("play", None, 4_000_000),
    ("hdr", "bytes=0-65535", 65536),
    ("tail", f"bytes={size - 6608}-", 6608),
    ("seek1", f"bytes={int(size * 0.25)}-", 1_000_000),
    ("seek2", f"bytes={int(size * 0.55)}-", 1_000_000),
    ("seek3", f"bytes={int(size * 0.80)}-", 1_000_000),
]

for rnd in range(ROUNDS):
    threads = [threading.Thread(target=worker, args=(f"r{rnd}:{t}", r, w)) for t, r, w in plan]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"--- round {rnd} done ---", flush=True)

print(f"\n{'tag':12s} {'range':26s} {'status':8s} {'bytes':>10s} {'sec':>6s}")
for tag, rng, status, got, sec in sorted(results):
    print(f"{tag:12s} {rng:26s} {str(status):8s} {got:10d} {sec:6.1f}")

text = LOG.read_text(encoding="utf-8", errors="replace")
print(
    f"\nstream begin={len(re.findall(r'PyNv frame stream begin', text))}"
    f"  probe={len(re.findall(r'frames probe', text))}"
    f"  gop-not-ready={len(re.findall(r'gop not ready', text))}"
    f"  oversize={len(re.findall(r'frames oversize', text))}"
)
