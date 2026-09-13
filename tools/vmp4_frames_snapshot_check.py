"""Materialise the frames VMP4 as a real file and check it decodes and seeks.

The frames backend serves a virtual MP4 that only ever exists as HTTP ranges, so
a decode failure on a headset is hard to attribute: container, content, realtime
generation and player behaviour all fail the same way. This tool freezes the
container half of that: it rebuilds the exact layout the route would serve, fills
it from the per-frame cache the filler already wrote, writes the whole thing to
one file, and runs ffprobe + a decode + a mid-file seek-and-decode over it.

A snapshot that decodes and seeks here is a plain, byte-stable MP4 - so anything
that still fails on a device is generation or player behaviour, not the layout.
Drop the snapshot into a media directory to test it on the headset over the
static /media path, with realtime out of the picture.

    uv run python -m tools.vmp4_frames_snapshot_check videos/movie.mp4 <digest> out.mp4

``<digest>`` is the ``X-Passthrough-VMP4-Cache`` response header (also the
directory name under runtime_cache/vmp4_frames). Frames not yet built are filled
with the layout's filler, so a partial cache still produces a valid file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.passthrough_vmp4_frames import iter_vmp4_frames_range  # noqa: E402
from pipeline.ffmpeg_io import probe_cached  # noqa: E402
from http_app import routes_media as rm  # noqa: E402

SRC = Path(sys.argv[1])
DIGEST = sys.argv[2]
OUT = Path(sys.argv[3])
cache = Path("runtime_cache/vmp4_frames") / DIGEST

info = probe_cached(SRC)
mode = "green"
tpl = rm._vmp4_slot_output_template(info, mode)
layout = rm._vmp4_frames_cached_layout(SRC, info, mode, tpl)
digest = rm._vmp4_frames_digest(SRC, layout, mode)
print(f"layout total={layout.total_size} source={SRC.stat().st_size} frames={layout.frame_count} "
      f"{tpl.width}x{tpl.height} source_budget={layout.source_budget}")
print(f"digest={digest}  (expected {DIGEST})  match={digest == DIGEST}")

ready = {}
for p in cache.glob("frame_*.bin"):
    ready[int(p.stem[6:])] = p
print(f"frames built {len(ready)}/{layout.frame_count}")

n = 0
with OUT.open("wb") as fh:
    for chunk in iter_vmp4_frames_range(layout, 0, layout.total_size - 1,
                                        chunk_size=1 << 20, payload_paths=ready):
        fh.write(chunk); n += len(chunk)
print(f"wrote {OUT} {n} bytes  match={n == layout.total_size}")

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")

rc, out = run(["ffprobe", "-v", "error", "-show_entries",
               "stream=codec_name,width,height,nb_frames,avg_frame_rate:format=duration",
               "-of", "default=nw=1", str(OUT)])
print(f"\nffprobe rc={rc}\n{out.strip()}")

rc, out = run(["ffmpeg", "-v", "error", "-i", str(OUT), "-frames:v", "600", "-f", "null", "-"])
print(f"\nffmpeg decode 600 frames rc={rc}")
print((out.strip()[:1500]) if out.strip() else "  (无错误输出)")

# seek 到中段再解码, 模拟播放器拖动
rc, out = run(["ffmpeg", "-v", "error", "-ss", "40", "-i", str(OUT), "-frames:v", "120", "-f", "null", "-"])
print(f"\nffmpeg seek 40s + decode 120 rc={rc}")
print((out.strip()[:1200]) if out.strip() else "  (无错误输出)")
