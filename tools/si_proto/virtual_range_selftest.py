"""Self-test: reconstruct a real fMP4 from VirtualRegions and prove that
iter_virtual_range is byte-stable and consistent with the whole file.

This closes the loop the prototype only hand-checked: it builds a mixed
memory(init segment) + file(fragments) VirtualRegion layout over an actual
fragmented MP4, then asserts that hundreds of random Range slices
  (a) equal the corresponding bytes of the whole file, and
  (b) are identical when sliced again (byte stability).

A tiny synthetic fMP4 is generated with ffmpeg (no large asset needed).

Run:
    .venv/Scripts/python.exe tools/si_proto/virtual_range_selftest.py
"""
from __future__ import annotations

import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.si_virtual_mp4 import (  # noqa: E402
    BoxSplittingSink,
    VirtualRegion,
    iter_virtual_range,
    virtual_size,
)


def make_synthetic_fmp4(out: Path) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+empty_moov+default_base_moof+frag_keyframe",
        "-f", "mp4", str(out),
    ]
    subprocess.run(cmd, check=True)


def build_regions(path: Path) -> tuple[list[VirtualRegion], bytes]:
    """Map the real fMP4 into: 1 memory region (ftyp+moov) + N file regions."""
    data = path.read_bytes()
    sink = BoxSplittingSink()
    sink.write(data)  # reuse the module's top-level box parser
    boxes = sink.boxes
    if not boxes:
        raise RuntimeError("no top-level boxes parsed")

    init_end = 0
    for box in boxes:
        if box.type in ("ftyp", "moov"):
            init_end = box.end
        else:
            break

    regions: list[VirtualRegion] = [VirtualRegion.memory(0, data[:init_end])]
    cursor = init_end
    for box in boxes:
        if box.start < init_end:
            continue
        regions.append(VirtualRegion.file(cursor, path, box.start, box.size))
        cursor += box.size
    return regions, data


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fmp4 = Path(tmp) / "sample_fmp4.mp4"
        make_synthetic_fmp4(fmp4)
        regions, ground = build_regions(fmp4)

        if virtual_size(regions) != len(ground):
            print(f"FAIL: virtual_size {virtual_size(regions)} != file {len(ground)}")
            return 1
        memory_regions = sum(1 for r in regions if r.kind == "memory")
        file_regions = sum(1 for r in regions if r.kind == "file")
        print(f"fMP4={len(ground)}B  regions={len(regions)} "
              f"({memory_regions} memory init + {file_regions} file fragments)")

        rng = random.Random(20260617)
        trials = 500
        for _ in range(trials):
            start = rng.randint(0, len(ground) - 1)
            end = rng.randint(start, len(ground) - 1)
            chunk_size = rng.choice([1, 3, 64, 65536])
            sliced = b"".join(iter_virtual_range(regions, start, end, chunk_size=chunk_size))
            if sliced != ground[start:end + 1]:
                print(f"FAIL consistency at [{start},{end}] chunk={chunk_size}")
                return 1
            again = b"".join(iter_virtual_range(regions, start, end, chunk_size=64 * 1024))
            if sliced != again:
                print(f"FAIL stability at [{start},{end}]")
                return 1

        whole = b"".join(iter_virtual_range(regions, 0, len(ground) - 1))
        if whole != ground:
            print("FAIL whole-file reconstruct")
            return 1

        print(f"PASS: {trials} random ranges consistent + stable; "
              f"whole-file reconstruct byte-identical ({len(ground):,}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
