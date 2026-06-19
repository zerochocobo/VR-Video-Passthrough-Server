"""Minimal frozen-runtime probe for PyAV.

This script is intentionally small: it validates that a PyInstaller onedir
build can import PyAV and use its bundled libav DLLs to open an external media
file. Run it before starting SI virtual-MP4 production work.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import av


def main() -> int:
    media = Path(sys.argv[1] if len(sys.argv) > 1 else "debug_output/si_proto/B_fmp4.mp4")
    if not media.is_file():
        print(json.dumps({"ok": False, "error": f"missing media: {media}"}, ensure_ascii=False))
        return 2

    rows: list[dict[str, object]] = []
    with av.open(str(media)) as container:
        for stream in container.streams:
            ctx = stream.codec_context
            rows.append(
                {
                    "type": stream.type,
                    "codec": ctx.name,
                    "width": getattr(ctx, "width", 0),
                    "height": getattr(ctx, "height", 0),
                    "time_base": str(stream.time_base),
                }
            )

    print(
        json.dumps(
            {
                "ok": True,
                "av_version": av.__version__,
                "media": str(media),
                "streams": rows,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
