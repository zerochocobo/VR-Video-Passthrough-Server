"""Decide whether a source can drive the /passthrough_seek source-size budget.

The frames VMP4 backend can either hand every frame a flat byte budget
(``PT_PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES``) or inherit the source file's
own per-GOP bit distribution, which makes the virtual file exactly the source's
size. Inheriting only works when the source's bitrate can still fund an
acceptable per-frame budget at our output frame rate, so this tool reports that
per file and says yes/no.

The number that matters is bytes per OUTPUT frame, not bitrate: output fps is
capped (``PT_PASSTHROUGH_MAX_FPS``), so a 60fps source hands each of our frames
roughly twice the bytes it spent on its own.

    uv run python -m tools.seek_source_budget_probe videos/*.mp4
    uv run python -m tools.seek_source_budget_probe --layout videos/2_2.mp4

``--layout`` additionally builds the real frames layout and checks that the
virtual file lands exactly on the source size with a self-consistent sample
table (needs an HEVC source, since it reuses the source ``stsd``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.bitrate_estimator import parse_bitrate  # noqa: E402
from pipeline.si_virtual_mp4 import (  # noqa: E402
    _find_track,
    _iter_boxes,
    _read_top_level_box,
    _require_child,
    _track_stbl,
    read_media_sample_table,
)
from pipeline.source_budget_plan import (  # noqa: E402
    SourceBudgetError,
    build_source_budget_plan,
)


def _pct(values, p: int):
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))]


def _mbps(byts: float, sec: float) -> float:
    return byts * 8.0 / sec / 1e6 if sec > 0 else 0.0


def _dims(moov: bytes) -> tuple[int, int]:
    try:
        stbl = _track_stbl(moov, _find_track(moov, "video").trak)
        stsd = _require_child(moov, stbl, b"stsd")
        for ent in _iter_boxes(moov, stsd.start + 16, stsd.end):
            body = moov[ent.start + 8: ent.end]
            if len(body) >= 28:
                w = int.from_bytes(body[24:26], "big")
                h = int.from_bytes(body[26:28], "big")
                if w and h:
                    return w, h
    except Exception:
        pass
    return 0, 0


def _check_layout(path: Path, duration: float, out_fps: float, plan) -> None:
    from pipeline.passthrough_vmp4_frames import build_passthrough_vmp4_frames_layout

    layout = build_passthrough_vmp4_frames_layout(
        path,
        duration_sec=duration,
        fps=out_fps,
        gop_frames=config.PASSTHROUGH_GOP,
        idr_budget=config.PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES,
        p_budget=config.PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES,
        frame_budgets=plan.frame_budgets,
    )
    size = path.stat().st_size
    print(f"  layout: total={layout.total_size} source={size} delta={layout.total_size - size}")
    print(f"          moov={layout.moov_size} frames={layout.frame_count} "
          f"source_budget={layout.source_budget}")
    ok = True
    if layout.total_size != size:
        print("          !! virtual size != source size")
        ok = False
    cursor = layout.mdat_payload_start
    for fr in layout.frames:
        if fr.offset != cursor:
            print(f"          !! offset gap at frame {fr.index}")
            ok = False
            break
        cursor += fr.budget
    if ok and cursor != layout.total_size:
        print("          !! samples do not fill the mdat")
        ok = False
    print(f"          {'LAYOUT OK' if ok else 'LAYOUT BROKEN'}")


def analyse(path: Path, *, check_layout: bool) -> None:
    size = path.stat().st_size
    moov = _read_top_level_box(path, b"moov")
    vt = read_media_sample_table(path, "video")
    try:
        at = read_media_sample_table(path, "audio")
        abytes = sum(s.size for s in at.samples)
    except Exception:
        abytes = 0

    dur = float(vt.duration_seconds or 0.0)
    n = len(vt.samples)
    src_fps = n / dur if dur > 0 else 0.0
    vbytes = sum(s.size for s in vt.samples)
    max_fps = float(config.PASSTHROUGH_MAX_FPS or 0.0)
    out_fps = min(src_fps, max_fps) if (src_fps and max_fps > 0) else (src_fps or max_fps)
    gop = int(config.PASSTHROUGH_GOP)
    gop_sec = gop / out_fps if out_fps else 0.0
    frame_count = max(1, int(round(dur * out_fps)))
    w, h = _dims(moov)
    px = w * h

    src_bps = vbytes * 8 / dur if dur > 0 else 0.0
    live_bps = min(
        float(parse_bitrate(config.PASSTHROUGH_HEVC_BITRATE)),
        src_bps * float(config.PASSTHROUGH_HEVC_SOURCE_MAX_MULTIPLIER or 0) or float("inf"),
    )
    flat = int(config.PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES)
    floor = int(config.PASSTHROUGH_SEEK_VMP4_FRAMES_FLOOR_BYTES)

    print(f"\n=== {path.name}")
    print(f"  {w}x{h} {vt.codec_name} | {dur / 60:.1f} min | {src_fps:.2f} -> {out_fps:.2f} fps"
          f" | {size / 1e9:.3f} GB")
    print(f"  source video {src_bps / 1e6:7.2f} Mbps"
          + (f"  ({src_bps / (px * src_fps):.4f} bpp/frame)" if px and src_fps else ""))
    print(f"  live path    {live_bps / 1e6:7.2f} Mbps   [min(HEVC_BITRATE, source x MULT)]")

    # Layout init size does not depend on budget values, so a flat probe layout
    # gives the exact payload the plan has to hand out.
    try:
        from pipeline.passthrough_vmp4_frames import build_passthrough_vmp4_frames_layout

        probe = build_passthrough_vmp4_frames_layout(
            path, duration_sec=dur, fps=out_fps, gop_frames=gop,
            idr_budget=flat, p_budget=flat,
        )
        init_size = probe.total_size - probe.mdat_payload_size
    except Exception as exc:
        print(f"  !! cannot build a probe layout ({type(exc).__name__}: {exc});"
              f" assuming a 1 MiB header")
        init_size = 1 << 20

    payload_total = size - init_size
    plan = None
    try:
        plan = build_source_budget_plan(
            path,
            payload_total=payload_total,
            frame_count=frame_count,
            gop_frames=gop,
            output_fps=out_fps,
            floor_bytes_per_frame=floor,
        )
    except SourceBudgetError as exc:
        print(f"  VERDICT: NO - {exc} (stays on the flat {flat // 1024} KiB budget)")

    win = [0] * ((frame_count + gop - 1) // gop)
    for s in vt.samples:
        k = int(s.time_seconds / gop_sec) if gop_sec else 0
        if 0 <= k < len(win):
            win[k] += s.size

    print(f"  per output frame KiB   source-spent -> our budget (floor {floor // 1024})")
    for p in (1, 5, 50, 90):
        raw = _pct(win, p) / gop / 1024
        got = (_pct(list(plan.frame_budgets), p) / 1024) if plan else float("nan")
        print(f"      p{p:<3d} {raw:8.1f}  ->  {got:8.1f}")
    if plan:
        mean = plan.mean_frame_bytes
        print(f"  mean {mean / 1024:.1f} KiB/frame vs flat {flat / 1024:.1f} KiB/frame"
              f"  ({100.0 * mean / flat:.0f}%)")
        print(f"  clamped GOPs {plan.clamped_gops}/{len(plan.gop_budgets)}"
              f" = {100.0 * plan.clamped_gops / max(1, len(plan.gop_budgets)):.1f}%")
        src_per_frame = vbytes / n if n else 0
        ratio = mean / src_per_frame if src_per_frame else 0
        print(f"  VERDICT: YES - {ratio:.2f}x the source's own bytes per frame"
              + ("" if ratio >= 1.5 else "   (thin: equal-fps source, no headroom for realtime NVENC)"))
        if check_layout:
            _check_layout(path, dur, out_fps, plan)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--layout", action="store_true",
                    help="also build the real frames layout and verify it")
    args = ap.parse_args(argv)
    for p in args.paths:
        if not p.is_file():
            print(f"!! not a file: {p}")
            continue
        try:
            analyse(p, check_layout=args.layout)
        except Exception as exc:
            print(f"\n=== {p.name}\n  !! {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
