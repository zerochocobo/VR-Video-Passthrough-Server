"""Per-GOP byte budgets inherited from the source MP4 for the frames VMP4 layout.

The frames backend gives every output frame a FIXED byte budget so ``co64``/
``stsz`` can be written before anything is encoded. Until now that budget was one
global constant (``PT_PASSTHROUGH_SEEK_VMP4_FRAMES_FRAME_BYTES``), which wastes
bandwidth on filler during static shots and starves high-motion ones.

The source file already carries a good bit allocation for its own content, so
this module reads the source ``moov`` and turns it into a per-GOP budget: for
each output GOP window, how many bytes the source itself spent on that stretch of
time. Frame sizes and frame counts are NOT copied - only the bit distribution -
so nothing here constrains the encoder to the source's per-frame sizes, its frame
count, or its codec (see ``summary_20260905_SEEK_SOURCE_BUDGET_CN.md``).

Two corrections on top of raw inheritance:

* **Floor + redistribution.** Source black/static stretches can drop to ~2 KiB
  per frame; our composited output is not necessarily static there, so any GOP
  below the floor is raised to it and the deficit is taken proportionally from
  GOPs above the floor. Total bytes are conserved, so ``virtual_size`` still
  lands exactly on the source file size.
* **Exact payload total.** The caller passes the mdat payload size it needs
  (source size minus our own ``ftyp+moov+mdat`` header), and the returned budgets
  sum to it exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.si_virtual_mp4 import read_media_sample_table
from utils.logger import get


log = get("source_budget")


class SourceBudgetError(RuntimeError):
    """The source cannot drive a byte budget (unreadable/too small/degenerate)."""


@dataclass(frozen=True)
class SourceBudgetPlan:
    frame_budgets: tuple[int, ...]
    gop_budgets: tuple[int, ...]
    gop_frames: int
    payload_total: int
    source_video_bytes: int
    source_audio_bytes: int
    source_size: int
    floor_bytes_per_frame: int
    idr_weight: float
    clamped_gops: int
    source_fps: float
    output_fps: float

    @property
    def mean_frame_bytes(self) -> int:
        return self.payload_total // max(1, len(self.frame_budgets))

    def gop_bitrate_bps(self, gop_index: int) -> float:
        """Encoder target for one GOP, in bits/sec at the output frame rate."""
        if not self.gop_budgets or self.output_fps <= 0:
            return 0.0
        idx = min(max(0, int(gop_index)), len(self.gop_budgets) - 1)
        frames = max(1, self.gop_frames)
        return self.gop_budgets[idx] * 8.0 * self.output_fps / frames


def _redistribute(values: list[float], floors: list[float]) -> tuple[list[float], int]:
    """Raise everything below its floor, funding it from what is above, in place.

    Floors are per-entry because the last GOP of a title usually holds fewer
    frames than the rest, and its floor has to scale with that - giving it a full
    GOP's floor is how a 2-frame tail once ended up with a 90 MB frame budget.

    Returns the adjusted values and how many entries were raised. Total is
    conserved. Raises ``SourceBudgetError`` when the floors cannot be met at all.
    """
    if sum(values) <= sum(floors):
        raise SourceBudgetError("source-budget-below-floor")
    raised = sum(1 for v, f in zip(values, floors) if v < f)
    for _ in range(32):
        deficit = sum(f - v for v, f in zip(values, floors) if v < f)
        if deficit <= 0:
            break
        donors = [i for i, v in enumerate(values) if v > floors[i]]
        pool = sum(values[i] - floors[i] for i in donors)
        if pool <= deficit:
            raise SourceBudgetError("source-budget-below-floor")
        for i in donors:
            values[i] -= deficit * (values[i] - floors[i]) / pool
        for i, v in enumerate(values):
            if v < floors[i]:
                values[i] = floors[i]
    return values, raised


def _quantize_to_total(values: list[float], total: int, minimums: list[int]) -> list[int]:
    """Round ``values`` to ints summing to exactly ``total``, each >= its minimum."""
    out = [max(minimums[i], int(v)) for i, v in enumerate(values)]
    drift = total - sum(out)
    if drift > 0:                       # hand the remainder out one byte at a time
        order = sorted(range(len(out)), key=lambda i: values[i], reverse=True)
        for i in range(drift):
            out[order[i % len(order)]] += 1
    elif drift < 0:
        order = sorted(range(len(out)), key=lambda i: out[i], reverse=True)
        need = -drift
        idx = 0
        while need > 0:
            i = order[idx % len(order)]
            take = min(need, out[i] - minimums[i])
            out[i] -= take
            need -= take
            idx += 1
            if idx > len(order) * 64:
                raise SourceBudgetError("source-budget-quantize-stuck")
    return out


def build_source_budget_plan(
    path: Path,
    *,
    payload_total: int,
    frame_count: int,
    gop_frames: int,
    output_fps: float,
    floor_bytes_per_frame: int,
    idr_weight: float = 4.0,
    flatten: float = 1.0,
) -> SourceBudgetPlan:
    """Turn the source MP4's own bit distribution into per-frame byte budgets.

    ``payload_total`` is the exact number of mdat payload bytes to hand out, so
    the virtual file lands on the intended size (normally the source size).
    """
    source = Path(path)
    total = int(payload_total)
    frames = int(frame_count)
    gop = max(1, int(gop_frames))
    fps = float(output_fps or 0.0)
    floor_frame = max(1024, int(floor_bytes_per_frame))
    if frames <= 0:
        raise SourceBudgetError("frame-count-missing")
    if fps <= 0:
        raise SourceBudgetError("fps-missing")
    if total < floor_frame * frames:
        raise SourceBudgetError("source-budget-below-floor")

    try:
        vt = read_media_sample_table(source, "video")
    except Exception as exc:
        raise SourceBudgetError(f"source-table-error:{type(exc).__name__}") from exc
    if not vt.samples:
        raise SourceBudgetError("source-video-samples-missing")
    try:
        at = read_media_sample_table(source, "audio")
        audio_bytes = sum(s.size for s in at.samples)
    except Exception:
        audio_bytes = 0

    video_bytes = sum(s.size for s in vt.samples)
    src_duration = float(vt.duration_seconds or 0.0)
    src_fps = (len(vt.samples) / src_duration) if src_duration > 0 else 0.0

    gop_count = (frames + gop - 1) // gop
    gop_sec = gop / fps
    weights = [0.0] * gop_count
    for s in vt.samples:
        k = int(s.time_seconds / gop_sec)
        if 0 <= k < gop_count:
            weights[k] += float(s.size)
    if sum(weights) <= 0:
        raise SourceBudgetError("source-weights-empty")
    # How many output frames each GOP actually holds; the last one is usually
    # short, and everything downstream (its floor, its share, how its budget is
    # split across frames) has to use this rather than the nominal GOP length.
    gop_sizes = [min(gop, frames - i * gop) for i in range(gop_count)]
    mean_weight = sum(weights) / gop_count
    blend = min(1.0, max(0.0, float(flatten)))
    for i, w in enumerate(weights):
        if w <= 0:                      # source had no samples in this window
            w = mean_weight
        # Flatten toward a uniform plan. The source's shape says what offline
        # multi-pass x265 found cheap, and our realtime encoder does not agree -
        # it costs about the same per frame throughout, so the thin end of the
        # inherited shape starves frames that then have to be truncated. See
        # PT_PASSTHROUGH_SEEK_VMP4_FRAMES_BUDGET_FLATTEN for the measurements.
        if blend > 0.0:
            w = w * (1.0 - blend) + mean_weight * blend
        # A window's weight is what the source spent over its full span, but the
        # last GOP emits fewer frames than that span covers. Scale every window by
        # the frames it actually produces, or the tail claims a whole window's
        # bytes for a couple of frames (a 2-frame tail once took 7.5 MB, and its
        # IDR share of that is a budget NVENC refuses to initialise for).
        weights[i] = w * gop_sizes[i] / gop

    scale = total / sum(weights)
    scaled = [w * scale for w in weights]
    floors = [float(floor_frame * n) for n in gop_sizes]
    adjusted, clamped = _redistribute(scaled, floors)
    gop_budgets = _quantize_to_total(adjusted, total, [floor_frame * n for n in gop_sizes])

    frame_budgets: list[int] = []
    weight = max(1.0, float(idr_weight))
    for gi, gop_budget in enumerate(gop_budgets):
        n = gop_sizes[gi]
        if n <= 0:
            break
        if n == 1:
            frame_budgets.append(gop_budget)
            continue
        # An IDR costs several times a P frame, so splitting the GOP evenly
        # starves exactly the frame a player must decode to show anything. On a
        # quiet 4K stretch that meant a ~52 KiB budget for an IDR that wanted
        # ~200 KiB: clamped, corrupt, and a black screen at the seek point.
        head = int(gop_budget * weight / (weight + n - 1))
        rest = (gop_budget - head) // (n - 1)
        if rest < floor_frame:
            # Don't let the head starve the P frames below the floor. The GOP
            # total already clears floor*n, so the head still clears the floor.
            rest = floor_frame
            head = gop_budget - rest * (n - 1)
        else:
            head = gop_budget - rest * (n - 1)   # remainder rides with the head
        frame_budgets.append(head)
        frame_budgets.extend([rest] * (n - 1))

    if len(frame_budgets) != frames or sum(frame_budgets) != total:
        raise SourceBudgetError(
            f"source-budget-mismatch:frames={len(frame_budgets)}/{frames}:"
            f"bytes={sum(frame_budgets)}/{total}"
        )

    try:
        source_size = source.stat().st_size
    except OSError:
        source_size = 0

    idr_budgets = frame_budgets[::gop] or [0]
    log.info(
        "source budget: %s payload=%d frames=%d gops=%d clamped=%d floor=%dKiB "
        "mean=%dKiB min=%dKiB max=%dKiB idr=%dKiB..%dKiB (w=%.1f) src_video=%.2fMbps",
        source.name, total, frames, gop_count, clamped, floor_frame // 1024,
        (total // frames) // 1024, min(frame_budgets) // 1024, max(frame_budgets) // 1024,
        min(idr_budgets) // 1024, max(idr_budgets) // 1024, weight,
        (video_bytes * 8 / src_duration / 1e6) if src_duration > 0 else 0.0,
    )

    return SourceBudgetPlan(
        frame_budgets=tuple(frame_budgets),
        gop_budgets=tuple(gop_budgets),
        gop_frames=gop,
        payload_total=total,
        source_video_bytes=video_bytes,
        source_audio_bytes=audio_bytes,
        source_size=source_size,
        floor_bytes_per_frame=floor_frame,
        idr_weight=weight,
        clamped_gops=clamped,
        source_fps=src_fps,
        output_fps=fps,
    )
