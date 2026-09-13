"""Measure what DLSS 5 Neural Rendering actually does to a frame.

"I cannot see much improvement" is not something more parameter tweaking can
answer, because there are four places the enhancement can be lost and they are
indistinguishable by eye:

  1. NR barely changes the picture at these settings
  2. NR changes it, but our CUDA colour conversion undoes part of it
  3. NR changes it, but shimmer suppression flattens it back out
  4. NR changes it, and the P1 encode at the seek budget throws it away

This probe answers (1) and (3) directly, on the host path, with no decode
kernels, no encoder and no virtual file in the way: decode a run of consecutive
frames with ffmpeg, hand the pixels to the same engine the server uses, and
measure what came back. What it reports is the ceiling - the most the pipeline could
possibly deliver. If the numbers here are small, no amount of encoder tuning
will help; if they are large and the picture still looks flat, the loss is
downstream and the next place to look is the encoder.

Usage:
    python tools/dlss5_probe.py VIDEO [--time 120] [--frames 4] [--sweep]
                                      [--out DIR] [--width N] [--crop eye]

    --sweep runs a set of parameter variants over the same frames and prints one
    row each, so the contribution of a single control is visible rather than
    inferred. Without it, only the settings config.py currently carries are run.

    --frames are CONSECUTIVE frames run as one NR session, because shimmer
    suppression only exists between neighbouring frames. --crop eye measures the
    middle of an SBS VR left eye instead of a frame that is mostly periphery.

Exports before/after/amplified-difference PNGs next to the numbers, because
"where did it change" is a question a number cannot answer.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.dlss5 import DLSS5Settings, is_dlss5_available
from utils.subprocess_hidden import hidden_subprocess_kwargs

# (label, field overrides). The first row is whatever config.py says, so a run
# always shows the settings the server would actually use alongside the rest.
SWEEP: tuple[tuple[str, dict], ...] = (
    ("current (config.py)", {}),
    ("shimmer 0.00", {"shimmer_suppression": 0.0}),
    ("shimmer 0.35", {"shimmer_suppression": 0.35}),
    ("detail-only", {"color_strength": 0.0, "tone_preservation": 1.0}),
    ("detail-only + shimmer 0", {"color_strength": 0.0, "tone_preservation": 1.0,
                                 "shimmer_suppression": 0.0}),
    ("intensity 1.50", {"intensity": 1.5}),
    ("intensity 2.00", {"intensity": 2.0}),
    ("passes 2", {"nr_passes": 2}),
    ("passes 4", {"nr_passes": 4}),
    ("structure 0.50", {"local_structure": 0.5}),
    ("structure 2.00", {"local_structure": 2.0}),
    ("face/skin protect 0.70", {"face_skin_protection": 0.7}),
    # Denoising is the part that costs high-frequency energy, and grain
    # preservation is the only control that asks for some of it back.
    ("grain 1.00", {"grain_preservation": 1.0}),
    ("structure 2 + grain 1", {"local_structure": 2.0, "grain_preservation": 1.0}),
    # The styles are separate models, not presets over one: worth a row each.
    ("style natural", {"style": 1}),
    ("style cinematic", {"style": 2}),
)


def _decode_frames(
    src: Path, at_seconds: float, count: int, width: int = 0, crop: str = ""
) -> list[np.ndarray]:
    """``count`` CONSECUTIVE frames as float32 [0,1] RGB, via one ffmpeg run.

    Consecutive matters: shimmer suppression stabilises each frame against the
    one before it, so frames sampled a second apart hand it two unrelated
    pictures to reconcile and it flattens both. That is a property of the
    sampling, not of the setting, and reading it as the setting's cost is how a
    probe lies to you.

    PNG files rather than a rawvideo pipe: each frame carries its own
    dimensions, so a scale or crop filter cannot silently disagree with what we
    reshape to.
    """
    from PIL import Image

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    filters = [f for f in (crop, f"scale={int(width)}:-2" if width > 0 else "") if f]
    with tempfile.TemporaryDirectory(prefix="dlss5_probe_") as work:
        pattern = str(Path(work) / "f_%04d.png")
        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-v", "error", "-ss", str(max(0.0, at_seconds)),
             "-i", str(src), "-frames:v", str(max(1, int(count))),
             *(["-vf", ",".join(filters)] if filters else []),
             "-y", pattern],
            capture_output=True, check=False, **hidden_subprocess_kwargs(),
        )
        files = sorted(Path(work).glob("f_*.png"))
        if probe.returncode != 0 or not files:
            detail = probe.stderr.decode("utf-8", "replace").strip().splitlines()
            raise SystemExit(
                f"ffmpeg could not decode frames: {detail[-1] if detail else 'no output'}"
            )
        frames = []
        for path in files:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.append(np.ascontiguousarray(pixels.astype(np.float32) / 255.0))
    return frames


def _laplacian_energy(rgb: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian on luma: a plain sharpness proxy."""
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lap = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
    )
    return float(lap.var())


def _measure(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    delta = after - before
    energy_before = _laplacian_energy(before)
    energy_after = _laplacian_energy(after)
    return {
        # Mean absolute change in 8-bit levels. Under ~1 is invisible; a real
        # enhancement on a soft source runs several levels.
        "mad": float(np.abs(delta).mean()) * 255.0,
        "p99": float(np.percentile(np.abs(delta), 99)) * 255.0,
        "max": float(np.abs(delta).max()) * 255.0,
        # >1 means the frame carries more high-frequency energy than it did.
        "sharpness": (energy_after / energy_before) if energy_before > 0 else 0.0,
        # Mean signed luma change: negative is a picture that got darker.
        "luma": float((after.mean(axis=2) - before.mean(axis=2)).mean()) * 255.0,
    }


def _save(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8)).save(path)


def _save_difference(path: Path, before: np.ndarray, after: np.ndarray, gain: float = 8.0) -> None:
    """The change alone, amplified around mid-grey so it can be seen at all."""
    _save(path, np.clip((after - before) * gain + 0.5, 0.0, 1.0))


def _run_sequence(frames: list[np.ndarray], settings: DLSS5Settings) -> list[np.ndarray]:
    """Run consecutive frames through NR as one session, not as single images.

    Shimmer suppression is temporal - it stabilises this frame's enhancement
    against the previous one's - so on a single reset frame it does nothing and
    a sweep row for it would read like every other row. Only the first frame is
    a cut, which is how the realtime stage drives it too.
    """
    from models.dlss5.neural_bridge import BRIDGE

    results: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        out = np.empty_like(frame)
        BRIDGE.process_host(frame, out, settings.to_params(reset=(index == 0)))
        results.append(out)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--time", type=float, default=60.0, help="seconds into the title")
    parser.add_argument("--frames", type=int, default=4,
                        help="consecutive frames run as one session; the last one is measured")
    parser.add_argument("--width", type=int, default=0, help="downscale before NR (0 = source)")
    parser.add_argument("--crop", default="",
                        help='ffmpeg crop, or "eye" for the middle of an SBS VR left eye. '
                             "A VR frame is mostly stretched periphery, and measuring that "
                             "dilutes whatever happened where the viewer is looking.")
    parser.add_argument("--sweep", action="store_true", help="run the parameter variants")
    parser.add_argument("--out", default="", help="directory for the PNGs (default: no PNGs)")
    args = parser.parse_args(argv)

    # "eye": the centre half of the left eye of a side-by-side pair.
    crop = "crop=iw/4:ih/2:iw/8:ih/4" if str(args.crop).strip().lower() == "eye" else str(args.crop).strip()

    if not is_dlss5_available():
        raise SystemExit("the DLSS5 runtime is not installed under models/dlss5/runtime")
    src = Path(args.video)
    if not src.is_file():
        raise SystemExit(f"no such file: {src}")

    out_dir = Path(args.out).resolve() if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    base = DLSS5Settings.from_config()
    variants = SWEEP if args.sweep else (("current (config.py)", {}),)

    print(f"source : {src.name}")
    print(f"base   : style={base.style} intensity={base.intensity:.2f} passes={base.nr_passes} "
          f"structure={base.local_structure:.2f} tone={base.local_tone:.2f} "
          f"colour={base.color_strength:.2f} tone_preserve={base.tone_preservation:.2f} "
          f"shimmer={base.shimmer_suppression:.2f}")
    print()

    count = max(1, int(args.frames))
    frames = _decode_frames(src, float(args.time), count, args.width, crop)
    if len(frames) < count:
        print(f"note: only {len(frames)} frame(s) available at t={args.time:.1f}s")
        count = len(frames)
    h, w, _ = frames[0].shape
    reported = count - 1        # the last one: the frame that has history behind it
    at = float(args.time)

    region = f", crop={crop}" if crop else ""
    print(f"--- {w}x{h}{region}, {count} consecutive frame(s) from t={at:.1f}s, "
          f"measured on the last ---")
    print(f"{'variant':<26} {'mad':>7} {'p99':>7} {'max':>7} {'sharp':>7} {'luma':>7}")
    if out_dir is not None:
        _save(out_dir / f"t{at:.0f}_before.png", frames[reported])
    for label, overrides in variants:
        settings = replace(base, **overrides) if overrides else base
        after = _run_sequence(frames, settings)[reported]
        m = _measure(frames[reported], after)
        print(f"{label:<26} {m['mad']:7.2f} {m['p99']:7.2f} {m['max']:7.2f} "
              f"{m['sharpness']:7.3f} {m['luma']:+7.2f}")
        if out_dir is not None:
            tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in label).strip("_")
            _save(out_dir / f"t{at:.0f}_{tag}.png", after)
            _save_difference(out_dir / f"t{at:.0f}_{tag}_diff.png", frames[reported], after)
    print()

    print("mad   mean absolute change, 8-bit levels. under ~1 is invisible.")
    print("p99   the change at the 99th percentile - where the work actually lands.")
    print("sharp high-frequency energy after / before. >1 sharper, <1 softer.")
    print("luma  mean brightness change. negative is a picture NR darkened.")
    if count < 2:
        print("")
        print("note: --frames 1 gives shimmer suppression no history to work against,")
        print("      so its rows read like every other. Use --frames 4 or more.")
    if out_dir is not None:
        print(f"\nPNGs in {out_dir} (the _diff ones are the change alone, 8x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
