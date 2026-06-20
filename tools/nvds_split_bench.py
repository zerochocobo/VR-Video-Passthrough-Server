"""Microbenchmark: NVDS split vs monolith inference cost (CUDA EP).

Isolates NVDS model time from the full 2DVR pipeline so we can see the backbone
de-duplication effect directly.
"""
from __future__ import annotations

import time

import numpy as np

from utils.runtime_dll_paths import apply_runtime_dll_paths

apply_runtime_dll_paths()

from offline import nvds_stabilizer as n  # noqa: E402  (after DLL paths)

H, W = n.NVDS_HEIGHT, n.NVDS_WIDTH
N_WARM = 6
N_ITER = 40

rng = np.random.default_rng(0)
frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(N_ITER + N_WARM)]
depths = [rng.uniform(1.0, 5.0, (H, W)).astype(np.float32) for _ in range(N_ITER + N_WARM)]


def bench(stab) -> float:
    for i in range(N_WARM):
        stab.stabilize(frames[i], depths[i])
    t0 = time.perf_counter()
    for i in range(N_WARM, N_WARM + N_ITER):
        stab.stabilize(frames[i], depths[i])
    return (time.perf_counter() - t0) * 1000.0 / N_ITER


def bench_sessions_split(stab) -> tuple[float, float]:
    """Time backbone-only and head-only for the split stabilizer."""
    frame4 = stab._preprocess(frames[0], depths[0])
    feats = stab.backbone_session.run(stab._backbone_outputs, {stab._backbone_input: frame4[None]})
    window = [feats] * n.NVDS_SEQUENCE
    feeds = {
        name: np.ascontiguousarray(np.concatenate([window[t][s] for t in range(n.NVDS_SEQUENCE)], axis=0))
        for s, name in enumerate(stab._head_feats)
    }
    feeds[stab._head_last_rgb] = np.ascontiguousarray(frame4[None, 0:3], dtype=np.float32)

    for _ in range(N_WARM):
        stab.backbone_session.run(stab._backbone_outputs, {stab._backbone_input: frame4[None]})
        stab.head_session.run([stab._head_output], feeds)
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        stab.backbone_session.run(stab._backbone_outputs, {stab._backbone_input: frame4[None]})
    bb = (time.perf_counter() - t0) * 1000.0 / N_ITER
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        stab.head_session.run([stab._head_output], feeds)
    hd = (time.perf_counter() - t0) * 1000.0 / N_ITER
    return bb, hd


import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "split"

if mode == "split":
    sp = n.NvdsDepthStabilizer(provider="cuda", split=True)
    print(f"split provider: {sp.providers[0]}")
    sp_ms = bench(sp)
    bb_ms, hd_ms = bench_sessions_split(sp)
    print(f"SPLIT  end-to-end stabilize: {sp_ms:6.1f} ms/frame  ({1000.0/sp_ms:4.1f} fps)")
    print(f"  backbone(1 frame): {bb_ms:6.1f} ms   head(4-frame): {hd_ms:6.1f} ms")
else:
    mo = n.NvdsDepthStabilizer(provider="cuda", split=False)
    print(f"mono provider: {mo.providers[0]}")
    mo_ms = bench(mo)
    print(f"MONO   end-to-end stabilize: {mo_ms:6.1f} ms/frame  ({1000.0/mo_ms:4.1f} fps)")
