"""PyAV feasibility prototype for SI virtual remux MP4 (Phase 1 de-risking).

This is a *throwaway research prototype*, not production code. It validates the
core claims behind the "virtual remux" approach (research summary 2026-06-17 §8/§9):

  1. The source video sample table (byte offset, size, pts/dts, keyframe) can be
     read with PyAV without re-encoding -> video samples come "for free".
  2. A source video sample's bytes == source_file[pos : pos+size]  -> we can
     splice video bytes straight from the source file by byte offset (no need to
     materialise the whole remuxed file = avoids the 8 GB disk cost).
  3. PyAV can mux video(copy) + mixed audio into a fragmented MP4 written to a
     Python file-like, and we can intercept top-level box boundaries
     (ftyp/moov = init segment; moof/mdat = fragments).
  4. The generated fMP4 is byte-deterministic across runs (prerequisite for a
     stable virtual file / stable Content-Length / stable ETag).

Run:
    .venv/Scripts/python.exe tools/si_proto/pyav_prototype.py videos/SI_TEST_8K.mp4 videos/SI_TEST_8K.si.wav
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import av


# ---------------------------------------------------------------------------
# 1 + 2: source video sample table + byte-splice identity
# ---------------------------------------------------------------------------
def probe_video_sample_table(src: Path, limit: int = 12) -> list[dict]:
    rows: list[dict] = []
    with av.open(str(src)) as container:
        vstream = container.streams.video[0]
        print(f"[probe] video: codec={vstream.codec_context.name} "
              f"{vstream.codec_context.width}x{vstream.codec_context.height} "
              f"time_base={vstream.time_base} extradata={len(vstream.codec_context.extradata or b'')}B (hvcC)")
        for i, packet in enumerate(container.demux(vstream)):
            if packet.size == 0:
                continue
            rows.append({
                "i": i,
                "pos": packet.pos,
                "size": packet.size,
                "pts": packet.pts,
                "dts": packet.dts,
                "key": bool(packet.is_keyframe),
            })
            if len(rows) >= limit:
                break
    return rows


def verify_byte_splice_identity(src: Path, rows: list[dict]) -> bool:
    """source_file[pos:pos+size] must equal the demuxed packet bytes."""
    ok = True
    with av.open(str(src)) as container, open(src, "rb") as raw:
        vstream = container.streams.video[0]
        want = {r["pos"]: r["size"] for r in rows if r["pos"] is not None}
        checked = 0
        for packet in container.demux(vstream):
            if packet.pos in want and packet.size == want[packet.pos]:
                raw.seek(packet.pos)
                file_bytes = raw.read(packet.size)
                pkt_bytes = bytes(packet)
                same = file_bytes == pkt_bytes
                ok = ok and same
                checked += 1
                if checked <= 3:
                    print(f"[splice] pos={packet.pos} size={packet.size} "
                          f"file==packet: {same}")
                if checked >= len(want):
                    break
    return ok


# ---------------------------------------------------------------------------
# helper: pre-generate the mixed audio sidecar with the real SI filter
# ---------------------------------------------------------------------------
def make_mixed_audio(src: Path, si_wav: Path, seconds: int, out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.si_filter import build_si_mix_filter

    filt = build_si_mix_filter("both", 100, 100, 1.0, True)
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y", "-t", str(seconds),
        "-i", str(src), "-i", str(si_wav),
        "-filter_complex", filt, "-map", "[si_track]",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        str(out),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 3: mux video(copy)+audio into fMP4 to a file-like + intercept box boundaries
# ---------------------------------------------------------------------------
class BoxSplittingSink(io.RawIOBase):
    """Captures all bytes written by libav and records top-level ISO BMFF boxes.

    fMP4 with empty_moov is written sequentially; we still implement seek/tell
    because libav may probe position.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.boxes: list[tuple[str, int, int]] = []  # (type, start, size)
        self._scan_cursor = 0

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        self.buf.extend(b)
        self._scan()
        return len(b)

    def seekable(self) -> bool:
        return True

    def seek(self, offset, whence=io.SEEK_SET) -> int:
        # empty_moov path is append-only; report end so libav stays happy.
        return len(self.buf)

    def tell(self) -> int:
        return len(self.buf)

    def _scan(self) -> None:
        while self._scan_cursor + 8 <= len(self.buf):
            off = self._scan_cursor
            size = int.from_bytes(self.buf[off:off + 4], "big")
            btype = bytes(self.buf[off + 4:off + 8]).decode("latin1", "replace")
            header = 8
            if size == 1:  # 64-bit size
                if off + 16 > len(self.buf):
                    break
                size = int.from_bytes(self.buf[off + 8:off + 16], "big")
                header = 16
            if size < header:
                break
            if off + size > len(self.buf):
                break  # box not fully written yet
            self.boxes.append((btype, off, size))
            self._scan_cursor = off + size


def build_fmp4_via_pyav(src: Path, audio: Path, seconds: int, sink: BoxSplittingSink) -> None:
    options = {"movflags": "empty_moov+default_base_moof+frag_keyframe"}
    out = av.open(sink, "w", format="mp4", options=options)
    in_v = av.open(str(src))
    in_a = av.open(str(audio))
    vin = in_v.streams.video[0]
    ain = in_a.streams.audio[0]
    vout = out.add_stream_from_template(vin)
    aout = out.add_stream_from_template(ain)

    cutoff = float(seconds)
    # interleave by dts time; simple two-queue merge is overkill for a prototype,
    # so write all video then all audio packets capped at `seconds`.
    for packet in in_v.demux(vin):
        if packet.dts is None:
            continue
        if float(packet.pts or packet.dts) * float(vin.time_base) > cutoff:
            break
        packet.stream = vout
        out.mux(packet)
    for packet in in_a.demux(ain):
        if packet.dts is None:
            continue
        if float(packet.pts or packet.dts) * float(ain.time_base) > cutoff:
            break
        packet.stream = aout
        out.mux(packet)
    out.close()
    in_v.close()
    in_a.close()


def summarize_boxes(sink: BoxSplittingSink) -> None:
    top = sink.boxes
    init_end = 0
    for btype, start, size in top:
        if btype in ("ftyp", "moov"):
            init_end = start + size
    frags = [b for b in top if b[0] in ("moof", "mdat", "styp", "sidx")]
    order = " ".join(b[0] for b in top[:14])
    print(f"[fmp4] total={len(sink.buf)}B  top_boxes={len(top)}  init_segment={init_end}B")
    print(f"[fmp4] first boxes: {order}")
    moofs = [b for b in top if b[0] == "moof"]
    print(f"[fmp4] fragments(moof)={len(moofs)}")
    for btype, start, size in top[:8]:
        print(f"         {btype} @{start} ({size}B)")


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "videos/SI_TEST_8K.mp4")
    si_wav = Path(sys.argv[2] if len(sys.argv) > 2 else "videos/SI_TEST_8K.si.wav")
    seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    print(f"=== source: {src.name} ({src.stat().st_size:,}B), si={si_wav.name}, clip={seconds}s ===\n")

    print("--- (1) video sample table via PyAV ---")
    rows = probe_video_sample_table(src, limit=10)
    for r in rows[:6]:
        print(f"   sample[{r['i']:>3}] pos={r['pos']:>12} size={r['size']:>9} "
              f"pts={r['pts']} key={r['key']}")
    print()

    print("--- (2) byte-splice identity (source_file[pos:size] == packet bytes) ---")
    identical = verify_byte_splice_identity(src, rows)
    print(f"   => video bytes are splice-able straight from source file: {identical}\n")

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "mixed.m4a"
        print("--- pre-generating mixed audio sidecar (real SI filter) ---")
        make_mixed_audio(src, si_wav, seconds, audio)
        print(f"   mixed audio: {audio.stat().st_size:,}B\n")

        print("--- (3) mux video(copy)+audio -> fMP4 via PyAV + intercept boxes ---")
        sink1 = BoxSplittingSink()
        build_fmp4_via_pyav(src, audio, seconds, sink1)
        summarize_boxes(sink1)
        h1 = hashlib.sha256(sink1.buf).hexdigest()
        print(f"   sha256(run1)={h1[:24]}...\n")

        print("--- (4) determinism: second identical run must be byte-identical ---")
        sink2 = BoxSplittingSink()
        build_fmp4_via_pyav(src, audio, seconds, sink2)
        h2 = hashlib.sha256(sink2.buf).hexdigest()
        print(f"   sha256(run2)={h2[:24]}...")
        print(f"   => deterministic (run1==run2): {h1 == h2}, same_size={len(sink1.buf)==len(sink2.buf)}\n")

        # round-trip validate the produced fMP4
        outp = Path("debug_output/si_proto/pyav_fmp4_out.mp4")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_bytes(sink1.buf)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
             "-of", "csv=p=0", str(outp)],
            capture_output=True, text=True,
        )
        print(f"--- round-trip ffprobe of PyAV output ---\n   {probe.stdout.strip().replace(chr(10), ' | ')}")

    print("\n=== prototype done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
