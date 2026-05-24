"""Estimate slate-induced A/V PTS lead from PTServer logs.

This is a lightweight guard for the live MPEG-TS slate path. It reads
``slate video begin`` and ``slate video end`` lines, estimates
``frames / fps - elapsed``, and fails when the lead is above a threshold.
When a playback bypasses slate because the AAC cache is already available, the
tool reports that explicitly instead of treating it as a failed slate sample.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "debug_output" / "server.log"

BEGIN_RE = re.compile(
    r"\[PYNV\]\[(?P<sid>\d+)\] slate video begin: .* burst_frames=(?P<burst>\d+) fps=(?P<fps>\d+(?:\.\d+)?)"
)
END_RE = re.compile(
    r"\[PYNV\]\[(?P<sid>\d+)\] slate video end: frames=(?P<frames>\d+) .* elapsed=(?P<elapsed>\d+(?:\.\d+)?)s"
)
WORKER_RE = re.compile(r"\[PYNV\]\[(?P<sid>\d+)\] worker start .* start=(?P<start>\d+(?:\.\d+)?)")
RUNTIME_RE = re.compile(r"\[PYNV\]\[(?P<sid>\d+)\] runtime config: .* output_fps=(?P<fps>\d+(?:\.\d+)?)")
REQUEST_RE = re.compile(r"passthrough_live\[(?P<rid>\d+)\] start: (?P<name>.+?) @ (?P<start>\d+(?:\.\d+)?)s")
FIRST_WRITE_RE = re.compile(r"\[PYNV\]\[(?P<sid>\d+)\] first real video bitstream written")
CACHE_HIT_RE = re.compile(r"\[PYNV\]\[(?P<sid>\d+)\] audio cache hit: (?P<cache>\S+)")
SLATE_SERVER_RE = re.compile(r"\[PYNV\]\[(?P<sid>\d+)\] slate audio server listening:")
SOURCE_AUDIO_RE = re.compile(
    r"\[PYNV\]\[(?P<sid>\d+)\] audio cache disabled; using source audio in pipe_ts final mux"
)


def _session(sessions: dict[int, dict[str, float | int | str | bool]], sid: int) -> dict[str, float | int | str | bool]:
    return sessions.setdefault(
        sid,
        {
            "sid": sid,
            "start_sec": 0.0,
            "fps": 0.0,
            "audio_cache_hit": False,
            "source_audio": False,
            "slate_audio_server": False,
            "slate_video": False,
            "first_real_video": False,
        },
    )


def analyze_log(path: Path, fallback_fps: float) -> dict[str, object]:
    active: dict[int, tuple[int, float]] = {}
    sessions: dict[int, dict[str, float | int | str | bool]] = {}
    latest_request: dict[str, float | int | str] | None = None
    records: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            request = REQUEST_RE.search(line)
            if request:
                latest_request = {
                    "rid": int(request.group("rid")),
                    "name": request.group("name"),
                    "start_sec": float(request.group("start")),
                }
                continue
            worker = WORKER_RE.search(line)
            if worker:
                sid = int(worker.group("sid"))
                session = _session(sessions, sid)
                session["start_sec"] = float(worker.group("start"))
                if latest_request is not None:
                    session["request_name"] = str(latest_request["name"])
                continue
            runtime = RUNTIME_RE.search(line)
            if runtime:
                sid = int(runtime.group("sid"))
                _session(sessions, sid)["fps"] = float(runtime.group("fps"))
                continue
            cache_hit = CACHE_HIT_RE.search(line)
            if cache_hit:
                sid = int(cache_hit.group("sid"))
                session = _session(sessions, sid)
                session["audio_cache_hit"] = True
                session["cache"] = cache_hit.group("cache")
                continue
            slate_server = SLATE_SERVER_RE.search(line)
            if slate_server:
                _session(sessions, int(slate_server.group("sid")))["slate_audio_server"] = True
                continue
            source_audio = SOURCE_AUDIO_RE.search(line)
            if source_audio:
                _session(sessions, int(source_audio.group("sid")))["source_audio"] = True
                continue
            first_write = FIRST_WRITE_RE.search(line)
            if first_write:
                _session(sessions, int(first_write.group("sid")))["first_real_video"] = True
                continue
            begin = BEGIN_RE.search(line)
            if begin:
                sid = int(begin.group("sid"))
                _session(sessions, sid)["slate_video"] = True
                active[sid] = (int(begin.group("burst")), float(begin.group("fps")))
                continue
            end = END_RE.search(line)
            if not end:
                continue
            sid = int(end.group("sid"))
            burst, fps = active.get(sid, (0, fallback_fps))
            frames = int(end.group("frames"))
            elapsed = float(end.group("elapsed"))
            video_pts = frames / fps if fps > 0 else 0.0
            lead = max(0.0, video_pts - elapsed)
            records.append(
                {
                    "sid": sid,
                    "frames": frames,
                    "fps": fps,
                    "burst_frames": burst,
                    "elapsed_sec": elapsed,
                    "video_pts_sec": video_pts,
                    "estimated_video_lead_sec": lead,
                }
            )
            active.pop(sid, None)
    return {"records": records, "sessions": list(sessions.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check slate-induced A/V sync risk from server.log.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--fps", type=float, default=30.0, help="fallback FPS for old logs without begin-line FPS")
    parser.add_argument("--max-lead", type=float, default=0.10, help="allowed video lead in seconds")
    parser.add_argument("--json", action="store_true", help="print full JSON records")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 2
    result = analyze_log(args.log, args.fps)
    records = list(result["records"])
    sessions = list(result["sessions"])
    if not records:
        slate_sessions = [s for s in sessions if bool(s.get("slate_audio_server")) or bool(s.get("slate_video"))]
        cache_hit_sessions = [s for s in sessions if bool(s.get("audio_cache_hit"))]
        source_audio_sessions = [s for s in sessions if bool(s.get("source_audio"))]
        if args.json:
            print(json.dumps({"records": records, "sessions": sessions}, indent=2, ensure_ascii=False))
        if slate_sessions:
            print("slate audio was used, but no slate video begin/end records were found", file=sys.stderr)
            return 2
        if cache_hit_sessions:
            last = cache_hit_sessions[-1]
            print(
                "no slate video records found: latest playback used cached AAC and bypassed slate "
                f"(sid={last['sid']} start={float(last.get('start_sec', 0.0)):.3f}s cache={last.get('cache', 'unknown')})"
            )
            return 0
        if source_audio_sessions:
            last = source_audio_sessions[-1]
            print(
                "no slate video records found: latest playback used direct source audio and bypassed slate "
                f"(sid={last['sid']} start={float(last.get('start_sec', 0.0)):.3f}s)"
            )
            return 0
        print("no slate video records found and no passthrough sessions were recognized", file=sys.stderr)
        return 2
    worst = max(records, key=lambda item: float(item["estimated_video_lead_sec"]))
    if args.json:
        print(json.dumps({"records": records, "sessions": sessions, "worst": worst}, indent=2, ensure_ascii=False))
    else:
        print(
            "worst slate video lead: "
            f"{float(worst['estimated_video_lead_sec']):.3f}s "
            f"(sid={worst['sid']} frames={worst['frames']} fps={float(worst['fps']):.3f} "
            f"elapsed={float(worst['elapsed_sec']):.3f}s burst={worst['burst_frames']})"
        )
    return 0 if float(worst["estimated_video_lead_sec"]) <= args.max_lead else 1


if __name__ == "__main__":
    raise SystemExit(main())
