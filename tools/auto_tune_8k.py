"""Phase automation harness for 8K realtime passthrough tuning.

The harness starts the production server, drives the DLNA Browse/playback path
through ``tools.dlna_client_probe``, parses server diagnostics, and writes a
machine-readable baseline bundle under ``baseline/``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from tools import dlna_client_probe  # noqa: E402
from ui.services.process_helpers import base_environment  # noqa: E402


PYNV_RE = re.compile(
    r"\[PYNV\]\[(?P<sid>\d+)\] frame (?P<frame>\d+)/(?P<target>\d+) "
    r"fps=(?P<fps>[\d.]+) interval_fps=(?P<interval_fps>[\d.]+) "
    r"src_idx=(?P<src_idx>\d+) bytes=(?P<bytes>\d+) out_bps=(?P<out_bps_mbps>[\d.]+)M "
    r"stage_avg_ms decode=(?P<avg_decode>[\d.]+) composite=(?P<avg_composite>[\d.]+) "
    r"sync=(?P<avg_sync>[\d.]+) encode=(?P<avg_encode>[\d.]+) mux=(?P<avg_mux>[\d.]+) "
    r"(?:mat_avg_ms pre=(?P<avg_mat_pre>[\d.]+) ort=(?P<avg_mat_ort>[\d.]+) kernel=(?P<avg_mat_kernel>[\d.]+) )?"
    r"stage_max_ms decode=(?P<max_decode>[\d.]+) composite=(?P<max_composite>[\d.]+) "
    r"sync=(?P<max_sync>[\d.]+) encode=(?P<max_encode>[\d.]+) mux=(?P<max_mux>[\d.]+)"
    r"(?: mat_max_ms pre=(?P<max_mat_pre>[\d.]+) ort=(?P<max_mat_ort>[\d.]+) kernel=(?P<max_mat_kernel>[\d.]+))?"
)
SLOW_MUX_RE = re.compile(
    r"\[PYNV\]\[(?P<sid>\d+)\] mux stdin write slow: frame=(?P<frame>\d+) "
    r"len=(?P<len>\d+) elapsed=(?P<elapsed_sec>[\d.]+)s"
)
SID_RE = re.compile(r"\[PYNV\]\[(?P<sid>\d+)\]")


@dataclass
class ServerRun:
    pid: int
    returncode: int | None
    stdout_tail: str
    stderr_tail: str


def _resolve_video(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    root_relative = (config.ROOT / p).resolve()
    if root_relative.exists():
        return root_relative
    if not p.is_absolute():
        p = config.VIDEO_DIR / p
    return p.resolve()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def _read_tail(path: Path, max_chars: int = 240_000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _wait_for_server(base_url: str, timeout: float) -> None:
    deadline = time.perf_counter() + timeout
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            with urlopen(f"{base_url}/description.xml", timeout=2.0) as resp:
                if int(resp.status) != 200:
                    last_error = f"/description.xml status {resp.status}"
                    time.sleep(0.5)
                    continue
            dlna_client_probe.browse(base_url, "0", timeout=5.0)
            return
        except Exception as e:  # readiness can fail while uvicorn is binding
            if isinstance(e, URLError):
                last_error = str(e.reason)
            else:
                last_error = f"{type(e).__name__}: {e}"
            time.sleep(0.5)
    raise TimeoutError(f"server did not become DLNA-ready within {timeout:.1f}s: {last_error}")


def _assert_process_alive(proc: subprocess.Popen[str]) -> None:
    rc = proc.poll()
    if rc is not None:
        raise RuntimeError(f"server process exited before readiness check completed: rc={rc}")


def _start_server(args: argparse.Namespace, video: Path, base_url: str) -> subprocess.Popen[str]:
    env = base_environment()
    for item in args.server_env:
        if "=" not in item:
            raise ValueError(f"--server-env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--server-env key is empty: {item!r}")
        env[key] = value
    env["PT_DEBUG_LOGS"] = "1"
    env["PT_HTTP_PORT"] = str(args.port)
    env.setdefault("PT_CUDNN_BIN", r"C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64")
    env = base_environment(env)
    env.setdefault("PT_STARTUP_GPU_WARMUP", "1")
    env.setdefault("PT_PASSTHROUGH_OUTPUT_MODE", "all" if args.prefer == "alpha" else "green")
    if args.video_dir:
        env["PT_VIDEO_DIR"] = str(Path(args.video_dir).resolve())
    elif video.exists():
        env["PT_VIDEO_DIR"] = str(video.parent)

    cmd = [sys.executable, str(config.ROOT / "main.py"), "--debug"]
    stdout_path = args.run_dir / "server_stdout.log"
    stderr_path = args.run_dir / "server_stderr.log"
    stdout = stdout_path.open("w", encoding="utf-8", buffering=1)
    stderr = stderr_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(config.ROOT),
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    args._server_stdout_file = stdout
    args._server_stderr_file = stderr
    args._server_stdout_path = stdout_path
    args._server_stderr_path = stderr_path
    _assert_process_alive(proc)
    _wait_for_server(base_url, args.startup_timeout)
    _assert_process_alive(proc)
    return proc


def _stop_server(proc: subprocess.Popen[str], args: argparse.Namespace) -> ServerRun:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=args.shutdown_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    for attr in ("_server_stdout_file", "_server_stderr_file"):
        handle = getattr(args, attr, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
    return ServerRun(
        pid=proc.pid,
        returncode=proc.returncode,
        stdout_tail=_read_tail(getattr(args, "_server_stdout_path", Path()), 40_000),
        stderr_tail=_read_tail(getattr(args, "_server_stderr_path", Path()), 40_000),
    )


def _run_client_probe(args: argparse.Namespace, video: Path, base_url: str, out_path: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(config.ROOT / "tools" / "dlna_client_probe.py"),
        args.name or video.name,
        "--base-url",
        base_url,
        "--profile",
        args.profile,
        "--prefer",
        args.prefer,
        "--duration",
        f"{args.duration:.3f}",
        "--timeout",
        f"{args.client_timeout:.3f}",
        "--chapter-index",
        str(args.chapter_index),
        "--out",
        str(out_path),
    ]
    if args.max_depth is not None:
        cmd.extend(["--max-depth", str(args.max_depth)])
    if args.with_lavf_side_probes:
        cmd.append("--with-lavf-side-probes")
    if args.duplicate_startup:
        cmd.extend(["--duplicate-startup", str(args.duplicate_startup)])
    for header in args.header:
        cmd.extend(["--header", header])
    return subprocess.run(
        cmd,
        cwd=str(config.ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=args.client_timeout + args.duration + 20,
    )


def _numbers(match: re.Match[str]) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for key, value in match.groupdict().items():
        if value is None:
            row[key] = 0.0
            continue
        if key in {"sid", "frame", "target", "src_idx", "bytes", "len"}:
            row[key] = int(value)
        else:
            row[key] = float(value)
    return row


def parse_server_log(text: str) -> dict[str, Any]:
    diagnostics = [_numbers(match) for match in PYNV_RE.finditer(text)]
    slow_mux = [_numbers(match) for match in SLOW_MUX_RE.finditer(text)]
    sids = sorted({int(match.group("sid")) for match in SID_RE.finditer(text)})
    pacing_lines = [
        line
        for line in text.splitlines()
        if "pacing" in line.lower() or "stall" in line.lower() or "timeout" in line.lower()
    ]
    selected_mode = ""
    for line in text.splitlines():
        if "mode=alpha" in line or "output_mode=alpha" in line:
            selected_mode = "alpha"
        elif "mode=green" in line or "output_mode=green" in line:
            selected_mode = "green"
    latest = diagnostics[-1] if diagnostics else {}
    avg_interval = (
        sum(float(row["interval_fps"]) for row in diagnostics) / len(diagnostics)
        if diagnostics
        else 0.0
    )
    return {
        "session_ids": sids,
        "selected_mode": selected_mode,
        "diagnostics": diagnostics,
        "latest": latest,
        "average_interval_fps": avg_interval,
        "slow_mux_warnings": slow_mux,
        "pacing_warnings": pacing_lines[-40:],
    }


def _classify(log_metrics: dict[str, Any], client: dict[str, Any]) -> str:
    latest = log_metrics.get("latest") or {}
    interval_fps = float(latest.get("interval_fps") or log_metrics.get("average_interval_fps") or 0.0)
    mux_slow = bool(log_metrics.get("slow_mux_warnings"))
    main = client.get("main") or {}
    duration = float(main.get("elapsed_sec") or 0.0)
    bytes_read = int(main.get("bytes_read") or 0)
    status = int(main.get("status") or 0)
    client_ok = status in {200, 206} and bytes_read > 0 and not main.get("error")
    if mux_slow:
        return "mux_back_pressure"
    if interval_fps >= 35.0 and (not client_ok or duration <= 0 or bytes_read <= 0):
        return "http_delivery_or_client_pull"
    if interval_fps > 0 and interval_fps < 30.0:
        return "producer_cap"
    if interval_fps >= 35.0:
        return "producer_healthy"
    return "inconclusive"


def _report_markdown(args: argparse.Namespace, data: dict[str, Any]) -> str:
    log_metrics = data.get("server_log_metrics") or {}
    client = data.get("client_probe") or {}
    main = client.get("main") or {}
    latest = log_metrics.get("latest") or {}
    slow_mux = log_metrics.get("slow_mux_warnings") or []
    pacing = log_metrics.get("pacing_warnings") or []
    return "\n".join(
        [
            f"# 8K Auto Tune {args.phase} Report",
            "",
            f"- Generated: {data['generated_at']}",
            f"- Video: `{data['video']}`",
            f"- Profile/prefer: `{args.profile}` / `{args.prefer}`",
            f"- Duration: {args.duration:.1f}s",
            f"- Server env overrides: `{args.server_env}`",
            f"- Attribution: `{data['attribution']}`",
            "",
            "## Client Pull",
            "",
            f"- HTTP status: `{main.get('status', 0)}`",
            f"- First byte: `{main.get('first_byte_sec')}` s",
            f"- Bytes read: `{main.get('bytes_read', 0)}`",
            f"- Average bitrate: `{float(main.get('average_bps') or 0.0) / 1_000_000:.2f}` Mbps",
            f"- Error: `{main.get('error', '')}`",
            "",
            "## Producer Diagnostics",
            "",
            f"- Session IDs: `{log_metrics.get('session_ids', [])}`",
            f"- Latest interval FPS: `{latest.get('interval_fps', 0)}`",
            f"- Average interval FPS: `{float(log_metrics.get('average_interval_fps') or 0.0):.2f}`",
            f"- Latest stage avg ms: decode `{latest.get('avg_decode', 0)}`, composite `{latest.get('avg_composite', 0)}`, sync `{latest.get('avg_sync', 0)}`, encode `{latest.get('avg_encode', 0)}`, mux `{latest.get('avg_mux', 0)}`",
            f"- Latest mat avg ms: pre `{latest.get('avg_mat_pre', 0)}`, ort `{latest.get('avg_mat_ort', 0)}`, kernel `{latest.get('avg_mat_kernel', 0)}`",
            f"- Slow mux warnings: `{len(slow_mux)}`",
            f"- Pacing/stall/timeout lines: `{len(pacing)}`",
            "",
            "## Files",
            "",
            f"- JSON: `{data['paths']['json']}`",
            f"- Client JSON: `{data['paths']['client_json']}`",
            f"- Server log excerpt: `{data['paths']['server_log_excerpt']}`",
            f"- Client stdout: `{data['paths']['client_stdout']}`",
            "",
        ]
    )


def run(args: argparse.Namespace) -> int:
    video = _resolve_video(args.video)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_dir = config.ROOT / "baseline" / f"auto_tune_8k_{args.phase}_{stamp}_artifacts"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    json_path = config.ROOT / "baseline" / f"auto_tune_8k_{args.phase}_{stamp}.json"
    md_path = config.ROOT / "baseline" / f"auto_tune_8k_{args.phase}_{stamp}.md"
    client_json_path = args.run_dir / "client_probe.json"
    server_log_excerpt_path = args.run_dir / "server_log_excerpt.log"
    client_stdout_path = args.run_dir / "client_stdout.log"
    client_stderr_path = args.run_dir / "client_stderr.log"

    proc: subprocess.Popen[str] | None = None
    server_run: ServerRun | None = None
    client_result: subprocess.CompletedProcess[str] | None = None
    error = ""
    try:
        proc = _start_server(args, video, base_url)
        client_result = _run_client_probe(args, video, base_url, client_json_path)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        if proc is not None:
            server_run = _stop_server(proc, args)

    if client_result is not None:
        client_stdout_path.write_text(client_result.stdout, encoding="utf-8")
        client_stderr_path.write_text(client_result.stderr, encoding="utf-8")

    server_log = _read_tail(config.ROOT / "debug_output" / "server.log")
    server_log_excerpt_path.write_text(server_log, encoding="utf-8")
    client_data: dict[str, Any] = {}
    if client_json_path.exists():
        client_data = json.loads(client_json_path.read_text(encoding="utf-8"))

    log_metrics = parse_server_log(server_log)
    data: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": args.phase,
        "video": str(video),
        "base_url": base_url,
        "profile": args.profile,
        "prefer": args.prefer,
        "duration_sec": args.duration,
        "server_env": args.server_env,
        "error": error,
        "server_run": asdict(server_run) if server_run is not None else None,
        "client_returncode": client_result.returncode if client_result is not None else None,
        "client_probe": client_data,
        "server_log_metrics": log_metrics,
        "attribution": _classify(log_metrics, client_data) if client_data else "inconclusive",
        "paths": {
            "json": str(json_path),
            "md": str(md_path),
            "client_json": str(client_json_path),
            "server_log_excerpt": str(server_log_excerpt_path),
            "client_stdout": str(client_stdout_path),
            "client_stderr": str(client_stderr_path),
        },
    }
    _write_json(json_path, data)
    _write_md(md_path, _report_markdown(args, data))
    print(f"[auto-tune] wrote {json_path}")
    print(f"[auto-tune] wrote {md_path}")
    if error:
        print(f"[auto-tune] ERROR: {error}", file=sys.stderr)
        return 1
    if client_result is not None and client_result.returncode != 0:
        print(f"[auto-tune] client probe returned {client_result.returncode}", file=sys.stderr)
        return client_result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated 8K passthrough tuning phases.")
    parser.add_argument("phase", choices=["phase1", "phase4"], help="phase gate to run")
    parser.add_argument("--video", required=True, help="8K video path or filename under PT_VIDEO_DIR")
    parser.add_argument("--name", default="", help="DLNA title/path fragment; defaults to video filename")
    parser.add_argument("--video-dir", default="", help="server PT_VIDEO_DIR override; default uses video parent")
    parser.add_argument("--profile", choices=["skybox", "moonvr", "nplayer", "quest", "lavf", "default"], default="quest")
    parser.add_argument("--prefer", choices=["alpha", "green"], default="alpha")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--port", type=int, default=int(config.HTTP_PORT))
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--shutdown-timeout", type=float, default=8.0)
    parser.add_argument("--client-timeout", type=float, default=60.0)
    parser.add_argument("--chapter-index", type=int, default=-1)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--duplicate-startup", type=int, default=0)
    parser.add_argument("--with-lavf-side-probes", action="store_true")
    parser.add_argument(
        "--server-env",
        action="append",
        default=[],
        help="extra environment override for the spawned server, as KEY=VALUE",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
