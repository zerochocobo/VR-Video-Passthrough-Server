"""Simulate DLNA Browse plus playback requests for live passthrough tuning.

This tool intentionally talks to the running HTTP/DLNA server instead of
calling ContentDirectory internals. It discovers the same DIDL item URLs a real
DLNA client sees, then pulls the selected live stream with configurable client
headers so server-side routing, pacing, cache, and concurrency behavior are
exercised.
"""
from __future__ import annotations

import argparse
import html
import json
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

SOAP_ACTION = '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"'
DIDL_NS = {"d": "urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/"}
DC_NS = "{http://purl.org/dc/elements/1.1/}"


@dataclass
class BrowseNode:
    id: str
    parent_id: str
    title: str
    is_container: bool
    child_count: int = 0
    url: str = ""
    protocol_info: str = ""


@dataclass
class PullResult:
    label: str
    url: str
    status: int
    elapsed_sec: float
    first_byte_sec: float | None
    bytes_read: int
    average_bps: float
    error: str = ""
    headers: dict[str, str] | None = None


def _base_url(args: argparse.Namespace) -> str:
    return args.base_url.rstrip("/") if args.base_url else f"http://127.0.0.1:{config.HTTP_PORT}"


def _soap_body(object_id: str, flag: str, start: int = 0, count: int = 0) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body>"
        '<u:Browse xmlns:u="urn:schemas-upnp-org:service:ContentDirectory:1">'
        f"<ObjectID>{html.escape(object_id)}</ObjectID>"
        f"<BrowseFlag>{flag}</BrowseFlag>"
        "<Filter>*</Filter>"
        f"<StartingIndex>{start}</StartingIndex>"
        f"<RequestedCount>{count}</RequestedCount>"
        "<SortCriteria></SortCriteria>"
        "</u:Browse>"
        "</s:Body></s:Envelope>"
    ).encode("utf-8")


def _http_post(url: str, body: bytes, timeout: float) -> bytes:
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": 'text/xml; charset="utf-8"',
            "SOAPAction": SOAP_ACTION,
            "User-Agent": "PTMediaServer-DLNA-Probe/1.0",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _text_of(elem: ET.Element, local_name: str) -> str:
    for child in elem.iter():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == local_name:
            return child.text or ""
    return ""


def browse(base_url: str, object_id: str, timeout: float) -> list[BrowseNode]:
    response = _http_post(
        f"{base_url}/control/cds",
        _soap_body(object_id, "BrowseDirectChildren"),
        timeout,
    )
    root = ET.fromstring(response)
    # ElementTree already resolves the SOAP-level escaped DIDL text. Running a
    # second html.unescape corrupts item URLs that contain query parameters.
    didl_text = _text_of(root, "Result")
    didl = ET.fromstring(didl_text)
    nodes: list[BrowseNode] = []
    for elem in list(didl):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag not in {"container", "item"}:
            continue
        title = elem.findtext(f"{DC_NS}title") or ""
        res = elem.find("{urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/}res")
        nodes.append(
            BrowseNode(
                id=elem.attrib.get("id", ""),
                parent_id=elem.attrib.get("parentID", ""),
                title=title,
                is_container=(tag == "container"),
                child_count=int(elem.attrib.get("childCount", "0") or 0),
                url=(res.text or "") if res is not None else "",
                protocol_info=res.attrib.get("protocolInfo", "") if res is not None else "",
            )
        )
    return nodes


def _matches_name(node: BrowseNode, wanted: str) -> bool:
    wanted_l = wanted.casefold()
    title_l = node.title.casefold()
    path_l = urlparse(node.url).path.casefold()
    return wanted_l in title_l or wanted_l in path_l


def _walk(base_url: str, object_id: str, timeout: float, max_depth: int) -> Iterable[BrowseNode]:
    if max_depth < 0:
        return
    children = browse(base_url, object_id, timeout)
    for child in children:
        yield child
        if child.is_container and max_depth > 0:
            yield from _walk(base_url, child.id, timeout, max_depth - 1)


def _live_score(node: BrowseNode, name: str, prefer: str) -> tuple[int, int]:
    title = node.title.casefold()
    url = node.url.casefold()
    if not _matches_name(node, name):
        return (-1000, 0)
    is_alpha = "alpha" in title or "mode=alpha" in url
    is_live = "live" in title or "/passthrough_live/" in url
    if not is_live:
        return (-100, 0)
    preference = 100 if ((prefer == "alpha" and is_alpha) or (prefer == "green" and not is_alpha)) else 50
    playable = 20 if node.url else 0
    return (preference + playable, len(title))


def select_live_node(args: argparse.Namespace) -> BrowseNode:
    base_url = _base_url(args)
    candidates = list(_walk(base_url, args.object_id, args.timeout, args.max_depth))
    direct = [node for node in candidates if not node.is_container and "/passthrough_live/" in node.url]
    direct.sort(key=lambda node: _live_score(node, args.name, args.prefer), reverse=True)
    if direct and _live_score(direct[0], args.name, args.prefer)[0] > 0:
        return direct[0]

    containers = [node for node in candidates if node.is_container and _live_score(node, args.name, args.prefer)[0] > 0]
    containers.sort(key=lambda node: _live_score(node, args.name, args.prefer), reverse=True)
    for container in containers:
        chapters = browse(base_url, container.id, args.timeout)
        playable = [node for node in chapters if not node.is_container and "/passthrough_live/" in node.url]
        playable.sort(key=lambda node: (args.start not in node.title, node.title))
        if playable:
            if args.chapter_index >= 0 and args.chapter_index < len(playable):
                return playable[args.chapter_index]
            return playable[0]

    raise SystemExit(f"no passthrough live item found for {args.name!r}; browsed {len(candidates)} DIDL nodes")


def _profile_headers(profile: str) -> dict[str, str]:
    if profile == "skybox":
        return {
            "User-Agent": "Mozilla/5.0 SkyboxVR libmpv",
            "Accept": "*/*",
            "Range": "bytes=0-",
            "transferMode.dlna.org": "Streaming",
            "getcontentFeatures.dlna.org": "1",
        }
    if profile == "moonvr":
        return {
            "User-Agent": "VLC/3.0.18 LibVLC/3.0.18",
            "Accept": "*/*",
            "Range": "bytes=0-",
            "transferMode.dlna.org": "Streaming",
            "getcontentFeatures.dlna.org": "1",
        }
    if profile == "nplayer":
        return {
            "User-Agent": "nPlayer/3.0",
            "Accept": "*/*",
            "Range": "bytes=0-",
            "transferMode.dlna.org": "Streaming",
            "getcontentFeatures.dlna.org": "1",
        }
    if profile == "quest":
        return {
            "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 12; Quest)",
            "Accept": "*/*",
            "transferMode.dlna.org": "Streaming",
            "getcontentFeatures.dlna.org": "1",
        }
    if profile == "lavf":
        return {
            "User-Agent": "Lavf/58.45.100",
            "Accept": "*/*",
            "Range": "bytes=564-",
        }
    return {
        "User-Agent": "PTMediaServer-DLNA-Probe/1.0",
        "Accept": "*/*",
        "transferMode.dlna.org": "Streaming",
    }


def _merge_headers(profile: str, extra: list[str]) -> dict[str, str]:
    headers = _profile_headers(profile)
    for item in extra:
        if ":" not in item:
            raise SystemExit(f"invalid --header value, expected Name: value: {item}")
        key, value = item.split(":", 1)
        headers[key.strip()] = value.strip()
    return headers


def pull_stream(label: str, url: str, headers: dict[str, str], duration: float, timeout: float, chunk_size: int) -> PullResult:
    started = time.perf_counter()
    first_byte: float | None = None
    total = 0
    status = 0
    response_headers: dict[str, str] = {}
    error = ""
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            response_headers = {key: value for key, value in resp.headers.items()}
            deadline: float | None = None
            while deadline is None or time.perf_counter() < deadline:
                try:
                    chunk = resp.read(chunk_size)
                except socket.timeout:
                    error = "socket timeout while reading"
                    break
                if not chunk:
                    break
                if first_byte is None:
                    now = time.perf_counter()
                    first_byte = now - started
                    deadline = now + duration
                total += len(chunk)
    except HTTPError as e:
        status = int(e.code)
        response_headers = {key: value for key, value in e.headers.items()}
        error = e.read(300).decode("utf-8", "ignore")
    except URLError as e:
        error = str(e.reason)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - started
    return PullResult(
        label=label,
        url=url,
        status=status,
        elapsed_sec=round(elapsed, 3),
        first_byte_sec=round(first_byte, 3) if first_byte is not None else None,
        bytes_read=total,
        average_bps=(total * 8.0 / elapsed) if elapsed > 0 else 0.0,
        error=error,
        headers=response_headers,
    )


def _side_probe(url: str, profile: str, delay: float, duration: float, timeout: float, chunk_size: int, results: list[PullResult]) -> None:
    time.sleep(max(0.0, delay))
    headers = _profile_headers(profile)
    results.append(pull_stream(f"side-{profile}", url, headers, duration, timeout, chunk_size))


def run(args: argparse.Namespace) -> int:
    node = select_live_node(args)
    url = node.url
    if args.time_seek:
        headers_extra = [*args.header, f"TimeSeekRange.dlna.org: npt={args.time_seek}-"]
    else:
        headers_extra = args.header
    headers = _merge_headers(args.profile, headers_extra)
    print(f"[dlna-probe] selected title={node.title!r} id={node.id!r}")
    print(f"[dlna-probe] url={url}")
    print(f"[dlna-probe] profile={args.profile} prefer={args.prefer}")

    side_results: list[PullResult] = []
    side_threads: list[threading.Thread] = []
    if args.with_lavf_side_probes:
        side_threads.append(
            threading.Thread(
                target=_side_probe,
                args=(url, "lavf", args.side_delay, args.side_duration, args.timeout, args.chunk_size, side_results),
                daemon=True,
            )
        )
    for _ in range(max(0, args.duplicate_startup)):
        side_threads.append(
            threading.Thread(
                target=_side_probe,
                args=(url, args.profile, args.side_delay, args.side_duration, args.timeout, args.chunk_size, side_results),
                daemon=True,
            )
        )
    for thread in side_threads:
        thread.start()

    main = pull_stream("main", url, headers, args.duration, args.timeout, args.chunk_size)
    for thread in side_threads:
        thread.join(timeout=args.timeout + args.side_duration + args.side_delay + 1.0)

    result = {
        "selected": asdict(node),
        "profile": args.profile,
        "prefer": args.prefer,
        "main": asdict(main),
        "side": [asdict(item) for item in side_results],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = (config.ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[dlna-probe] wrote {out}")
    return 0 if main.status in {200, 206} and main.bytes_read > 0 and not main.error else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulate DLNA Browse and live passthrough playback.")
    ap.add_argument("name", help="video filename/title fragment to locate through ContentDirectory")
    ap.add_argument("--base-url", default="", help="server base URL; default uses config.HTTP_PORT on 127.0.0.1")
    ap.add_argument("--object-id", default="0", help="ContentDirectory ObjectID to start browsing from")
    ap.add_argument("--max-depth", type=int, default=8, help="recursive Browse depth when locating the item")
    ap.add_argument("--prefer", choices=["alpha", "green"], default="alpha", help="prefer alpha-live or green passthrough-live")
    ap.add_argument("--profile", choices=["skybox", "moonvr", "nplayer", "quest", "lavf", "default"], default="skybox")
    ap.add_argument("-d", "--duration", type=float, default=20.0, help="seconds to read the main stream")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP connect/read timeout")
    ap.add_argument("--chunk-size", type=int, default=256 * 1024)
    ap.add_argument("--chapter-index", type=int, default=-1, help="chapter item index if live item is a container")
    ap.add_argument("--start", default="00:00", help="chapter title fragment to prefer when selecting chapters")
    ap.add_argument("--time-seek", default="", help="send TimeSeekRange.dlna.org, e.g. 120.0 or 00:02:00")
    ap.add_argument("--header", action="append", default=[], help="extra request header, format 'Name: value'")
    ap.add_argument("--duplicate-startup", type=int, default=0, help="number of duplicate same-profile side requests")
    ap.add_argument("--with-lavf-side-probes", action="store_true", help="send one Lavf side Range request while main playback runs")
    ap.add_argument("--side-delay", type=float, default=1.0)
    ap.add_argument("--side-duration", type=float, default=3.0)
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
