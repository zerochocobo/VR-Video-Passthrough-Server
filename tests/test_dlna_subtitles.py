"""Exercise subtitle discovery through actual SOAP and HTTP responses."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import quote
from xml.etree import ElementTree as ET

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import config
import dlna.content_directory as cds
import http_app.routes_dlna as routes_dlna
import http_app.routes_media as routes_media
from media_library import MediaLibrary, build_media_roots
from utils.media_index import IndexedVideo, MediaIndex
from utils.subtitles import find_external_subtitles


NS = {
    "d": cds.DIDL_NS,
    "sec": "http://www.sec.co.kr/",
    "pv": "http://www.pv.com/pvns/",
}
SRT = "1\r\n00:00:00,000 --> 00:00:02,000\r\n中文字幕 & Subtitle\r\n".encode("utf-8")


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "Videos"
    root.mkdir()
    library = MediaLibrary(build_media_roots([root]))
    index = MediaIndex(tmp_path / "index.db")
    app = FastAPI()
    app.include_router(routes_dlna.router)
    app.include_router(routes_media.router)
    with ExitStack() as stack:
        stack.enter_context(patch.object(config, "MEDIA_LIBRARY", library))
        stack.enter_context(patch.object(config, "SUBTITLE_ENABLE", False))
        for module in (cds, routes_media):
            stack.enter_context(patch.object(module, "MEDIA_LIBRARY", library))
            stack.enter_context(patch.object(module, "LAN_IP", "testserver"))
            stack.enter_context(patch.object(module, "HTTP_PORT", 80))
        stack.enter_context(patch.object(cds, "VIDEO_DIR", root))
        stack.enter_context(patch.object(cds, "get_media_index", return_value=index))
        stack.enter_context(patch.object(cds, "PASSTHROUGH_OUTPUT_MODE", "none"))
        stack.enter_context(patch.object(cds, "DLNA_ALL_VIDEOS_ENABLED", False))
        stack.enter_context(patch.object(cds, "_face_beauty_dlna_enabled", return_value=False))
        stack.enter_context(patch.object(cds, "_rm_dlna_enabled", return_value=False))
        stack.enter_context(patch.object(cds, "_si_dlna_enabled", return_value=False))
        # Synthetic media avoids ffprobe/GPU work; scanning, cache, DIDL and
        # HTTP delivery still execute their real implementations.
        stack.enter_context(patch.object(index, "_video_for", return_value=IndexedVideo(duration=2.0)))
        stack.enter_context(patch.object(cds, "probe_cached", return_value=SimpleNamespace(duration=2.0, width=1920, height=1080, fps=24)))
        stack.enter_context(patch.object(cds, "probe_video_metadata", return_value=SimpleNamespace(codec=SimpleNamespace(codec_name="h264"), timing=SimpleNamespace(source_fps=24), color=None)))
        stack.enter_context(patch.object(cds, "select_backend", return_value=SimpleNamespace(verdict="pynv_hevc")))
        cds.clear_dir_items_cache()
        try:
            with TestClient(app) as client:
                yield root, client, library
        finally:
            cds.clear_dir_items_cache()
            index.close()


def browse(client, object_id="0", flag="BrowseDirectChildren"):
    body = (
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"><s:Body>'
        '<u:Browse xmlns:u="urn:schemas-upnp-org:service:ContentDirectory:1">'
        f"<ObjectID>{object_id.replace('&', '&amp;')}</ObjectID><BrowseFlag>{flag}</BrowseFlag>"
        "<Filter>*</Filter><StartingIndex>0</StartingIndex><RequestedCount>0</RequestedCount>"
        "<SortCriteria></SortCriteria></u:Browse></s:Body></s:Envelope>"
    )
    response = client.post("/control/cds", content=body.encode("utf-8"), headers={"SOAPAction": '"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"'})
    assert response.status_code == 200
    envelope = ET.fromstring(response.content)
    result = next(element for element in envelope.iter() if element.tag.rsplit("}", 1)[-1] == "Result")
    return ET.fromstring(result.text)


def test_browse_metadata_caption_and_all_resource_urls_are_fetchable(server):
    root, client, _ = server
    video = root / "影片 [2026] & demo.mp4"
    video.write_bytes(b"unaltered video bytes")
    # An ASS file sorts first by language, but single-link clients need SRT.
    video.with_suffix(".ass").write_bytes(b"ass subtitle")
    video.with_name(f"{video.stem}.ZH-CN.SRT").write_bytes(SRT)
    video.with_name(f"{video.stem}.en.srt").write_bytes(b"English subtitle")
    item = browse(client).find("d:item", NS)
    assert item is not None
    resources = item.findall("d:res", NS)
    assert resources[0].attrib["protocolInfo"].startswith("http-get:*:video/")
    caption = item.find("sec:CaptionInfoEx", NS)
    assert caption.attrib == {f"{{{NS['sec']}}}type": "srt"}
    assert item.find("sec:CaptionInfo", NS).text == caption.text
    assert len(item.findall("sec:CaptionInfoEx", NS)) == 1
    assert resources[0].attrib[f"{{{NS['pv']}}}subtitleFileUri"] == caption.text
    assert resources[0].attrib[f"{{{NS['pv']}}}subtitleFileType"] == "srt"
    assert client.get(caption.text).content == SRT
    assert client.get(caption.text).headers["content-type"] == "text/srt"
    for resource in resources[1:]:
        response = client.get(resource.text)
        assert response.status_code == 200
        assert response.headers["content-type"] == resource.attrib["protocolInfo"].split(":")[2]
    for method in (client.get, client.head):
        response = method(resources[0].text, headers={"getCaptionInfo.sec": "1"})
        assert response.headers["captioninfo.sec"] == caption.text
        assert "getcaptioninfo.sec" not in response.headers
    assert client.get(resources[0].text).content == video.read_bytes()
    ranged_video = client.get(resources[0].text, headers={"Range": "bytes=2-6"})
    assert ranged_video.status_code == 206
    assert ranged_video.content == video.read_bytes()[2:7]
    assert ranged_video.headers["captioninfo.sec"] == caption.text
    metadata = browse(client, item.attrib["id"], "BrowseMetadata").find("d:item", NS)
    assert metadata is not None
    assert metadata.attrib["id"] == item.attrib["id"]
    assert metadata.find("sec:CaptionInfoEx", NS).text == caption.text
    legacy = browse(client, f"v_{video.name}", "BrowseMetadata").find("d:item", NS)
    assert legacy.attrib["id"] == f"v_{video.name}"
    assert legacy.find("sec:CaptionInfoEx", NS).text == caption.text


def test_cached_browse_tracks_subtitle_creation_replacement_and_removal(server):
    root, client, _ = server
    video = root / "movie.mp4"
    video.write_bytes(b"video")
    subtitle = video.with_suffix(".srt")
    first = browse(client).find("d:item", NS)
    assert first.find("sec:CaptionInfoEx", NS) is None
    subtitle.write_bytes(SRT)
    created = browse(client).find("d:item", NS)
    caption = created.find("sec:CaptionInfoEx", NS).text
    assert client.get(caption).content == SRT
    subtitle.write_bytes(b"replacement subtitle")
    assert client.get(caption).content == b"replacement subtitle"
    # Rename switches the cached caption URI even though video is unchanged.
    subtitle.rename(root / "movie.en.srt")
    renamed = browse(client).find("d:item", NS)
    assert renamed.find("sec:CaptionInfoEx", NS).text != caption
    (root / "movie.en.srt").unlink()
    removed = browse(client).find("d:item", NS)
    assert removed.find("sec:CaptionInfoEx", NS) is None
    assert len(removed.findall("d:res", NS)) == 1
    assert "captioninfo.sec" not in client.head("/media/movie.mp4").headers


@pytest.mark.parametrize("suffix,mime,payload", [
    ("ass", "application/x-ass", (
        "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\i1}中文字幕{\\i0}\\NSubtitle\n"
    ).encode("utf-8")),
    ("ssa", "application/x-ssa", (
        "[Script Info]\nScriptType: v4.00\n\n[Events]\n"
        "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: Marked=0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\b1}中文字幕{\\b0}\n"
    ).encode("utf-8")),
    ("vtt", "text/vtt", (
        "WEBVTT\n\nSTYLE\n::cue { color: lime; }\n\n"
        "00:00:00.000 --> 00:00:02.000 align:start position:10%\n"
        "<i>中文字幕</i> & Subtitle\n"
    ).encode("utf-8")),
])
def test_non_srt_caption_fallback_keeps_original_format(server, suffix, mime, payload):
    root, client, _ = server
    (root / "movie.mp4").write_bytes(b"video")
    (root / f"movie.zh.{suffix.upper()}").write_bytes(payload)
    item = browse(client).find("d:item", NS)
    caption = item.find("sec:CaptionInfoEx", NS)
    assert caption.attrib[f"{{{NS['sec']}}}type"] == suffix
    for method in (client.get, client.head):
        full = method(caption.text)
        assert full.status_code == 200
        assert full.headers["content-type"] == mime
        assert int(full.headers["content-length"]) == len(payload)
        assert full.content == (payload if method == client.get else b"")
        ranged = method(caption.text, headers={"Range": "bytes=3-15"})
        assert ranged.status_code == 206
        assert ranged.headers["content-type"] == mime
        assert ranged.headers["content-range"] == f"bytes 3-15/{len(payload)}"
        assert ranged.content == (payload[3:16] if method == client.get else b"")
    assert client.head("/media/movie.mp4").headers["captioninfo.sec"] == caption.text
    resources = item.findall("d:res", NS)
    assert len(resources) == 2
    assert resources[1].attrib["protocolInfo"] == f"http-get:*:{mime}:*"
    assert resources[1].attrib["{http://www.w3.org/XML/1998/namespace}lang"] == "zh"
    assert resources[0].attrib[f"{{{NS['pv']}}}subtitleFileType"] == suffix
    metadata = browse(client, item.attrib["id"], "BrowseMetadata").find("d:item", NS)
    assert metadata.find("sec:CaptionInfoEx", NS).text == caption.text
    assert metadata.find("sec:CaptionInfo", NS).attrib[f"{{{NS['sec']}}}type"] == suffix


@pytest.mark.parametrize("mime", [None, "text/srt"])
def test_subtitle_get_head_ranges_and_invalid_ranges(server, mime):
    root, client, _ = server
    (root / "movie.srt").write_bytes(SRT)
    url = "/subs/movie.srt" + (f"?mime={mime}" if mime else "")
    expected_mime = mime or "application/x-subrip"
    for method in (client.get, client.head):
        full = method(url)
        assert full.status_code == 200
        assert full.headers["content-type"] == expected_mime
        assert int(full.headers["content-length"]) == len(SRT)
        for byte_range, start, end in [("bytes=3-11", 3, 11), ("bytes=5-", 5, len(SRT)-1), ("bytes=-7", len(SRT)-7, len(SRT)-1)]:
            response = method(url, headers={"Range": byte_range})
            assert response.status_code == 206
            assert response.headers["content-type"] == expected_mime
            assert response.headers["content-range"] == f"bytes {start}-{end}/{len(SRT)}"
            assert int(response.headers["content-length"]) == end-start+1
            assert response.content == (SRT[start:end+1] if method == client.get else b"")
        for invalid in ("bytes=99999-", "bytes=9-2", "bytes=-0", "bytes=", "bytes=0-2,5-7"):
            response = method(url, headers={"Range": invalid})
            assert response.status_code == 416
            assert response.headers["content-range"] == f"bytes */{len(SRT)}"


def test_subtitle_route_rejects_other_files_mime_and_outside_paths(server):
    root, client, _ = server
    (root / "movie.srt").write_bytes(SRT)
    (root / "movie.ass").write_bytes(b"ASS")
    (root / "private.txt").write_bytes(b"private")
    root.parent.joinpath("outside.srt").write_bytes(SRT)
    for method in (client.get, client.head):
        assert method("/subs/movie.srt?mime=text/html").status_code == 400
        assert method("/subs/movie.ass?mime=text/srt").status_code == 400
        assert method("/subs/private.txt").status_code == 404
        assert method("/subs/missing.srt").status_code == 404
        assert method("/subs/%2E%2E%2Foutside.srt").status_code == 403
    (root / "directory.srt").mkdir()
    assert client.get("/subs/directory.srt").status_code == 404


def test_literal_discovery_rejects_nearby_titles_and_directories(server):
    root, _, _ = server
    video = root / "Movie [1].mp4"
    video.write_bytes(b"video")
    for name in ("movie [1].ZH.SRT", "Movie 1.en.srt", "Movie [1] sequel.srt"):
        (root / name).write_bytes(SRT)
    (root / "Movie [1].fr.srt").mkdir()
    tracks = find_external_subtitles(video)
    assert [track.path.name for track in tracks] == ["movie [1].ZH.SRT"]


def test_multi_root_subtitle_keys_and_identical_movie_names(server):
    root, client, _ = server
    other_root = root.parent / "Other Videos"
    other_root.mkdir()
    library = MediaLibrary(build_media_roots([root, other_root]))
    for folder, payload in ((root, SRT), (other_root, b"second library subtitle")):
        (folder / "movie.mp4").write_bytes(b"video")
        (folder / "movie.srt").write_bytes(payload)
    with patch.object(config, "MEDIA_LIBRARY", library), patch.object(cds, "MEDIA_LIBRARY", library), patch.object(routes_media, "MEDIA_LIBRARY", library):
        folders = browse(client).findall("d:container", NS)
        assert len(folders) == 2
        captions = [browse(client, folder.attrib["id"]).find("d:item/sec:CaptionInfoEx", NS).text for folder in folders]
        assert captions[0] != captions[1]
        assert [client.get(url).content for url in captions] == [SRT, b"second library subtitle"]
        for folder, caption in zip((root, other_root), captions):
            video_url = "/media/" + quote(library.path_to_key(folder / "movie.mp4"))
            assert client.head(video_url).headers["captioninfo.sec"] == caption
