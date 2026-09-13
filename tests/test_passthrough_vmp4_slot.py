from __future__ import annotations

import asyncio
import time
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import http_app.routes_media as routes_media
from pipeline.passthrough_vmp4_slot import (
    Vmp4SlotLayoutError,
    Vmp4SlotOutputTemplate,
    _write_manifest_atomic,
    build_passthrough_vmp4_slot_layout,
    hevc_annexb_to_length_prefixed_sample,
    ensure_vmp4_slot_manifest,
    iter_annexb_nal_units,
    iter_vmp4_slot_range,
    vmp4_slot_index_for_offset,
    write_vmp4_slot_hevc_annexb_payload,
)
from pipeline.si_virtual_mp4 import read_media_sample_table


def _box(box_type: str, payload: bytes) -> bytes:
    body = box_type.encode("latin1") + payload
    return (len(body) + 4).to_bytes(4, "big") + body


def _full_box(box_type: str, payload: bytes, *, version: int = 0, flags: int = 0) -> bytes:
    return _box(box_type, bytes([version]) + int(flags).to_bytes(3, "big") + payload)


def _mvhd(timescale: int, duration: int) -> bytes:
    payload = bytearray(100)
    payload[8:12] = int(timescale).to_bytes(4, "big")
    payload[12:16] = int(duration).to_bytes(4, "big")
    payload[-4:] = (2).to_bytes(4, "big")
    return _full_box("mvhd", bytes(payload))


def _tkhd(track_id: int, duration: int) -> bytes:
    payload = bytearray(72)
    payload[8:12] = int(track_id).to_bytes(4, "big")
    payload[16:20] = int(duration).to_bytes(4, "big")
    return _full_box("tkhd", bytes(payload), flags=0x000007)


def _mdhd(timescale: int, duration: int) -> bytes:
    payload = (
        b"\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00"
        + int(timescale).to_bytes(4, "big")
        + int(duration).to_bytes(4, "big")
        + b"\x55\xc4\x00\x00"
    )
    return _full_box("mdhd", payload)


def _hdlr(handler_type: bytes) -> bytes:
    payload = bytearray(24)
    payload[8:12] = handler_type
    return _box("hdlr", bytes(payload))


def _stsd_video(codec: str = "hvc1") -> bytes:
    return _full_box("stsd", (1).to_bytes(4, "big") + _box(codec, b"\x00" * 8))


def _stts(entries: list[tuple[int, int]]) -> bytes:
    payload = bytearray()
    payload += len(entries).to_bytes(4, "big")
    for count, delta in entries:
        payload += int(count).to_bytes(4, "big")
        payload += int(delta).to_bytes(4, "big")
    return _full_box("stts", bytes(payload))


def _stsc(entries: list[tuple[int, int, int]]) -> bytes:
    payload = bytearray()
    payload += len(entries).to_bytes(4, "big")
    for first_chunk, samples_per_chunk, sample_description_index in entries:
        payload += int(first_chunk).to_bytes(4, "big")
        payload += int(samples_per_chunk).to_bytes(4, "big")
        payload += int(sample_description_index).to_bytes(4, "big")
    return _full_box("stsc", bytes(payload))


def _stsz(sizes: list[int]) -> bytes:
    payload = bytearray()
    payload += (0).to_bytes(4, "big")
    payload += len(sizes).to_bytes(4, "big")
    for size in sizes:
        payload += int(size).to_bytes(4, "big")
    return _full_box("stsz", bytes(payload))


def _stco(offsets: list[int]) -> bytes:
    payload = bytearray()
    payload += len(offsets).to_bytes(4, "big")
    for offset in offsets:
        payload += int(offset).to_bytes(4, "big")
    return _full_box("stco", bytes(payload))


def _stss(sample_numbers: list[int]) -> bytes:
    payload = bytearray()
    payload += len(sample_numbers).to_bytes(4, "big")
    for number in sample_numbers:
        payload += int(number).to_bytes(4, "big")
    return _full_box("stss", bytes(payload))


def _write_source_mp4(path: Path, samples: list[bytes], *, target_size: int = 4096, codec: str = "hvc1") -> None:
    ftyp = _box("ftyp", b"isom\x00\x00\x02\x00isomiso2")
    sizes = [len(sample) for sample in samples]

    def make_moov(offsets: list[int]) -> bytes:
        stbl = _box(
            "stbl",
            _stsd_video(codec)
            + _stts([(len(samples), 2000)])
            + _stsc([(1, 1, 1)])
            + _stsz(sizes)
            + _stco(offsets)
            + _stss(list(range(1, len(samples) + 1))),
        )
        mdia = _box("mdia", _mdhd(1000, len(samples) * 2000) + _hdlr(b"vide") + _box("minf", stbl))
        trak = _box("trak", _tkhd(1, len(samples) * 2000) + mdia)
        return _box("moov", _mvhd(1000, len(samples) * 2000) + trak)

    moov = make_moov([0] * len(samples))
    for _ in range(3):
        mdat_payload_start = len(ftyp) + len(moov) + 8
        offsets: list[int] = []
        cursor = mdat_payload_start
        for sample in samples:
            offsets.append(cursor)
            cursor += len(sample)
        new_moov = make_moov(offsets)
        if len(new_moov) == len(moov):
            moov = new_moov
            break
        moov = new_moov

    payload = b"".join(samples)
    data = ftyp + moov + _box("mdat", payload)
    if len(data) < target_size:
        data += b"\x00" * (target_size - len(data))
    path.write_bytes(data)


class PassthroughVmp4SlotLayoutTests(unittest.TestCase):
    def test_hevc_annexb_to_length_prefixed_sample(self) -> None:
        annexb = (
            b"\x00\x00\x00\x01\x40\x01vps"
            b"\x00\x00\x01\x42\x01sps\x00\x00"
            b"\x00\x00\x00\x01\x44\x01pps"
        )
        nals = list(iter_annexb_nal_units(annexb))
        sample = hevc_annexb_to_length_prefixed_sample(annexb, nal_length_size=4)

        self.assertEqual(nals, [b"\x40\x01vps", b"\x42\x01sps", b"\x44\x01pps"])
        self.assertEqual(
            sample,
            b"\x00\x00\x00\x05\x40\x01vps"
            b"\x00\x00\x00\x05\x42\x01sps"
            b"\x00\x00\x00\x05\x44\x01pps",
        )

    def test_slot_layout_maps_ranges_to_stable_hevc_placeholder_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "movie.mp4"
            _write_source_mp4(source, [b"AAAA", b"BBBBBB", b"CCC"])
            placeholder = b"HEVC!!"
            layout = build_passthrough_vmp4_slot_layout(
                source,
                duration_sec=6.0,
                fps=1.0,
                total_size=source.stat().st_size,
                gop_frames=2,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=placeholder,
            )

            slot_start = layout.mdat_payload_start + layout.slot_stride
            first = b"".join(iter_vmp4_slot_range(layout, slot_start, slot_start + 5))
            second = b"".join(iter_vmp4_slot_range(layout, slot_start, slot_start + 5))
            filler = b"".join(iter_vmp4_slot_range(layout, slot_start + 6, slot_start + 11))
            init = b"".join(iter_vmp4_slot_range(layout, 0, len(layout.init) - 1))
            virtual_path = Path(tmp) / "virtual.mp4"
            virtual_path.write_bytes(b"".join(iter_vmp4_slot_range(layout, 0, layout.total_size - 1)))
            virtual_table = read_media_sample_table(virtual_path, "video")

        self.assertEqual(layout.slot_count, 3)
        self.assertEqual(vmp4_slot_index_for_offset(layout, slot_start), 1)
        self.assertEqual(first, placeholder)
        self.assertEqual(second, first)
        self.assertEqual(filler[4:6], b"\x4c\x01")
        self.assertEqual(init, layout.init)
        self.assertEqual([sample.size for sample in virtual_table.samples], [layout.slot_size] * layout.slot_count)
        self.assertEqual(virtual_table.codec_name, "hevc")

    def test_slot_layout_uses_hevc_output_stsd_for_av1_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "movie.mp4"
            _write_source_mp4(source, [b"AAAA", b"BBBBBB", b"CCC"], codec="av01")
            layout = build_passthrough_vmp4_slot_layout(
                source,
                duration_sec=6.0,
                fps=1.0,
                total_size=source.stat().st_size,
                gop_frames=2,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=b"HEVC",
            )
            virtual_path = Path(tmp) / "virtual.mp4"
            virtual_path.write_bytes(b"".join(iter_vmp4_slot_range(layout, 0, layout.total_size - 1)))
            virtual_table = read_media_sample_table(virtual_path, "video")

        self.assertEqual(layout.source_codec_name, "av01")
        self.assertEqual(layout.codec_name, "hevc")
        self.assertEqual(virtual_table.codec_name, "hevc")

    def test_slot_manifest_marks_ready_payload_and_reader_prefers_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "movie.mp4"
            _write_source_mp4(source, [b"AAAA", b"BBBBBB", b"CCC"])
            layout = build_passthrough_vmp4_slot_layout(
                source,
                duration_sec=6.0,
                fps=1.0,
                total_size=source.stat().st_size,
                gop_frames=2,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=b"HEVC!!",
            )
            cache_root = root / "cache"
            first_status = ensure_vmp4_slot_manifest(layout, cache_root, output_mode="green", fps=1.0, gop_frames=2)
            payload = first_status.cache_dir / "slot_000001.bin"
            payload.write_bytes(b"PAYLOAD")
            status = ensure_vmp4_slot_manifest(layout, cache_root, output_mode="green", fps=1.0, gop_frames=2)

            slot_start = layout.mdat_payload_start + layout.slot_stride
            body = b"".join(
                iter_vmp4_slot_range(
                    layout,
                    slot_start,
                    slot_start + 6,
                    payload_paths=status.ready_payloads,
                )
            )
            manifest_text = status.manifest_path.read_text(encoding="utf-8-sig")

        self.assertEqual(body, b"PAYLOAD")
        self.assertEqual(status.slot(1).state, "ready")
        self.assertEqual(status.slot(1).payload_size, 7)
        self.assertIn('"state": "ready"', manifest_text)

    def test_slot_manifest_marks_zero_byte_payload_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "movie.mp4"
            _write_source_mp4(source, [b"AAAA", b"BBBBBB", b"CCC"])
            layout = build_passthrough_vmp4_slot_layout(
                source,
                duration_sec=6.0,
                fps=1.0,
                total_size=source.stat().st_size,
                gop_frames=2,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=b"HEVC!!",
            )
            first_status = ensure_vmp4_slot_manifest(layout, root / "cache", output_mode="green", fps=1.0, gop_frames=2)
            payload = first_status.cache_dir / "slot_000001.bin"
            payload.write_bytes(b"")

            status = ensure_vmp4_slot_manifest(layout, root / "cache", output_mode="green", fps=1.0, gop_frames=2)

        self.assertEqual(status.slot(1).state, "failed")
        self.assertEqual(status.slot(1).reason, "payload-empty")
        self.assertNotIn(1, status.ready_payloads)

    def test_manifest_atomic_write_uses_unique_temp_files_under_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "manifest.json"
            errors: list[BaseException] = []

            def worker(index: int) -> None:
                try:
                    for seq in range(40):
                        _write_manifest_atomic(target, {"worker": index, "seq": seq, "slots": []})
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            leftover = list(Path(tmp).glob(".manifest.json.*.tmp"))
            target_exists = target.is_file()

        self.assertEqual(errors, [])
        self.assertEqual(leftover, [])
        self.assertTrue(target_exists)

    def test_write_hevc_annexb_payload_converts_for_ready_slot_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "movie.mp4"
            _write_source_mp4(source, [b"AAAA", b"BBBBBB", b"CCC"])
            layout = build_passthrough_vmp4_slot_layout(
                source,
                duration_sec=6.0,
                fps=1.0,
                total_size=source.stat().st_size,
                gop_frames=2,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=b"HEVC!!",
            )
            target = root / "slot_000001.bin"
            written = write_vmp4_slot_hevc_annexb_payload(
                layout,
                1,
                target,
                b"\x00\x00\x00\x01\x40\x01vps\x00\x00\x01\x26\x01idr",
            )

            self.assertEqual(written, target.stat().st_size)
            self.assertEqual(target.read_bytes(), b"\x00\x00\x00\x05\x40\x01vps\x00\x00\x00\x05\x26\x01idr")


class PassthroughVmp4SlotRouteTests(unittest.TestCase):
    def test_seek_route_slot_backend_serves_layout_without_cache_lookup(self) -> None:
        class _FakeRequest:
            headers = {"user-agent": "VLC/3.0", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        request = _FakeRequest()
        info = SimpleNamespace(duration=6.0, fps=1.0, width=3840, height=2160)
        estimate = SimpleNamespace(source="test")

        async def run(range_header: str) -> tuple[int, dict[str, str], bytes]:
            response = await routes_media.passthrough_seek_get(
                request,
                "movie.mp4",
                mode=None,
                range_header=range_header,
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            body = bytearray()
            async for chunk in response.body_iterator:
                body.extend(chunk)
            return response.status_code, dict(response.headers), bytes(body)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "movie.mp4"
            _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
            layout = build_passthrough_vmp4_slot_layout(
                path,
                duration_sec=6.0,
                fps=1.0,
                total_size=path.stat().st_size,
                gop_frames=2,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=b"HEVC!!",
            )
            slot_start = layout.mdat_payload_start + layout.slot_stride

            with (
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4", True),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_BACKEND", "slot"),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", False),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY", False),
                patch.object(routes_media, "PASSTHROUGH_GOP", 2),
                patch.object(routes_media, "RUNTIME_CACHE_DIR", Path(tmp) / "cache"),
                patch.object(
                    routes_media,
                    "_vmp4_slot_output_template",
                    return_value=Vmp4SlotOutputTemplate(
                        stsd=_stsd_video("hvc1"),
                        payload=b"HEVC!!",
                        codec_name="hevc",
                        nal_length_size=4,
                        width=3840,
                        height=2160,
                    ),
                ),
                patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None, None)),
                patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "vlc")),
                patch.object(routes_media, "annotate_request", return_value=None),
                patch.object(routes_media, "probe_cached", return_value=info),
                patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, estimate)),
                patch.object(routes_media, "_vmp4_mode_candidate", side_effect=AssertionError("cache lookup not used")),
            ):
                status_code, headers, body = asyncio.run(run(f"bytes={slot_start}-{slot_start + 5}"))

        self.assertEqual(status_code, 206)
        self.assertEqual(body, b"HEVC!!")
        self.assertEqual(headers["x-passthrough-vmp4-backend"], "slot")
        self.assertEqual(headers["x-passthrough-vmp4-phase"], "slot-layout")
        self.assertEqual(headers["x-passthrough-vmp4-output-codec"], "hevc")
        self.assertEqual(headers["x-passthrough-vmp4-slot-ready-only"], "0")
        self.assertEqual(headers["x-passthrough-vmp4-slot"], "1")
        self.assertEqual(headers["x-passthrough-vmp4-slot-state"], "placeholder")
        self.assertEqual(headers["x-passthrough-vmp4-slot-build"], "disabled")
        self.assertEqual(headers["x-passthrough-vmp4-slot-states"], "placeholder:3")
        self.assertEqual(headers["x-passthrough-vmp4-slot-sample-size"], str(layout.slot_size))
        self.assertEqual(headers["x-passthrough-vmp4-slot-source-bytes"], "6")
        self.assertEqual(headers["x-passthrough-vmp4-slot-placeholder-bytes"], "6")
        self.assertEqual(headers["x-passthrough-vmp4-slot-payload-bytes"], "6")
        self.assertEqual(headers["x-passthrough-vmp4-slot-filler-bytes"], str(layout.slot_size - 6))
        self.assertEqual(headers["content-range"], f"bytes {slot_start}-{slot_start + 5}/4096")

    def test_seek_route_slot_ready_only_waits_for_requested_payload(self) -> None:
        class _FakeRequest:
            headers = {"user-agent": "VLC/3.0", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        async def run(range_header: str) -> tuple[int, dict[str, str], bytes]:
            response = await routes_media.passthrough_seek_get(
                _FakeRequest(),
                "movie.mp4",
                mode=None,
                range_header=range_header,
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            body = bytearray()
            iterator = getattr(response, "body_iterator", None)
            if iterator is not None:
                async for chunk in iterator:
                    body.extend(chunk)
            else:
                body.extend(getattr(response, "body", b"") or b"")
            return response.status_code, dict(response.headers), bytes(body)

        original_builds = dict(routes_media._vmp4_slot_builds)
        try:
            routes_media._vmp4_slot_builds.clear()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "movie.mp4"
                _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
                layout = build_passthrough_vmp4_slot_layout(
                    path,
                    duration_sec=6.0,
                    fps=1.0,
                    total_size=path.stat().st_size,
                    gop_frames=2,
                    output_stsd=_stsd_video("hvc1"),
                    output_codec_name="hevc",
                    placeholder_payload=b"HEVC!!",
                )
                slot_start = layout.mdat_payload_start + layout.slot_stride
                annexb = b"\x00\x00\x00\x01\x26\x01"
                payload_bytes = hevc_annexb_to_length_prefixed_sample(annexb)

                with (
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_BACKEND", "slot"),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_WAIT", 2.0),
                    patch.object(routes_media, "PASSTHROUGH_GOP", 2),
                    patch.object(routes_media, "RUNTIME_CACHE_DIR", Path(tmp) / "cache"),
                    patch.object(
                        routes_media,
                        "_vmp4_slot_output_template",
                        return_value=Vmp4SlotOutputTemplate(
                            stsd=_stsd_video("hvc1"),
                            payload=b"HEVC!!",
                            codec_name="hevc",
                            nal_length_size=4,
                            width=3840,
                            height=2160,
                        ),
                    ),
                    patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None, None)),
                    patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "vlc")),
                    patch.object(routes_media, "annotate_request", return_value=None),
                    patch.object(routes_media, "probe_cached", return_value=SimpleNamespace(duration=6.0, fps=1.0, width=3840, height=2160)),
                    patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, SimpleNamespace(source="test"))),
                    patch.object(routes_media, "_vmp4_mode_candidate", side_effect=AssertionError("cache lookup not used")),
                    patch.object(routes_media, "acquire_matter", return_value=object()),
                    patch.object(routes_media, "release_matter", return_value=None),
                    patch.object(routes_media, "build_passthrough_hevc_annexb_gop", return_value=annexb),
                ):
                    status_code, headers, body = asyncio.run(run(f"bytes={slot_start}-{slot_start + 5}"))

                    payload = Path(tmp) / "cache" / "vmp4_slot"
                    deadline = time.time() + 2.0
                    while time.time() < deadline and not list(payload.glob("*/slot_000001.bin")):
                        time.sleep(0.01)

            self.assertEqual(status_code, 206)
            self.assertEqual(body, payload_bytes)
            self.assertEqual(headers["x-passthrough-vmp4-phase"], "slot-layout")
            self.assertEqual(headers["x-passthrough-vmp4-slot"], "1")
            self.assertEqual(headers["x-passthrough-vmp4-slot-state"], "ready")
            self.assertEqual(headers["x-passthrough-vmp4-slot-build"], "ready")
            self.assertEqual(headers["x-passthrough-vmp4-slot-ready-only"], "1")
            self.assertEqual(headers["x-passthrough-vmp4-slot-wait"], "1")
            self.assertEqual(headers["x-passthrough-vmp4-slot-wait-ready"], "1")
            self.assertNotIn("retry-after", headers)
        finally:
            routes_media._vmp4_slot_builds.clear()
            routes_media._vmp4_slot_builds.update(original_builds)

    def test_seek_route_slot_ready_only_no_range_streams_with_slot_waits(self) -> None:
        class _FakeRequest:
            headers = {"user-agent": "Skybox/1.0", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        async def run() -> tuple[int, dict[str, str], bytes]:
            response = await routes_media.passthrough_seek_get(
                _FakeRequest(),
                "movie.mp4",
                mode=None,
                range_header=None,
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            body = bytearray()
            iterator = getattr(response, "body_iterator", None)
            if iterator is not None:
                async for chunk in iterator:
                    body.extend(chunk)
            else:
                body.extend(getattr(response, "body", b"") or b"")
            return response.status_code, dict(response.headers), bytes(body)

        original_builds = dict(routes_media._vmp4_slot_builds)
        try:
            routes_media._vmp4_slot_builds.clear()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "movie.mp4"
                _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
                annexb = b"\x00\x00\x00\x01\x26\x01"
                payload_bytes = hevc_annexb_to_length_prefixed_sample(annexb)

                with (
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_BACKEND", "slot"),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_WAIT", 2.0),
                    patch.object(routes_media, "PASSTHROUGH_GOP", 2),
                    patch.object(routes_media, "RUNTIME_CACHE_DIR", Path(tmp) / "cache"),
                    patch.object(
                        routes_media,
                        "_vmp4_slot_output_template",
                        return_value=Vmp4SlotOutputTemplate(
                            stsd=_stsd_video("hvc1"),
                            payload=b"HEVC!!",
                            codec_name="hevc",
                            nal_length_size=4,
                            width=3840,
                            height=2160,
                        ),
                    ),
                    patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None, None)),
                    patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "skybox")),
                    patch.object(routes_media, "annotate_request", return_value=None),
                    patch.object(routes_media, "probe_cached", return_value=SimpleNamespace(duration=6.0, fps=1.0, width=3840, height=2160)),
                    patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, SimpleNamespace(source="test"))),
                    patch.object(routes_media, "_vmp4_mode_candidate", side_effect=AssertionError("cache lookup not used")),
                    patch.object(routes_media, "acquire_matter", return_value=object()),
                    patch.object(routes_media, "release_matter", return_value=None),
                    patch.object(routes_media, "build_passthrough_hevc_annexb_gop", return_value=annexb),
                ):
                    routes_media._vmp4_slot_layout_cache.clear()
                    status_code, headers, body = asyncio.run(run())

                    payload_root = Path(tmp) / "cache" / "vmp4_slot"
                    deadline = time.time() + 2.0
                    while time.time() < deadline and not list(payload_root.glob("*/slot_000000.bin")):
                        time.sleep(0.01)
                    built = list(payload_root.glob("*/slot_000000.bin"))

            self.assertEqual(status_code, 200)
            self.assertEqual(len(body), 4096)
            self.assertIn(payload_bytes, body)
            self.assertEqual(headers["x-passthrough-vmp4-phase"], "slot-layout")
            self.assertEqual(headers["x-passthrough-vmp4-slot"], "0")
            self.assertEqual(headers["x-passthrough-vmp4-slot-state"], "ready")
            self.assertEqual(headers["x-passthrough-vmp4-slot-ready-only"], "1")
            self.assertEqual(headers["x-passthrough-vmp4-slot-wait"], "0")
            self.assertEqual(headers["x-passthrough-vmp4-slot-wait-ready"], "1")
            self.assertNotIn("retry-after", headers)
            self.assertTrue(built)
        finally:
            routes_media._vmp4_slot_builds.clear()
            routes_media._vmp4_slot_builds.update(original_builds)

    def test_seek_route_slot_ready_only_zero_open_waits_instead_of_init_only(self) -> None:
        class _FakeRequest:
            headers = {"user-agent": "SKYBOX/2.0.2", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        async def run() -> tuple[int, dict[str, str], bytes]:
            response = await routes_media.passthrough_seek_get(
                _FakeRequest(),
                "movie.mp4",
                mode=None,
                range_header="bytes=0-",
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            body = bytearray()
            async for chunk in response.body_iterator:
                body.extend(chunk)
            return response.status_code, dict(response.headers), bytes(body)

        original_builds = dict(routes_media._vmp4_slot_builds)
        try:
            routes_media._vmp4_slot_builds.clear()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "movie.mp4"
                _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
                layout = build_passthrough_vmp4_slot_layout(
                    path,
                    duration_sec=6.0,
                    fps=1.0,
                    total_size=path.stat().st_size,
                    gop_frames=2,
                    output_stsd=_stsd_video("hvc1"),
                    output_codec_name="hevc",
                    placeholder_payload=b"HEVC!!",
                )
                slot1_start = layout.mdat_payload_start + layout.slot_stride
                annexb = b"\x00\x00\x00\x01\x26\x01"
                payload_bytes = hevc_annexb_to_length_prefixed_sample(annexb)

                with (
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_BACKEND", "slot"),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_WAIT", 2.0),
                    patch.object(routes_media, "PASSTHROUGH_GOP", 2),
                    patch.object(routes_media, "RUNTIME_CACHE_DIR", Path(tmp) / "cache"),
                    patch.object(
                        routes_media,
                        "_vmp4_slot_output_template",
                        return_value=Vmp4SlotOutputTemplate(
                            stsd=_stsd_video("hvc1"),
                            payload=b"HEVC!!",
                            codec_name="hevc",
                            nal_length_size=4,
                            width=3840,
                            height=2160,
                        ),
                    ),
                    patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None, None)),
                    patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "skybox")),
                    patch.object(routes_media, "annotate_request", return_value=None),
                    patch.object(routes_media, "probe_cached", return_value=SimpleNamespace(duration=6.0, fps=1.0, width=3840, height=2160)),
                    patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, SimpleNamespace(source="test"))),
                    patch.object(routes_media, "_vmp4_mode_candidate", side_effect=AssertionError("cache lookup not used")),
                    patch.object(routes_media, "acquire_matter", return_value=object()),
                    patch.object(routes_media, "release_matter", return_value=None) as release_mock,
                    patch.object(routes_media, "build_passthrough_hevc_annexb_gop", return_value=annexb),
                ):
                    routes_media._vmp4_slot_layout_cache.clear()
                    status_code, headers, body = asyncio.run(run())
                    deadline = time.time() + 2.0
                    while time.time() < deadline and release_mock.call_count < 2:
                        time.sleep(0.01)

            self.assertEqual(status_code, 206)
            self.assertEqual(headers["x-passthrough-vmp4-slot"], "0")
            self.assertEqual(headers["x-passthrough-vmp4-slot-state"], "ready")
            self.assertEqual(headers["x-passthrough-vmp4-slot-ready-bounded"], "1")
            self.assertEqual(headers["x-passthrough-vmp4-slot-wait"], "0")
            self.assertEqual(headers["x-passthrough-vmp4-slot-wait-ready"], "1")
            self.assertEqual(headers["content-range"], f"bytes 0-{slot1_start - 1}/4096")
            self.assertEqual(len(body), slot1_start)
            self.assertGreater(len(body), len(layout.init))
            self.assertIn(payload_bytes, body)
        finally:
            routes_media._vmp4_slot_builds.clear()
            routes_media._vmp4_slot_builds.update(original_builds)

    def test_seek_route_slot_ready_only_bounds_open_range_before_next_unready_slot(self) -> None:
        class _FakeRequest:
            headers = {"user-agent": "VLC/3.0", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        async def run(range_header: str) -> tuple[int, dict[str, str], bytes]:
            response = await routes_media.passthrough_seek_get(
                _FakeRequest(),
                "movie.mp4",
                mode=None,
                range_header=range_header,
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            body = bytearray()
            async for chunk in response.body_iterator:
                body.extend(chunk)
            return response.status_code, dict(response.headers), bytes(body)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "movie.mp4"
            cache_root = Path(tmp) / "cache" / "vmp4_slot"
            _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
            layout = build_passthrough_vmp4_slot_layout(
                path,
                duration_sec=6.0,
                fps=1.0,
                total_size=path.stat().st_size,
                gop_frames=2,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=b"HEVC!!",
            )
            status = ensure_vmp4_slot_manifest(layout, cache_root, output_mode="green", fps=1.0, gop_frames=2)
            (status.cache_dir / "slot_000000.bin").write_bytes(b"READY0")
            ensure_vmp4_slot_manifest(layout, cache_root, output_mode="green", fps=1.0, gop_frames=2)
            slot1_start = layout.mdat_payload_start + layout.slot_stride

            with (
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4", True),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_BACKEND", "slot"),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", False),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY", True),
                patch.object(routes_media, "PASSTHROUGH_GOP", 2),
                patch.object(routes_media, "RUNTIME_CACHE_DIR", Path(tmp) / "cache"),
                patch.object(
                    routes_media,
                    "_vmp4_slot_output_template",
                    return_value=Vmp4SlotOutputTemplate(
                        stsd=_stsd_video("hvc1"),
                        payload=b"HEVC!!",
                        codec_name="hevc",
                        nal_length_size=4,
                        width=3840,
                        height=2160,
                    ),
                ),
                patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None, None)),
                patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "vlc")),
                patch.object(routes_media, "annotate_request", return_value=None),
                patch.object(routes_media, "probe_cached", return_value=SimpleNamespace(duration=6.0, fps=1.0, width=3840, height=2160)),
                patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, SimpleNamespace(source="test"))),
                patch.object(routes_media, "_vmp4_mode_candidate", side_effect=AssertionError("cache lookup not used")),
            ):
                routes_media._vmp4_slot_layout_cache.clear()
                status_code, headers, body = asyncio.run(run("bytes=0-"))

        self.assertEqual(status_code, 206)
        self.assertEqual(headers["x-passthrough-vmp4-slot-ready-bounded"], "1")
        self.assertEqual(headers["x-passthrough-vmp4-next-slot"], "1")
        self.assertEqual(headers["x-passthrough-vmp4-next-slot-build"], "disabled")
        self.assertEqual(headers["content-range"], f"bytes 0-{slot1_start - 1}/4096")
        self.assertEqual(int(headers["content-length"]), slot1_start)
        self.assertEqual(len(body), slot1_start)
        self.assertNotIn(b"HEVC!!", body[slot1_start - layout.slot_stride:])

    def test_seek_route_slot_layout_is_cached_per_source_template(self) -> None:
        class _FakeRequest:
            headers = {"user-agent": "VLC/3.0", "accept": "*/*"}
            client = SimpleNamespace(host="client")

            async def is_disconnected(self) -> bool:
                return False

        request = _FakeRequest()
        info = SimpleNamespace(duration=6.0, fps=1.0, width=3840, height=2160)

        async def run(range_header: str) -> int:
            response = await routes_media.passthrough_seek_get(
                request,
                "movie.mp4",
                mode=None,
                range_header=range_header,
                time_seek_range=None,
                get_content_features=None,
                transfer_mode=None,
            )
            body = bytearray()
            async for chunk in response.body_iterator:
                body.extend(chunk)
            return response.status_code

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "movie.mp4"
            _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
            layout = build_passthrough_vmp4_slot_layout(
                path,
                duration_sec=6.0,
                fps=1.0,
                total_size=path.stat().st_size,
                gop_frames=2,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=b"HEVC!!",
            )
            slot_start = layout.mdat_payload_start + layout.slot_stride

            with (
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4", True),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_BACKEND", "slot"),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", False),
                patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_READY_ONLY", False),
                patch.object(routes_media, "PASSTHROUGH_GOP", 2),
                patch.object(routes_media, "RUNTIME_CACHE_DIR", Path(tmp) / "cache"),
                patch.object(
                    routes_media,
                    "_vmp4_slot_output_template",
                    return_value=Vmp4SlotOutputTemplate(
                        stsd=_stsd_video("hvc1"),
                        payload=b"HEVC!!",
                        codec_name="hevc",
                        nal_length_size=4,
                        width=3840,
                        height=2160,
                    ),
                ),
                patch.object(routes_media, "_safe_seek_video_path", return_value=(path, None, None)),
                patch.object(routes_media, "_seek_route_allowed", return_value=(True, "profile_allowed", "vlc")),
                patch.object(routes_media, "annotate_request", return_value=None),
                patch.object(routes_media, "probe_cached", return_value=info),
                patch.object(routes_media, "estimate_for_media", return_value=(1000, 2000, SimpleNamespace(source="test"))),
                patch.object(routes_media, "_vmp4_mode_candidate", side_effect=AssertionError("cache lookup not used")),
                patch.object(
                    routes_media,
                    "build_passthrough_vmp4_slot_layout",
                    wraps=build_passthrough_vmp4_slot_layout,
                ) as build_mock,
            ):
                routes_media._vmp4_slot_layout_cache.clear()
                first = asyncio.run(run(f"bytes={slot_start}-{slot_start + 5}"))
                second = asyncio.run(run(f"bytes={slot_start}-{slot_start + 5}"))

        self.assertEqual(first, 206)
        self.assertEqual(second, 206)
        self.assertEqual(build_mock.call_count, 1)

    def test_slot_build_queue_materializes_requested_slot_from_annexb(self) -> None:
        original_builds = dict(routes_media._vmp4_slot_builds)
        try:
            routes_media._vmp4_slot_builds.clear()
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "movie.mp4"
                _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
                layout = build_passthrough_vmp4_slot_layout(
                    path,
                    duration_sec=6.0,
                    fps=1.0,
                    total_size=path.stat().st_size,
                    gop_frames=2,
                    output_stsd=_stsd_video("hvc1"),
                    output_codec_name="hevc",
                    placeholder_payload=b"HEVC!!",
                )
                status = ensure_vmp4_slot_manifest(layout, root / "cache", output_mode="green", fps=1.0, gop_frames=2)
                annexb = b"\x00\x00\x00\x01\x26\x01"
                payload_bytes = hevc_annexb_to_length_prefixed_sample(annexb)
                fake_matter = object()

                with (
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                    patch.object(routes_media, "acquire_matter", return_value=fake_matter),
                    patch.object(routes_media, "release_matter", return_value=None) as release_mock,
                    patch.object(routes_media, "build_passthrough_hevc_annexb_gop", return_value=annexb) as build_mock,
                ):
                    view = routes_media._vmp4_slot_schedule_placeholder_build(layout, status, 1, "green")

                    payload = status.cache_dir / "slot_000001.bin"
                    deadline = time.time() + 2.0
                    while time.time() < deadline and (not payload.is_file() or release_mock.call_count == 0):
                        time.sleep(0.01)

                refreshed = ensure_vmp4_slot_manifest(layout, root / "cache", output_mode="green", fps=1.0, gop_frames=2)
                slot_start = layout.mdat_payload_start + layout.slot_stride
                body = b"".join(
                    iter_vmp4_slot_range(
                        layout,
                        slot_start,
                        slot_start + 5,
                        payload_paths=refreshed.ready_payloads,
                    )
                )

                self.assertIn(view.state, {"running", "ready"})
                self.assertTrue(payload.is_file())
                self.assertEqual(refreshed.slot(1).state, "ready")
                self.assertEqual(body, payload_bytes)
                build_mock.assert_called_once()
                self.assertEqual(build_mock.call_args.kwargs["start_sec"], layout.samples[1].start_time_sec)
                self.assertEqual(build_mock.call_args.kwargs["frame_count"], 1)
                self.assertEqual(build_mock.call_args.kwargs["matter"], fake_matter)
                self.assertEqual(build_mock.call_args.kwargs["output_mode"], "green")
                release_mock.assert_called_once_with(fake_matter)
        finally:
            routes_media._vmp4_slot_builds.clear()
            routes_media._vmp4_slot_builds.update(original_builds)

    def test_slot_build_failure_marks_failed_without_placeholder_fallback(self) -> None:
        original_builds = dict(routes_media._vmp4_slot_builds)
        try:
            routes_media._vmp4_slot_builds.clear()
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "movie.mp4"
                _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
                layout = build_passthrough_vmp4_slot_layout(
                    path,
                    duration_sec=6.0,
                    fps=1.0,
                    total_size=path.stat().st_size,
                    gop_frames=2,
                    output_stsd=_stsd_video("hvc1"),
                    output_codec_name="hevc",
                    placeholder_payload=b"HEVC!!",
                )
                status = ensure_vmp4_slot_manifest(layout, root / "cache", output_mode="green", fps=1.0, gop_frames=2)
                fake_matter = object()

                with (
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                    patch.object(routes_media, "acquire_matter", return_value=fake_matter),
                    patch.object(routes_media, "release_matter", return_value=None),
                    patch.object(routes_media, "build_passthrough_hevc_annexb_gop", side_effect=RuntimeError("gpu-failed")),
                ):
                    view = routes_media._vmp4_slot_schedule_placeholder_build(layout, status, 1, "green")

                    refreshed = ensure_vmp4_slot_manifest(layout, root / "cache", output_mode="green", fps=1.0, gop_frames=2)
                    deadline = time.time() + 2.0
                    while time.time() < deadline and refreshed.slot(1).state != "failed":
                        time.sleep(0.01)
                        refreshed = ensure_vmp4_slot_manifest(layout, root / "cache", output_mode="green", fps=1.0, gop_frames=2)

                payload = status.cache_dir / "slot_000001.bin"

                self.assertIn(view.state, {"running", "failed"})
                self.assertFalse(payload.exists())
                self.assertEqual(refreshed.slot(1).state, "failed")
                self.assertIn("gpu-failed", refreshed.slot(1).reason)
        finally:
            routes_media._vmp4_slot_builds.clear()
            routes_media._vmp4_slot_builds.update(original_builds)


    def test_slot_size_stays_tight_so_filler_lands_in_gap_not_access_unit(self) -> None:
        # A small per-slot sample cap is intentional: slot_size caps the stsz
        # entry (the access unit the decoder reads), and the rest of slot_stride
        # becomes unreferenced gap. Keeping slot_size < slot_stride means the
        # large source-size padding lands in the gap, not as multi-MB in-AU filler.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "movie.mp4"
            # target_size large vs payloads -> slot_stride much bigger than the cap
            _write_source_mp4(path, [b"A" * 16, b"B" * 16, b"C" * 16], target_size=6_000_000)
            layout = build_passthrough_vmp4_slot_layout(
                path,
                duration_sec=6.0,
                fps=1.0,
                total_size=path.stat().st_size,
                gop_frames=2,
                max_sample_bytes=256 * 1024,
                output_stsd=_stsd_video("hvc1"),
                output_codec_name="hevc",
                placeholder_payload=b"HEVC!!",
            )
        self.assertEqual(layout.slot_size, 256 * 1024)
        self.assertLess(layout.slot_size, layout.slot_stride)
        self.assertEqual(layout.slot_gap_size, layout.slot_stride - layout.slot_size)
        self.assertGreater(layout.slot_gap_size, 0)

    def _build_tiny_layout_and_status(self, root: Path):
        path = root / "movie.mp4"
        _write_source_mp4(path, [b"AAAA", b"BBBBBB", b"CCC"])
        layout = build_passthrough_vmp4_slot_layout(
            path,
            duration_sec=6.0,
            fps=1.0,
            total_size=path.stat().st_size,
            gop_frames=2,
            output_stsd=_stsd_video("hvc1"),
            output_codec_name="hevc",
            placeholder_payload=b"HEVC!!",
        )
        status = ensure_vmp4_slot_manifest(layout, root / "cache", output_mode="green", fps=1.0, gop_frames=2)
        return layout, status

    def _wait_failed(self, key: str, timeout: float = 2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            build = routes_media._vmp4_slot_builds.get(key)
            if build is not None and build.state == "failed":
                return build
            time.sleep(0.01)
        return routes_media._vmp4_slot_builds.get(key)

    def test_slot_build_permanent_oversize_failure_is_not_retried(self) -> None:
        original_builds = dict(routes_media._vmp4_slot_builds)
        try:
            routes_media._vmp4_slot_builds.clear()
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                layout, status = self._build_tiny_layout_and_status(root)
                key = routes_media._vmp4_slot_build_key(status, 1)
                with (
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                    patch.object(routes_media, "acquire_matter", return_value=object()),
                    patch.object(routes_media, "release_matter", return_value=None),
                    patch.object(routes_media, "build_passthrough_hevc_annexb_gop", return_value=b"\x00\x00\x00\x01\x26\x01") as build_mock,
                    patch.object(routes_media, "write_vmp4_slot_hevc_annexb_payload", side_effect=Vmp4SlotLayoutError("slot-payload-oversize")),
                ):
                    routes_media._vmp4_slot_schedule_placeholder_build(layout, status, 1, "green")
                    build = self._wait_failed(key)
                    self.assertIsNotNone(build)
                    self.assertEqual(build.state, "failed")
                    self.assertTrue(build.permanent)
                    self.assertEqual(build_mock.call_count, 1)
                    # A player's Retry-After loop must not re-run a permanent failure.
                    routes_media._vmp4_slot_schedule_placeholder_build(layout, status, 1, "green")
                    time.sleep(0.05)
                    self.assertEqual(build_mock.call_count, 1)
                    self.assertEqual(routes_media._vmp4_slot_builds[key].state, "failed")
        finally:
            routes_media._vmp4_slot_builds.clear()
            routes_media._vmp4_slot_builds.update(original_builds)

    def test_slot_build_transient_failure_waits_for_backoff_before_retry(self) -> None:
        original_builds = dict(routes_media._vmp4_slot_builds)
        try:
            routes_media._vmp4_slot_builds.clear()
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                layout, status = self._build_tiny_layout_and_status(root)
                key = routes_media._vmp4_slot_build_key(status, 1)
                with (
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_BUILD_PLACEHOLDER", True),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_RETRY_BASE", 30.0),
                    patch.object(routes_media, "PASSTHROUGH_SEEK_VMP4_SLOT_RETRY_MAX", 60.0),
                    patch.object(routes_media, "acquire_matter", return_value=object()),
                    patch.object(routes_media, "release_matter", return_value=None),
                    patch.object(routes_media, "build_passthrough_hevc_annexb_gop", side_effect=RuntimeError("gpu-hiccup")) as build_mock,
                ):
                    routes_media._vmp4_slot_schedule_placeholder_build(layout, status, 1, "green")
                    build = self._wait_failed(key)
                    self.assertIsNotNone(build)
                    self.assertFalse(build.permanent)
                    self.assertGreater(build.next_retry_at, time.time())
                    self.assertEqual(build_mock.call_count, 1)
                    # Within the backoff window the transient failure is not retried.
                    routes_media._vmp4_slot_schedule_placeholder_build(layout, status, 1, "green")
                    time.sleep(0.05)
                    self.assertEqual(build_mock.call_count, 1)
                    # Once the backoff elapses (simulated), the next request retries.
                    routes_media._vmp4_slot_builds[key].next_retry_at = time.time() - 1.0
                    routes_media._vmp4_slot_schedule_placeholder_build(layout, status, 1, "green")
                    deadline = time.time() + 2.0
                    while time.time() < deadline and build_mock.call_count < 2:
                        time.sleep(0.01)
                    self.assertEqual(build_mock.call_count, 2)
        finally:
            routes_media._vmp4_slot_builds.clear()
            routes_media._vmp4_slot_builds.update(original_builds)


if __name__ == "__main__":
    unittest.main()


class AnnexBStartCodeScanTests(unittest.TestCase):
    """The find()-based scan must match the byte-by-byte one it replaced."""

    @staticmethod
    def _reference(data: bytes):
        i, n = 0, len(data)
        while i + 3 <= n:
            if i + 4 <= n and data[i:i + 4] == b"\x00\x00\x00\x01":
                yield i, i + 4
                i += 4
            elif data[i:i + 3] == b"\x00\x00\x01":
                yield i, i + 3
                i += 3
            else:
                i += 1

    def _assert_same(self, data: bytes) -> None:
        from pipeline.passthrough_vmp4_slot import _annexb_start_codes

        self.assertEqual(
            list(self._reference(data)), list(_annexb_start_codes(data)), data[:48]
        )

    def test_edge_shapes(self) -> None:
        for data in (
            b"",
            b"\x00",
            b"\x00\x00\x01",
            b"\x00\x00\x00\x01",
            b"\x00\x00\x00\x00\x01",          # 3 zeros: prefix starts at index 1
            b"\x00\x00\x00\x00\x00\x01",      # 4 zeros
            b"\x00\x00\x01ab\x00\x00\x00\x01cd\x00\x00\x01",
            b"ab\x00\x00\x01\x00\x00\x00\x01\x00\x00\x01xy",
            b"\x01\x00\x00\x00\x00\x00\x01\x00",
        ):
            self._assert_same(data)

    def test_random_streams_dense_in_zeros(self) -> None:
        import random

        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(0, 300)
            self._assert_same(
                bytes(rng.choice([0, 0, 0, 1, rng.randint(0, 255)]) for _ in range(n))
            )

    def test_round_trip_through_a_length_prefixed_sample(self) -> None:
        from pipeline.passthrough_vmp4_slot import (
            hevc_annexb_to_length_prefixed_sample,
            iter_annexb_nal_units,
        )

        nals = [b"\x26\x01payload-one", b"\x02\x01payload-two"]
        stream = b"".join(b"\x00\x00\x00\x01" + n for n in nals)
        self.assertEqual(list(iter_annexb_nal_units(stream)), nals)
        sample = hevc_annexb_to_length_prefixed_sample(stream, nal_length_size=4)
        cursor, seen = 0, []
        while cursor < len(sample):
            size = int.from_bytes(sample[cursor:cursor + 4], "big")
            cursor += 4
            seen.append(sample[cursor:cursor + size])
            cursor += size
        self.assertEqual(seen, nals)
