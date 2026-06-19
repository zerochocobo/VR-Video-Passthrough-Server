"""Bounded DLNA/media request history for player-compat diagnostics."""
from __future__ import annotations

import itertools
import json
import hashlib
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode

from starlette.requests import Request

from utils.player_compat import classify_request_shadow


_trace_ids = itertools.count(1)
_trace_token = secrets.token_hex(3)
_STATE_KEY = "pt_request_history_annotations"
_MEDIA_ROUTE_PREFIXES = (
    "/media/",
    "/thumb/",
    "/subs/",
    "/media_si/",
    "/passthrough/",
    "/passthrough_live/",
)
_SENSITIVE_ANNOTATION_KEYS = {
    "media_path",
    "media_name",
    "ObjectID",
    "object_id",
}
# New URL query parameters should be reviewed for sensitivity before adding
# them here; omitted keys are dropped when request-history redaction is enabled.
_QUERY_REDACT_KEEP_KEYS = {"mode", "ptv"}


@dataclass
class RequestRecord:
    trace_id: str
    ts: str
    method: str
    path: str
    query: str
    client_host: str
    user_agent: str
    range: str
    time_seek_range: str
    transfer_mode: str
    get_content_features: str
    status_code: int
    elapsed_ms: float
    content_length: str = ""
    annotations: dict[str, Any] = field(default_factory=dict)
    route_profile: str = ""
    profile_class: str = ""
    profile_confidence: float = 0.0
    intent: str = ""
    intent_confidence: float = 0.0
    decision: str = ""
    fired_rules: tuple[str, ...] = ()

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fired_rules"] = list(self.fired_rules)
        return payload


class RequestHistory:
    def __init__(
        self,
        *,
        max_records: int = 500,
        jsonl_dir: Path | None = None,
        jsonl_enabled: bool = False,
        flush_every: int = 16,
        redact: bool = True,
    ) -> None:
        self.max_records = max(0, int(max_records))
        self.jsonl_dir = jsonl_dir
        self.jsonl_enabled = bool(jsonl_enabled)
        self.flush_every = max(1, int(flush_every))
        self.redact = bool(redact)
        self._records: deque[RequestRecord] = deque(maxlen=self.max_records)
        self._pending: list[RequestRecord] = []
        self._lock = threading.Lock()

    def add(self, record: RequestRecord) -> None:
        with self._lock:
            if self.max_records > 0:
                self._records.append(record)
            if self.jsonl_enabled and self.jsonl_dir is not None:
                self._pending.append(record)
                if len(self._pending) >= self.flush_every:
                    self._flush_locked()

    def snapshot(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._records)
        if limit is not None and limit >= 0:
            items = items[-limit:]
        return [item.to_jsonable() for item in items]

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def clear_for_test(self) -> None:
        with self._lock:
            self._records.clear()
            self._pending.clear()

    def _flush_locked(self) -> None:
        if not self._pending or self.jsonl_dir is None:
            return
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        name = f"request_history_{datetime.now().strftime('%Y%m%d')}.jsonl"
        out = self.jsonl_dir / name
        lines = [
            json.dumps(record.to_jsonable(), ensure_ascii=False, separators=(",", ":"))
            for record in self._pending
        ]
        with out.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
                f.write("\n")
        self._pending.clear()


_history = RequestHistory(max_records=0)


def configure_request_history(
    *,
    enabled: bool,
    max_records: int,
    jsonl_enabled: bool,
    jsonl_dir: Path,
    flush_every: int,
    redact: bool = True,
) -> None:
    global _history
    _history = RequestHistory(
        max_records=max_records if enabled else 0,
        jsonl_dir=jsonl_dir,
        jsonl_enabled=enabled and jsonl_enabled,
        flush_every=flush_every,
        redact=redact,
    )


def get_request_history() -> RequestHistory:
    return _history


def next_trace_id() -> str:
    return f"pt-{_trace_token}-{next(_trace_ids):08d}"


def _sha8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:8]


def _redact_client_host(client_host: str) -> str:
    return f"client-{_sha8(client_host)}" if client_host else ""


def _redact_media_route_path(path: str) -> str:
    for prefix in _MEDIA_ROUTE_PREFIXES:
        if path.startswith(prefix) and len(path) > len(prefix):
            return f"{prefix}<media-{_sha8(path[len(prefix):])}>"
    return path


def _redact_annotations(annotations: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(annotations)
    for key in _SENSITIVE_ANNOTATION_KEYS:
        if key in redacted and redacted[key] not in (None, ""):
            redacted[key] = f"<{key}-{_sha8(str(redacted[key]))}>"
    return redacted


def _redact_query(query: str) -> str:
    if not query:
        return ""
    kept = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key in _QUERY_REDACT_KEEP_KEYS
    ]
    return urlencode(kept)


def annotate_request(request: Request, **values: Any) -> None:
    current = getattr(request.state, _STATE_KEY, None)
    if current is None:
        current = {}
        setattr(request.state, _STATE_KEY, current)
    current.update({key: value for key, value in values.items() if value is not None})


def request_annotations(request: Request) -> dict[str, Any]:
    current = getattr(request.state, _STATE_KEY, None)
    return dict(current) if isinstance(current, dict) else {}


def build_record(
    *,
    request: Request,
    trace_id: str,
    status_code: int,
    elapsed_ms: float,
    content_length: str = "",
    default_profile: str = "vlc",
    redact: bool | None = None,
) -> RequestRecord:
    headers = {key.lower(): value for key, value in request.headers.items()}
    annotations = request_annotations(request)
    client_host = request.client.host if request.client else ""
    raw_path = request.url.path
    decision = classify_request_shadow(
        method=request.method,
        path=raw_path,
        headers=headers,
        client_host=client_host,
        annotations=annotations,
        default_profile=default_profile,
    )
    if redact is None:
        redact = get_request_history().redact
    output_path = _redact_media_route_path(raw_path) if redact else raw_path
    output_client_host = _redact_client_host(client_host) if redact else client_host
    output_annotations = _redact_annotations(annotations) if redact else annotations
    output_query = _redact_query(request.url.query) if redact else request.url.query
    profile = decision.profile
    intent = decision.intent
    return RequestRecord(
        trace_id=trace_id,
        ts=datetime.now().isoformat(timespec="milliseconds"),
        method=request.method,
        path=output_path,
        query=output_query,
        client_host=output_client_host,
        user_agent=headers.get("user-agent", ""),
        range=headers.get("range", ""),
        time_seek_range=headers.get("timeseekrange.dlna.org", ""),
        transfer_mode=headers.get("transfermode.dlna.org", ""),
        get_content_features=headers.get("getcontentfeatures.dlna.org", ""),
        status_code=int(status_code),
        elapsed_ms=round(float(elapsed_ms), 3),
        content_length=content_length,
        annotations=output_annotations,
        route_profile=profile.route_profile,
        profile_class=profile.profile_class,
        profile_confidence=profile.confidence,
        intent=intent.intent,
        intent_confidence=intent.confidence,
        decision=decision.decision,
        fired_rules=decision.fired_rules,
    )


def record_request(
    *,
    request: Request,
    trace_id: str,
    status_code: int,
    started_at: float,
    content_length: str = "",
    default_profile: str = "vlc",
) -> RequestRecord:
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    record = build_record(
        request=request,
        trace_id=trace_id,
        status_code=status_code,
        elapsed_ms=elapsed_ms,
        content_length=content_length,
        default_profile=default_profile,
    )
    _history.add(record)
    return record
