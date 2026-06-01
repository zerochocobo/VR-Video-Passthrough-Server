"""Player compatibility shadow classification helpers.

This module intentionally separates two concepts:

- ``route_profile`` preserves the existing route-level live behavior labels
  used by ``http_app.routes_media`` (for example ``libmpv`` or ``lavf``).
- ``profile_class`` is the higher-level behavior class used for diagnostics
  and future policy tables.

The functions here are observation-oriented. They do not acquire GPU resources
or change response behavior.
"""
from __future__ import annotations

import re
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import unquote


PROFILE_AUTO = "auto"
PROFILE_VLC_LIKE = "vlc_like"
PROFILE_LIBMPV_LIKE = "libmpv_like"
PROFILE_NPLAYER_LIKE = "nplayer_like"
PROFILE_QUEST_AVPRO_LIKE = "quest_avpro_like"
PROFILE_STRICT_LIVE = "strict_live"
PROFILE_UNKNOWN = "unknown"

INTENT_BROWSE = "browse"
INTENT_METADATA = "metadata"
INTENT_THUMBNAIL_ENDPOINT = "thumbnail_endpoint"
INTENT_RAW_MEDIA_PREVIEW = "raw_media_preview"
INTENT_STARTUP_PROBE = "startup_probe"
INTENT_DUPLICATE_STARTUP = "duplicate_startup"
INTENT_TAIL_PROBE = "tail_probe"
INTENT_SIDE_PROBE = "side_probe"
INTENT_PLAYBACK_PRIMARY = "playback_primary"
INTENT_SUBTITLE = "subtitle"
INTENT_UNKNOWN = "unknown"

DECISION_OBSERVE_ONLY = "observe_only"
DECISION_ALLOW_GPU = "allow_gpu"
DECISION_REJECT_OR_CACHE_BEFORE_GPU = "reject_or_cache_before_gpu"
DECISION_REUSE_SESSION_OR_DEBOUNCE = "reuse_session_or_debounce"

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)", re.IGNORECASE)
_SESSION_TTL_SEC = 300.0
_SESSION_MAX = 128
_TAIL_PROBE_RATIO = 0.95
_TAIL_PROBE_MAX_BYTES = 512 * 1024


@dataclass(frozen=True)
class ByteRangeShape:
    start: int | None = None
    end: int | None = None
    suffix: bool = False
    open_ended: bool = False
    valid: bool = False

    @property
    def is_zero_open(self) -> bool:
        return self.valid and self.start == 0 and self.open_ended and not self.suffix

    @property
    def is_nonzero_open(self) -> bool:
        return self.valid and self.start is not None and self.start > 0 and self.open_ended

    @property
    def is_small_fixed(self) -> bool:
        if not self.valid or self.start is None or self.end is None or self.open_ended or self.suffix:
            return False
        return max(0, self.end - self.start + 1) <= 64 * 1024


@dataclass(frozen=True)
class ProfileMatch:
    route_profile: str
    profile_class: str
    confidence: float
    fired_rules: tuple[str, ...]


@dataclass(frozen=True)
class IntentMatch:
    intent: str
    confidence: float
    fired_rules: tuple[str, ...]


@dataclass(frozen=True)
class CompatibilityDecision:
    profile: ProfileMatch
    intent: IntentMatch
    decision: str
    fired_rules: tuple[str, ...]


@dataclass
class DeviceSession:
    key: str
    created_at: float
    updated_at: float
    forced_profile_class: str | None = None
    profile_class: str = PROFILE_UNKNOWN
    route_profile: str = ""
    confidence: float = 0.0
    fired_rules: tuple[str, ...] = ()
    # Count since this in-memory session was created; not a rolling window.
    side_probe_total: int = 0
    side_probe_observed_at: float | None = None
    last_32_paths: deque[str] = field(default_factory=lambda: deque(maxlen=32))
    recent_uas: deque[str] = field(default_factory=lambda: deque(maxlen=8))

    def observe(
        self,
        *,
        path: str,
        user_agent: str,
        route_profile: str,
        profile_class: str,
        confidence: float,
        fired_rules: Iterable[str],
        observed_at: float | None = None,
    ) -> None:
        observed_at = observed_at if observed_at is not None else time.time()
        self.updated_at = observed_at
        if path:
            self.last_32_paths.append(path)
        if user_agent and (not self.recent_uas or self.recent_uas[-1] != user_agent):
            self.recent_uas.append(user_agent)
        if route_profile == "lavf" or is_lavf_user_agent(user_agent):
            self.side_probe_total += 1
            self.side_probe_observed_at = observed_at
            return
        if self.forced_profile_class:
            self.profile_class = self.forced_profile_class
            self.route_profile = route_profile
            self.confidence = 1.0
            self.fired_rules = ("forced_profile_class",)
            return
        # Deliberately do not let UNKNOWN overwrite a known profile; unknown
        # observations age out through the DeviceSession TTL instead.
        changed_player_signal = (
            profile_class != PROFILE_UNKNOWN
            and profile_class != self.profile_class
            and confidence >= 0.75
        )
        if confidence >= self.confidence or self.profile_class == PROFILE_UNKNOWN or changed_player_signal:
            self.profile_class = profile_class
            self.route_profile = route_profile
            self.confidence = confidence
            self.fired_rules = tuple(fired_rules)


_sessions: dict[str, DeviceSession] = {}
_host_primary_session_keys: dict[str, str] = {}


def _drop_session(key: str) -> None:
    _sessions.pop(key, None)
    for host, primary_key in list(_host_primary_session_keys.items()):
        if primary_key == key:
            _host_primary_session_keys.pop(host, None)


def parse_range_shape(value: str | None) -> ByteRangeShape:
    if not value:
        return ByteRangeShape()
    match = _RANGE_RE.match(value.strip())
    if not match:
        return ByteRangeShape()
    left, right = match.group(1), match.group(2)
    if not left and not right:
        return ByteRangeShape(valid=True)
    if not left:
        return ByteRangeShape(end=int(right), suffix=True, valid=True)
    return ByteRangeShape(
        start=int(left),
        end=int(right) if right else None,
        open_ended=not bool(right),
        valid=True,
    )


def live_response_profile_from_ua(user_agent: str, default_profile: str = "vlc") -> str:
    """Preserve the existing /passthrough_live route profile mapping."""
    ua = (user_agent or "").lower()
    if "nplayer" in ua:
        return "nplayer"
    if "avpromobilevideo" in ua or "exoplayerlib" in ua:
        return "avpro"
    if "libmpv" in ua or "skybox" in ua:
        return "libmpv"
    if "heresphere" in ua:
        return "4xvr"
    if "dalvik/" in ua:
        return "4xvr"
    if "vlc" in ua or "libvlc" in ua or "moonvr" in ua:
        return "vlc"
    if "lavf/" in ua:
        return "lavf"
    return default_profile


def is_nplayer_user_agent(user_agent: str) -> bool:
    return "nplayer" in (user_agent or "").lower()


def is_lavf_user_agent(user_agent: str) -> bool:
    return "lavf/" in (user_agent or "").lower()


def is_libmpv_screenshot_probe_ua(user_agent: str) -> bool:
    """Skybox sends a bare ``libmpv`` User-Agent (no version, no other
    tokens) when it is generating chapter thumbnails: it fires one
    ``/passthrough_live?t=<chapter>`` request per chapter time-offset in a
    tight burst, reads only the prefix, then disconnects. The actual Skybox
    playback path uses ``SKYBOX/x.y.z``. Real mpv builds advertise
    ``libmpv/<version>`` or include other tokens. Match exactly ``libmpv``
    (case-insensitive) so only Skybox's probe pattern is diverted.
    """
    return (user_agent or "").strip().lower() == "libmpv"


def is_skybox_player_ua(user_agent: str) -> bool:
    """The real Skybox playback path uses ``SKYBOX/x.y.z`` UA. Skybox is
    strict about DLNA live vs VOD signalling: the same response that other
    VR players accept (TimeSeekRange + finite duration + chunked transfer
    without Content-Length) makes Skybox treat the resource as VOD, then
    fail because the byte-Range/Content-Length VOD contract is not honored.
    Detect SKYBOX here so the live response can be stripped down to pure-
    live signalling for it without touching other player paths.
    """
    return "skybox" in (user_agent or "").lower()


def profile_class_for_route_profile(route_profile: str) -> str:
    if route_profile in {"nplayer"}:
        return PROFILE_NPLAYER_LIKE
    if route_profile in {"avpro", "4xvr"}:
        return PROFILE_QUEST_AVPRO_LIKE
    if route_profile == "libmpv":
        return PROFILE_LIBMPV_LIKE
    if route_profile in {"vlc", "default"}:
        return PROFILE_VLC_LIKE
    if route_profile == "lavf":
        # Lavf is a side-request signal, not a player behavior class.
        return PROFILE_UNKNOWN
    return PROFILE_UNKNOWN


def match_profile(user_agent: str, default_profile: str = "vlc") -> ProfileMatch:
    route_profile = live_response_profile_from_ua(user_agent, default_profile)
    profile_class = profile_class_for_route_profile(route_profile)
    ua = (user_agent or "").lower()
    rules: list[str] = []
    confidence = 0.35
    if "nplayer" in ua:
        rules.append("ua_contains_nplayer")
        confidence = 0.95
    elif "avpromobilevideo" in ua or "exoplayerlib" in ua:
        rules.append("ua_contains_avpro_or_exoplayer")
        confidence = 0.9
    elif "libmpv" in ua or "skybox" in ua:
        rules.append("ua_contains_libmpv_or_skybox")
        confidence = 0.9
    elif "heresphere" in ua:
        rules.append("ua_contains_heresphere")
        confidence = 0.9
    elif "dalvik/" in ua:
        rules.append("ua_contains_dalvik")
        confidence = 0.75
    elif "vlc" in ua or "libvlc" in ua or "moonvr" in ua:
        rules.append("ua_contains_vlc_or_moonvr")
        confidence = 0.85
    elif "lavf/" in ua:
        rules.append("ua_contains_lavf_side_signal")
        confidence = 0.7
    else:
        rules.append("default_live_profile")
    return ProfileMatch(route_profile, profile_class, confidence, tuple(rules))


def path_kind(path: str) -> str:
    clean = unquote((path or "").split("?", 1)[0])
    if clean == "/control/cds":
        return "cds"
    if clean == "/control/cm":
        return "cm_control"
    if clean in {"/description.xml", "/cds.xml", "/cm.xml"}:
        return "metadata"
    if clean.startswith("/thumb/"):
        return "thumb"
    if clean.startswith("/media/"):
        return "media"
    if clean.startswith("/passthrough_live/"):
        return "passthrough_live"
    if clean.startswith("/passthrough/"):
        return "passthrough"
    if clean.startswith("/subs/"):
        return "subs"
    return "other"


def _annotation_int(annotations: dict, key: str) -> int:
    try:
        return max(0, int(annotations.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def _is_tail_probe_shape(range_shape: ByteRangeShape, total_size: int) -> bool:
    if not range_shape.valid:
        return False
    if range_shape.suffix:
        return (range_shape.end or 0) <= _TAIL_PROBE_MAX_BYTES
    if total_size <= 0 or range_shape.start is None or range_shape.end is None:
        return False
    length = max(0, range_shape.end - range_shape.start + 1)
    return (
        range_shape.start > 0
        and range_shape.start >= int(total_size * _TAIL_PROBE_RATIO)
        and length <= _TAIL_PROBE_MAX_BYTES
    )


def match_intent(
    *,
    method: str,
    path: str,
    user_agent: str = "",
    range_header: str | None = None,
    route_profile: str = "",
    annotations: dict | None = None,
) -> IntentMatch:
    annotations = annotations or {}
    kind = path_kind(path)
    range_shape = parse_range_shape(range_header)
    rules: list[str] = []

    if kind == "cds":
        browse_flag = str(annotations.get("BrowseFlag") or annotations.get("browse_flag") or "")
        rules.append("path_control_cds")
        if browse_flag:
            rules.append(f"browse_flag_{browse_flag}")
        return IntentMatch(INTENT_BROWSE, 0.9, tuple(rules))
    if kind in {"metadata", "cm_control"}:
        return IntentMatch(INTENT_METADATA, 0.9, ("path_metadata",))
    if kind == "thumb":
        return IntentMatch(INTENT_THUMBNAIL_ENDPOINT, 1.0, ("path_thumb",))
    if kind == "subs":
        return IntentMatch(INTENT_SUBTITLE, 0.95, ("path_subs",))
    if kind == "media":
        rules.append("path_media_no_gpu")
        if method.upper() == "HEAD" or range_shape.is_small_fixed or range_shape.is_nonzero_open:
            rules.append("media_preview_shape")
            return IntentMatch(INTENT_RAW_MEDIA_PREVIEW, 0.65, tuple(rules))
        return IntentMatch(INTENT_PLAYBACK_PRIMARY, 0.45, tuple(rules))

    if kind in {"passthrough_live", "passthrough"}:
        if is_lavf_user_agent(user_agent) or route_profile == "lavf":
            return IntentMatch(INTENT_SIDE_PROBE, 0.85, ("ua_lavf_side_probe",))
        if annotations.get("duplicate_startup"):
            return IntentMatch(INTENT_DUPLICATE_STARTUP, 0.8, ("duplicate_startup_annotation",))
        total_size = _annotation_int(annotations, "total_estimated_size")
        if _is_tail_probe_shape(range_shape, total_size):
            return IntentMatch(INTENT_TAIL_PROBE, 0.75, ("tail_range_ratio",))
        if range_shape.is_nonzero_open:
            return IntentMatch(INTENT_STARTUP_PROBE, 0.7, ("nonzero_open_range",))
        return IntentMatch(INTENT_PLAYBACK_PRIMARY, 0.55, ("passthrough_candidate",))

    return IntentMatch(INTENT_UNKNOWN, 0.2, ("unknown_path",))


def decide_shadow(profile: ProfileMatch, intent: IntentMatch) -> CompatibilityDecision:
    rules = [*profile.fired_rules, *intent.fired_rules]
    if intent.intent in {INTENT_BROWSE, INTENT_METADATA, INTENT_THUMBNAIL_ENDPOINT, INTENT_SUBTITLE, INTENT_RAW_MEDIA_PREVIEW}:
        decision = DECISION_OBSERVE_ONLY
    elif intent.intent in {INTENT_SIDE_PROBE, INTENT_TAIL_PROBE, INTENT_STARTUP_PROBE}:
        decision = DECISION_REJECT_OR_CACHE_BEFORE_GPU
    elif intent.intent == INTENT_DUPLICATE_STARTUP:
        decision = DECISION_REUSE_SESSION_OR_DEBOUNCE
    elif intent.intent == INTENT_PLAYBACK_PRIMARY:
        decision = DECISION_ALLOW_GPU
    else:
        decision = DECISION_OBSERVE_ONLY
    return CompatibilityDecision(profile, intent, decision, tuple(rules))


def _hash6(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:6]


def session_key(client_host: str, user_agent: str = "") -> str:
    host = client_host or "<unknown>"
    ua = (user_agent or "").strip()
    if not ua:
        return f"{host}|ua:none"
    return f"{host}|ua:{_hash6(ua)}"


def observe_device_session(
    *,
    client_host: str,
    path: str,
    user_agent: str,
    profile: ProfileMatch,
    now: float | None = None,
) -> DeviceSession:
    now = now if now is not None else time.time()
    stale = [key for key, session in _sessions.items() if now - session.updated_at > _SESSION_TTL_SEC]
    for key in stale:
        _drop_session(key)
    while len(_sessions) > _SESSION_MAX:
        oldest = min(_sessions.items(), key=lambda item: item[1].updated_at)[0]
        _drop_session(oldest)

    host = client_host or "<unknown>"
    is_lavf = profile.route_profile == "lavf" or is_lavf_user_agent(user_agent)
    # Lavf has no reliable player identity by itself. Attach it to the most
    # recently observed primary player for this host; if Lavf arrives first,
    # keep a temporary Lavf-keyed session so early side-probe evidence is not
    # lost. That orphan expires through the normal TTL once a primary appears.
    key = _host_primary_session_keys.get(host) if is_lavf else session_key(client_host, user_agent)
    if not key:
        key = session_key(client_host, user_agent)
    session = _sessions.get(key)
    if session is None:
        session = DeviceSession(key=key, created_at=now, updated_at=now)
        _sessions[key] = session
    session.observe(
        path=path,
        user_agent=user_agent,
        route_profile=profile.route_profile,
        profile_class=profile.profile_class,
        confidence=profile.confidence,
        fired_rules=profile.fired_rules,
        observed_at=now,
    )
    if not is_lavf:
        _host_primary_session_keys[host] = key
    return session


def clear_device_sessions() -> None:
    _sessions.clear()
    _host_primary_session_keys.clear()


def clear_device_sessions_for_test() -> None:
    clear_device_sessions()


def classify_request_shadow(
    *,
    method: str,
    path: str,
    headers: dict[str, str],
    client_host: str = "",
    annotations: dict | None = None,
    default_profile: str = "vlc",
) -> CompatibilityDecision:
    normalized_headers = {str(key).lower(): str(value) for key, value in headers.items()}
    user_agent = normalized_headers.get("user-agent", "")
    profile = match_profile(user_agent, default_profile)
    observe_device_session(client_host=client_host, path=path, user_agent=user_agent, profile=profile)
    intent = match_intent(
        method=method,
        path=path,
        user_agent=user_agent,
        range_header=normalized_headers.get("range"),
        route_profile=profile.route_profile,
        annotations=annotations,
    )
    return decide_shadow(profile, intent)


def replay_scenario(records: Iterable[dict], default_profile: str = "vlc") -> list[CompatibilityDecision]:
    decisions: list[CompatibilityDecision] = []
    for record in records:
        headers = record.get("headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        decisions.append(
            classify_request_shadow(
                method=str(record.get("method") or "GET"),
                path=str(record.get("path") or ""),
                headers={str(k): str(v) for k, v in headers.items()},
                client_host=str(record.get("client_host") or ""),
                annotations=record.get("annotations") if isinstance(record.get("annotations"), dict) else {},
                default_profile=default_profile,
            )
        )
    return decisions
