# Player Compatibility Strategy Review

Date: 2026-06-23

## Background

A user reported that 4XVR on Apple Vision Pro cannot play realtime passthrough streams. The current compatibility work has mainly been validated on Android/Quest headsets, not Apple Vision Pro. The likely first hypothesis is that the Apple-side 4XVR User-Agent does not match the current rules and falls back to the default live route profile.

This pass first reviewed the current strategy. After the logs confirmed that Apple Vision Pro 4XVR uses `Vision4XVR/2 CFNetwork/3860.600.12 Darwin/25.5.0`, a UA rule was added so clients carrying a `4xvr` token now use the `4xvr` route profile.

## Current Player Detection

The main mapping lives in `utils/player_compat.py` in `live_response_profile_from_ua()`:

- `nPlayer` -> `nplayer`
- `AVProMobileVideo` / `ExoPlayerLib` -> `avpro`
- `libmpv` / `skybox` -> `libmpv`
- `HereSphere` -> `4xvr`
- `Dalvik/` -> `4xvr`
- `VLC` / `LibVLC` / `MoonVR` -> `vlc`
- `Lavf/` -> `lavf`
- Unknown UA -> `PASSTHROUGH_LIVE_DEFAULT_PROFILE`, currently `vlc`

Added from the 2026-06-23 real-device logs:

- `Vision4XVR/2 CFNetwork/... Darwin/...` and other UA strings containing `4xvr` -> `4xvr`

The diagnostic layer, including `match_profile()`, `match_intent()`, and `decide_shadow()`, is currently used for request history and audit fields. It does not directly change playback behavior.

## Route Behavior Differences

`/media/{name}` is a raw file Range route and does not start GPU work. It is useful as a preview/screenshot behavior signal, but it is not the realtime GPU risk path.

`/passthrough_live/{name}` is the main realtime MPEG-TS path and can acquire GPU/NVENC resources.

Current live route behavior:

- `vlc` default route: direct streaming, no managed `LiveSession`; non-zero live Range requests are rejected with `416` before startup; VLC/MoonVR preroll is used to reduce HEVC TS audio-only startup risk.
- `4xvr` / `avpro` route: uses managed `LiveSession`; allows same-device new live requests to replace old streams; better matches AVPro/ExoPlayer-style reconnect behavior.
- `libmpv` / Skybox route: uses managed `LiveSession`, prefix cache, and startup debounce; bare `libmpv` screenshot probes return fast `503` so they do not consume GPU or playback bandwidth.
- `nplayer` route: uses managed `LiveSession`; ignores live Range for session key stability; has duplicate-startup debounce and a stall watchdog.
- `lavf` route: rejected by default as a side/probe request to protect the realtime production path.

## DLNA Directory Strategy

DLNA directory shaping is separate from route profile detection.

Except for a narrow DeoVR fingerprint, live URLs get a `.ts` suffix so Skybox selects its MPEG-TS pipeline. The route strips `.ts`, `.m2ts`, and `.mpegts` before resolving the actual source file.

For non-DeoVR clients, live DIDL uses a more pure-live shape with `DLNA.ORG_OP=00` and omits file-like duration/bitrate attributes. DeoVR keeps the legacy shape: no `.ts` suffix, `DLNA.ORG_OP=10`, and duration/bitrate attributes.

## Vision Pro 4XVR Assessment

If Apple-side 4XVR does not include existing match tokens such as `AVProMobileVideo`, `ExoPlayerLib`, `Dalvik/`, or `HereSphere`, it falls into the default `vlc` profile.

That route is materially different from the Android/Quest 4XVR `4xvr`/`avpro` behavior. If Apple 4XVR sends non-zero startup Range requests, repeats the same live URL, or depends on reconnect/session reuse, the default `vlc` route can plausibly fail.

The provided logs confirmed that this Vision Pro 4XVR request hit the default `vlc` profile:

- CDS directory request UA: `Darwin/25.5.0, UPnP/1.0, Portable SDK for UPnP devices/1.14.21`
- UA: `Vision4XVR/2 CFNetwork/3860.600.12 Darwin/25.5.0`
- route profile: `vlc`
- path: `/passthrough_live/...`
- the server did enter the PyNv realtime MPEG-TS production path and returned `200`

The immediate fix is therefore to map the `4xvr` UA token to the `4xvr` profile, not to change the global default profile.

## Recommendation

Do not immediately change the global default route from `vlc` to `4xvr`/`avpro`. The default affects all unknown players, while the `4xvr`/`avpro` route is more permissive around managed sessions, same-device takeover, and Range tolerance. That could change resource ownership behavior for unrelated unknown clients.

The safer next step is to collect Vision Pro 4XVR request history and confirm the UA, Range headers, request sequence, and status codes. If this is only a UA miss, add the Apple 4XVR UA token to the `4xvr` or `avpro` route instead of changing the unknown default.

If a broader fallback is needed later, add an explicit compatibility mode or device-level profile binding so one device can be forced to `4xvr`/`avpro` behavior while still preserving probe/side/tail intent protection.
