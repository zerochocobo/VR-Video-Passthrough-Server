# 播放器兼容策略梳理

日期：2026-06-23

## 背景

用户反馈 Apple Vision Pro 上的 4XVR 播放器无法播放。当前项目主要验证过 Android/Quest 头显上的播放器行为，未覆盖 Apple Vision Pro。初步怀疑 Apple 端 4XVR 的 User-Agent 未命中现有识别规则，导致 `/passthrough_live` 走默认路线。

本轮先整理现有策略。随后根据日志确认 Apple Vision Pro 4XVR 的 UA 为 `Vision4XVR/2 CFNetwork/3860.600.12 Darwin/25.5.0`，并已补充 UA 识别规则，让带 `4xvr` 标识的客户端走 `4xvr` route profile。

## 当前播放器识别

核心入口在 `utils/player_compat.py` 的 `live_response_profile_from_ua()`：

- `nPlayer` -> `nplayer`
- `AVProMobileVideo` / `ExoPlayerLib` -> `avpro`
- `libmpv` / `skybox` -> `libmpv`
- `HereSphere` -> `4xvr`
- `Dalvik/` -> `4xvr`
- `VLC` / `LibVLC` / `MoonVR` -> `vlc`
- `Lavf/` -> `lavf`
- 未识别 UA -> `PASSTHROUGH_LIVE_DEFAULT_PROFILE`，当前默认 `vlc`

2026-06-23 根据实机日志新增：

- `Vision4XVR/2 CFNetwork/... Darwin/...` 这类包含 `4xvr` 的 UA -> `4xvr`

诊断层的 `match_profile()`、`match_intent()`、`decide_shadow()` 会写入 request history，但目前主要是观察和审计，不直接改变播放行为。

## 不同播放路线差异

`/media/{name}` 是原始文件 Range 服务，不启动 GPU。它可以作为截图、预览和行为识别信号，但不是实时 GPU 风险点。

`/passthrough_live/{name}` 是主要实时 MPEG-TS 播放路径，会占用 GPU/NVENC。

当前 live route 的关键差异：

- `vlc` 默认路线：直接 streaming，不启用 managed `LiveSession`；非起点 Range 会在启动前返回 `416`；有 VLC/MoonVR preroll，用于降低 HEVC TS audio-only 风险。
- `4xvr` / `avpro` 路线：启用 managed `LiveSession`；允许同设备新 live 请求接管旧流；更适合 AVPro/ExoPlayer 类客户端的重连行为。
- `libmpv` / Skybox 路线：启用 managed `LiveSession`、prefix cache 和 startup debounce；裸 `libmpv` 截图探测会快速 `503`，避免抢 GPU 或挤占真实播放带宽。
- `nplayer` 路线：启用 managed `LiveSession`；忽略 live Range 对 session key 的影响；有重复启动 debounce 和 stall watchdog。
- `lavf` 路线：默认 `reject`，作为副请求/探测保护，避免抢占实时生产链路。

## DLNA 目录层策略

DLNA 目录输出与 UA route profile 是两层不同策略。

除 DeoVR 窄指纹外，live URL 默认追加 `.ts` 后缀，让 Skybox 选择 MPEG-TS pipeline。路由端会剥离 `.ts`、`.m2ts`、`.mpegts` 后缀再查找真实源文件。

默认 live DIDL 对非 DeoVR 使用更纯 live 的 `DLNA.ORG_OP=00`，并省略 duration/bitrate 等 file-like 属性。DeoVR 使用旧形态：不追加 `.ts`，保留 `DLNA.ORG_OP=10`、duration、bitrate。

## 对 Vision Pro 4XVR 的判断

如果 Apple 端 4XVR 的 UA 不包含当前命中词，例如 `AVProMobileVideo`、`ExoPlayerLib`、`Dalvik/`、`HereSphere` 等，它会落入默认 `vlc` profile。

这与 Android/Quest 侧 4XVR 的 `4xvr`/`avpro` 行为差异较大。若 Apple 4XVR 启动时发非起点 Range、重复打开同一 live URL、或依赖重连复用，默认 `vlc` 路线确实可能失败。

日志已确认本次 Vision Pro 4XVR 请求命中了默认 `vlc` profile：

- CDS 目录请求 UA：`Darwin/25.5.0, UPnP/1.0, Portable SDK for UPnP devices/1.14.21`
- UA：`Vision4XVR/2 CFNetwork/3860.600.12 Darwin/25.5.0`
- route profile：`vlc`
- 请求路径：`/passthrough_live/...`
- 服务端已进入 PyNv realtime MPEG-TS 生产链路并返回 `200`

因此本次优先修复为：将 `4xvr` UA token 映射到 `4xvr` profile，而不是调整全局默认 profile。

## 建议

不建议立即把全局默认路线从 `vlc` 改成 `4xvr`/`avpro`。原因是默认路线会影响所有未知播放器，而 `4xvr`/`avpro` 的 managed session、同设备接管和 Range 容忍度更宽，可能改变其它未知客户端的资源占用和抢占行为。

更稳妥的下一步是先拿 Vision Pro 4XVR 的 request history，确认 UA、Range、请求序列和状态码。若证实只是 UA 漏识别，应补 Apple 4XVR 的 UA 规则到 `4xvr` 或 `avpro` 路线，而不是全局调整 unknown default。

如果后续需要兜底，可考虑新增一个明确的兼容模式开关或设备级 profile 绑定，让用户把某台设备强制按 `4xvr`/`avpro` 处理，同时保留 request intent 的 probe/side/tail 保护。
