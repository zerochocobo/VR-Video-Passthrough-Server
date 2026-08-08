# VR Video Passthrough Server

English | [中文](README.zh-CN.md) | [日本語](README.ja-JP.md)

Website: [https://wapok.com](https://wapok.com)

VR Video Passthrough Server aims to make every VR video passthrough-capable, enabling mixed reality (MR).

![VR Video Passthrough Server overview](assets/intro_en_s.png)

It is a Windows-first VR DLNA local media server with a multilingual desktop UI and dedicated offline tools. It exposes local video libraries over DLNA/UPnP and provides realtime Green-screen and Alpha passthrough, 2D perspective conversion, 2D-to-3D / VR generation, NVIDIA RTX Video Super Resolution, hard subtitles, light matching, and dubbing / simultaneous-interpretation playback from same-stem `.si.wav` sidecars. It is optimized for VR180 half-equirectangular sources while also supporting eligible flat 2D videos.

Current desktop release: **v1.2.0**.

## Project Origin

I originally wanted to build a VR video passthrough tool.   
Someone said: "You’re just reinventing the wheel."  
I replied: "That old wheel from years ago is too outdated. It's time for a new one."  
Seven days later, the new wheel was born.    
This is the miracle of the AI era.    

## Features

- DLNA discovery and ContentDirectory browsing
- Realtime passthrough streaming with GPU matting and HEVC output
- Realtime subtitle embedding in the passthrough stream
- Green-screen and alpha passthrough modes
- Offline passthrough video generation
- Realtime and offline 2D-to-3D / VR generation for flat 2D videos using DA3 depth and GPU stereo rendering
- Optional 2D-to-3D depth stabilization, including built-in temporal stabilization and NVDS ONNX stabilization for offline 16:9 jobs
- Realtime and offline NVIDIA RTX Video Super Resolution for eligible 2D and VR sources, including adaptive `8K VR / 4K 2D` output
- Split-eye GPU processing for 2:1 SBS VR, with 8192x4096 HEVC output available for offline 4K-VR-to-8K-VR conversion
- Low / Medium / High / Ultra RTX VSR quality controls and Off / Natural / Vivid SDR HDR-look controls
- Realtime and offline Face Beauty (under development and testing; enable it from Settings)
- Realtime and offline mosaic removal (under development and testing; enable it from Settings)
- Dubbing / simultaneous interpretation playback with same-stem `.si.wav` sidecars, `[SI]` DLNA entries, start-time selection, channel mixing, and Light / Normal / Strong ducking
- DLNA Live time-index folders for choosing a playback start time, with 10-minute groups, minute folders, and 5-second playback points
- Multi-root local video library support
- PySide6 desktop UI with a navigation rail, feature-card dashboard, Offline Tools, Subtitle Style, Logs, and Settings pages
- Editable DLNA server name and HTTP port, with Chinese, English, and Japanese translations
- Subtitle preview and style configuration
- Realtime light matching with color-temperature, tint, exposure, contrast, gamma, saturation, and presets
- Aggressive VRAM-aware pipeline tuning aimed at keeping realtime output smooth, including 8K-class source playback targets where the hardware can sustain them


| ![MainWindow](assets/soft_mainwindow_en.png) |


## Passthrough Output Examples

| Alpha Passthrough | Green-screen Passthrough |
| --- | --- |
| ![Alpha Passthrough example](assets/sample_alpha.jpg) | ![Green-screen Passthrough example](assets/sample_green.jpg) |
| ![Screenshot](assets/passthrough_screenshot.jpg) |


## Requirements

- Windows 10 / 11
- Python 3.12
- NVIDIA GPU for GPU processing. Rough recommendation: RTX 20 series or newer; RTX VSR also requires a GPU/driver supported by NVIDIA RTX Video. Check your exact model on NVIDIA's official list: <https://developer.nvidia.com/cuda/gpus>. Recommended VRAM: 6 GB or more for the realtime server, RVM offline generation, and normal DA3 2D-to-3D; about 15 GB or more for MatAnyone2 / SAM3 offline workflows. HD/Large DA3, NVDS, and 8K SuperRes are high-load offline workflows.
- FFmpeg / FFprobe

Performance note: 4K SBS VR to 8K VR at Ultra quality is extremely GPU-intensive. An RTX 5060 Ti measured about 23-24 processing FPS. Lower-end GPUs should use a lower Target quality or reduce Output FPS in Global Settings; NVENC P1 controls encoding speed and is separate from NGX Target quality.

## Quick Start

```bash
uv run python main.py
```

Launch the desktop UI:

```bash
uv run python -m ui.app
```

## Ports and Firewall

When the realtime server starts, it uses the following network ports:

| Purpose | Protocol / Port | Description |
| --- | --- | --- |
| HTTP media service | TCP 8200 | Serves the DLNA device description, media catalog, thumbnails, source videos, and realtime passthrough streams. You can change it with `PT_HTTP_PORT`. |
| SSDP / UPnP discovery | UDP 1900 | Lets VR players discover the local DLNA server on your LAN. |
| Startup status | TCP 8299 (localhost only) | Used by the desktop UI to read GPU warmup / startup status. It is intended for local UI use only by default. You can change it with `PT_STARTUP_STATUS_PORT`. |

On first startup, the application tries to add these Windows Firewall inbound rules automatically:

- `PTServer HTTP Private`: allows inbound TCP 8200 on private networks.
- `PTServer SSDP Private`: allows inbound UDP 1900 on private networks.

If Windows shows a UAC / firewall confirmation prompt, allow it. For normal home LAN usage, allow private networks only; do not expose the server on public networks.

If you accidentally denied the prompt, or if your VR player cannot discover the server, add the rules manually. Open PowerShell or Command Prompt as Administrator and run:

```powershell
netsh advfirewall firewall add rule name="PTServer HTTP Private" dir=in action=allow protocol=TCP localport=8200 profile=private edge=no enable=yes
netsh advfirewall firewall add rule name="PTServer SSDP Private" dir=in action=allow protocol=UDP localport=1900 profile=private edge=no enable=yes
```

If you changed `PT_HTTP_PORT`, replace `8200` in the first command with your actual port. UDP 1900 is the standard UPnP/SSDP discovery port and normally should not be changed.

You can also configure this from the Windows UI:

1. Open "Windows Security" -> "Firewall & network protection" -> "Advanced settings".
2. Go to "Inbound Rules" and create a new rule.
3. Choose "Port" as the rule type.
4. Add `TCP 8200` and `UDP 1900` as separate rules.
5. Choose "Allow the connection".
6. Select the "Private" profile only when possible.
7. Name the rules `PTServer HTTP Private` and `PTServer SSDP Private`.

## Supported VR Players

Tested on Meta Quest 3.

| Player | Alpha passthrough | Gray green screen | ChromaKey green screen | Website | Notes |
| --- | --- | --- | --- | --- | --- |
| Skybox VR Player 2.0.2 Preview | Supported | - | Supported | [Official site](https://skybox.xyz) | [Installation notes](https://forum.skybox.xyz/d/2920-skybox-quest-v202-preview-performance-improvements) |
| Moon Player | - | Supported | Supported | [Official site](https://moonvrplayer.com) | - |
| 4XVR Video Player | Supported | - | Supported | [Official site](https://www.4xvr.net/) | - |
| DeoVR player | Supported | - | Supported | [Official site](https://deovr.com/) | - |
| HereSphere VR Video Player | Supported | - | Supported | [Official site](https://heresphere.com/) | - |

## Configuration Notes

- `PT_VIDEO_DIR` supports multiple roots separated by `|`
- Media roots must be local paths. Cloud drives mounted as local drive letters/directories are supported; UNC network-share roots are rejected.
- `PT_PASSTHROUGH_OUTPUT_MODE` supports `none`, `green`, `alpha`, `two_dvr`, `superres`, comma-separated combinations such as `green,alpha,two_dvr,superres`, and legacy `all` for green + alpha
- `Alpha Passthrough` is the DLNA virtual title used in alpha mode
- Realtime 2D-to-3D uses `PT_TWO_DVR_MODEL`, `PT_TWO_DVR_STRENGTH`, and related `PT_TWO_DVR_*` settings; offline 2D-to-3D / VR exposes model, quality-speed, temporal stability, and skip-existing controls in the desktop UI.
- Realtime SuperRes uses `PT_RTX_VSR_TARGET_HEIGHT`, `PT_RTX_VSR_QUALITY`, and `PT_RTX_VSR_HDR_LOOK`. The adaptive 4096 target means 8192x4096 for recognized 2:1 SBS VR and 3840x2160 for ordinary 2D.
- Same-stem `.si.wav` files enable `[SI]` DLNA entries. Current DLNA playback uses realtime MPEG-TS through `/si_live` with start offsets; the older progressive `/media_si` implementation remains a fallback route.
- DLNA Live directories use `[GREEN]`, `[ALPHA]`, `[2D>3D]`, `[SuperRes]`, and `[SI]` markers where applicable and include a localized `[Select Time Index]` folder for start-time selection.
- The desktop Settings page can change the advertised DLNA server name and HTTP port. Restart the server after saving network identity changes.
- Packaged Windows builds include the validated CUDA 12.6 RTX VSR bridge, NGX runtime, local CUDA runtime, license, and version metadata; end users do not compile the bridge.
- TensorRT acceleration is controlled from the desktop UI Performance panel. Build the cache first in `TensorRT -> Configure`; the first build can take several minutes. If the cache is missing or stale after a driver/CUDA/TensorRT/model change, the server falls back to CUDA automatically.
- UI settings are stored separately from backend runtime configuration

## Project Layout

```text
main.py        Server entry point
config.py      Runtime configuration
dlna/          UPnP / DLNA discovery and catalog
http_app/      FastAPI routes
pipeline/      Decode, matting, encode, thumbnail, subtitle pipeline
offline/       Production offline conversion entry points
ui/            PySide6 desktop UI, pages, i18n, and process control
tools/         Developer probes and diagnostics
models/        Local model files and manifests
resources/     Packaged UI/runtime assets
prompt/        Handover notes and investigation reports
```

## Referenced Open Source Models

VR Video Passthrough Server does not train matting models itself. It consumes upstream models and model files from the projects below.

| Model | Role | Upstream |
| --- | --- | --- |
| Robust Video Matting (RVM) | Primary realtime matting path, including `rvm_mobilenetv3_fp16.onnx`, `rvm_mobilenetv3_fp32.onnx`, and `rvm_resnet50_fp32.onnx` | [GitHub](https://github.com/PeterL1n/RobustVideoMatting) |
| MatAnyone2 | Slower, higher-quality matting path for offline conversion and experimental workflows | [GitHub](https://github.com/pq-yang/MatAnyone2) |
| Segment Anything Model 3 (SAM 3) | Optional helper used by high-quality offline alpha prepass workflows | [GitHub](https://github.com/facebookresearch/sam3) |
| Depth Anything 3 (DA3) | Monocular depth model used by realtime and offline 2D-to-3D / VR generation | [GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| NVDS | Optional offline 2D-to-3D depth / near-map temporal stabilizer for 16:9 sources | [GitHub](https://github.com/RaymondWang987/NVDS) |

## Referenced Dependencies

- [PySide6](https://www.qt.io/qt-for-python)
- [FastAPI](https://github.com/fastapi/fastapi)
- [Uvicorn](https://github.com/encode/uvicorn)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [CuPy](https://github.com/cupy/cupy)
- [PyNvVideoCodec](https://github.com/NVIDIA/VideoProcessingFramework)
- [PyAV](https://github.com/PyAV-Org/PyAV)
- [NVIDIA RTX Video SDK](https://developer.nvidia.com/rtx-video-sdk)

## Notes

- The codebase is currently tuned for a local Windows machine rather than a hosted deployment.
- Generated modes are exposed as separate DLNA entries and are processed on demand; original media remains available unchanged.
- The current pipeline is optimized for VR180 half-equirectangular sources, with additional supported workflows for flat 2D video.
- See [README.zh-CN.md](README.zh-CN.md) for the Chinese version and [README.ja-JP.md](README.ja-JP.md) for the Japanese version.

## License

License: `AGPL-3.0-or-later`.

See the repository license for project terms. Upstream model repositories keep their own licenses and usage terms.
