# SI Progressive Virtual MP4 原声音画不同步研究修正版

日期: 2026-06-18

## 0. 修正声明

上一版 summary 对问题性质判断有误: 它把“明显音画不同步”过早归因到 `DEFAULT_SI_DELAY_SECONDS=1.0`。用户已澄清,实际测试中错位的是**原本视频声音相对画面**，不是“同声传译 SI 语音相对原声”的错位。

因此必须修正结论:

- `si_delay_seconds` 只影响 SI WAV/翻译语音相对原片原声的偏移。
- 它不能解释“原片自带声音相对画面错位”。
- `DEFAULT_SI_DELAY_SECONDS=1.0 -> 0.0` 仍然是合理的默认值修正,但它不是本次“原声 vs 画面”错位的根因闭环。
- 后续排查必须聚焦到: sidecar 中原声是否被整体平移、virtual MP4 组合后 audio track 是否相对 video track 被解释错、播放器/客户端是否缓存旧 moov 或用非标准 byte seek。

本文件是给外部专家继续研究的修正版,应替代旧判断。

## 1. 当前实现背景

`/media_si` 当前实现为 progressive virtual MP4 remux:

- 视频不重编码,直接复用源 MP4 video samples。
- 源音频 `[0:a:0]` 与 SI WAV `[1:a:0]` 先由 ffmpeg filter 混成一个 AAC sidecar MP4。
- virtual MP4 输出逻辑文件:
  - 内存中提供 `ftyp + rewritten moov + mdat header`。
  - `mdat` payload 由源视频 sample regions 和 AAC sidecar audio sample regions 组成。
- `moov` 处理:
  - 源 video trak 原样保留 timing/description boxes。
  - sidecar audio trak 导入。
  - 仅重写 `stsc` 为 one-sample-per-chunk,并把 `stco/co64` 改为新虚拟文件中的绝对 offset。

相关代码:

- `pipeline/si_virtual_mp4.py`
- `http_app/routes_media.py`
- `dlna/content_directory.py`
- `utils/si_filter.py`

## 2. `si_delay_seconds` 的真实语义

`si_delay_seconds` 是 UI 原本允许配置的“翻译语音 SI 相对原视频声音延迟多少秒”。

代码位置: `utils/si_filter.py::build_si_mix_filter()`

输入关系:

- `[0:a:0]`: 源视频自带音频。
- `[1:a:0]`: `.si.wav` 同声传译/翻译音频。

`si_delay_seconds` 只作用在 `[1:a:0]`:

```text
[1:a:0] ... adelay={si_delay_ms} ...
```

含义:

- 正数: SI 翻译声更晚。
- `0.0`: SI 翻译声不额外延迟。
- 不改变源音频 `[0:a:0]` 的时间位置。

所以:

- 如果用户听到的是“SI 翻译声比原声晚 1 秒”,旧默认 `1.0` 可以解释。
- 如果用户听到的是“原片自带声音比画面早/晚”,旧默认 `1.0` 不能解释。

## 3. 已知 sidecar 和媒体数据

测试片:

- `videos\SI_TEST_8K.mp4`
- `videos\SI_TEST_8K.si.wav`

旧 sidecar:

- `runtime_cache\si_virtual_mp4\37be0fbab73c053fa0708d1ffc8a04afaf05931006c4894421482301c859ec62.audio.mp4`
- 对应参数: `si_delay_seconds=1.0`
- 文件大小: `72,596,626` bytes

新 sidecar:

- `runtime_cache\si_virtual_mp4\b6a4e5fc94f1f98a15f57e2abeddbd39a12f1711a82dbb42bd04ac9debd7b0ab.audio.mp4`
- 对应参数: `si_delay_seconds=0.0`
- 文件大小: `72,568,392` bytes

两份 sidecar 的 `ffprobe` audio stream 级元数据一致:

```text
codec_name=aac
time_base=1/48000
start_time=0.000000
duration=3007.104000
nb_frames=140959
```

## 4. 外部专家已做的结构性取证

外部专家反馈已经检查了实际 MP4 box 结构,要点如下:

| 检查项 | 源视频 | 0.0s sidecar | 判定 |
|---|---:|---:|---|
| movie timescale (`mvhd`) | 1000 | 1000 | 一致 |
| 视频 edit list | `media_time=528 @16000 ~=33ms` | - | CTS 补偿 |
| 音频 edit list | - | `media_time=1024 @48000 ~=21.3ms` | AAC priming 修剪 |
| video `stsz/stts/co64` 计数 | 180240 | - | 一致 |
| audio `stsz/stts/co64` 计数 | - | 140959 | 一致 |

该反馈支持以下判断:

- movie timescale 不匹配不是当前证据下的主因。
- sidecar audio edit list/AAC priming 信息存在且被保留。
- sample table 条目数与 PyAV demux 出的 offset 数一致,没有明显样本计数错位。
- 即使播放器忽略 edit list,视频 33ms 与音频 21.3ms 的差也只有约 12ms,不应造成“明显”错位。

这些结论有价值,但它们不等同于“`si_delay=1.0` 是根因”。用户已澄清错的是原声相对画面。

## 5. 新增取证: sidecar 中原声没有被 mix/filter 平移

为了验证“原声是否在 sidecar 生成阶段被整体平移”,做了 PCM cross-correlation:

方法:

- 用 ffmpeg 从源 MP4 解码 `[0:a:0]` 为 48kHz mono f32 PCM。
- 用 ffmpeg 从 sidecar 解码 audio track 为同规格 PCM。
- 对 20s 窗口做 FFT cross-correlation。
- 正 lag 表示 sidecar 音频相对源音频更晚。
- 分别测试开头、中段和后段。

结果:

```text
delay1_37be
  start=  0.000s lag_ms=    4.958 score=0.1242
  start=600.000s lag_ms=    4.479 score=0.2690
  start=1500.000s lag_ms=   4.479 score=0.3473

delay0_b6a4
  start=  0.000s lag_ms=    4.979 score=0.1085
  start=600.000s lag_ms=    4.479 score=0.1795
  start=1500.000s lag_ms=   4.479 score=0.2944
```

解释:

- 旧 `delay=1.0` sidecar 和新 `delay=0.0` sidecar 中,原声成分相对源原声都只晚约 `4.5-5ms`。
- 该偏移远低于可感知“明显错位”。
- 600s 和 1500s 仍是约 `4.5ms`,没有随时间漂移。
- 因此 ffmpeg mix/filter 阶段没有把源原声整体错开。
- `si_delay_seconds` 的变化也没有影响原声对齐,符合代码语义: 它只影响 SI WAV。

这个结果是当前最重要的新增证据。

## 6. 当前可以排除或降低优先级的方向

基于外部 box 取证和 PCM 相关性测试:

1. **SI delay 导致原声错位**
   - 排除。
   - 它只影响 SI 翻译声,不影响源原声。

2. **ffmpeg mix/filter 把原声整体平移**
   - 基本排除。
   - 源原声 vs sidecar 原声成分只差约 `4.5-5ms`,无明显漂移。

3. **sidecar 内部原声随时间漂移**
   - 基本排除。
   - 600s/1500s 相关性 lag 仍稳定。

4. **明显的 sample table 计数错位**
   - 外部专家已用 `stsz/stts/co64` 计数排除。

5. **movie timescale 不匹配**
   - 外部专家已排除: source 和 sidecar `mvhd` timescale 都是 1000。

## 7. 仍然成立的疑点

### 7.1 实机播放的究竟是哪条音轨/哪份 moov

需要在下一次实机测试后查看 `server.log`:

```powershell
rg -n "SI mixed AAC sidecar|built SI progressive virtual MP4|media_si" debug_output\server.log
```

关键日志应包含:

```text
digest=... delay=...
```

但注意: 对“原声 vs 画面”错位而言,`delay=1.0/0.0` 本身不是解释。日志的意义是确认服务端参数、sidecar 和播放器请求路径。

### 7.2 virtual MP4 组合后的 audio track 是否被播放器解释错

sidecar 本身原声对齐源原声,但导入 virtual MP4 后,播放器看到的是一个新组合的 movie:

- source video trak
- imported sidecar audio trak
- rewritten `stsc/co64`
- copied timing boxes / edit lists

如果原声相对画面仍明显错位,问题更可能在:

- 实际生成的 virtual `moov` 与预期结构不一致。
- 播放器对这种“source video trak + sidecar audio trak”的组合解释有兼容性问题。
- 播放器在 progressive MP4 下没有完全按 sample table seek,而是做了 byte-position heuristic。
- 客户端缓存了旧 moov 或旧 DLNA item。

### 7.3 原片自身在目标播放器/目标路径上是否同步

必须做对照:

- 同一个播放器直接播放原始 `SI_TEST_8K.mp4` 是否音画同步。
- 通过普通 `/media` 或原始 DLNA item 播放是否同步。
- 只有 `/media_si` 不同步,还是原始文件在该播放器上也不同步。

如果原片自身在目标播放器上就有原声错位,则 `/media_si` 不是根因。

## 8. 下一步建议

### 8.1 修正实机观察问题

下一次测试请明确记录:

- 错的是“原声 vs 画面”,还是“SI 翻译声 vs 原声”。
- 原声是早还是晚。
- 大概偏移量: 100ms、500ms、1s、更多。
- 开头播放和 seek 到中段后偏移是否一致。
- 同一个播放器播放原始 `SI_TEST_8K.mp4` 是否同步。

### 8.2 对 HTTP/virtual 输出做直接 ffmpeg/ffprobe 验证

如果 server 正常运行,建议对 `/media_si` URL 直接做 probe/解码,而不是只检查 sidecar:

```powershell
ffprobe -v error -show_streams -show_format "http://127.0.0.1:8200/media_si/SI_TEST_8K.mp4"
```

并从 HTTP virtual MP4 解码 audio,再与源音频做 cross-correlation:

```powershell
ffmpeg -v error -ss 600 -i "http://127.0.0.1:8200/media_si/SI_TEST_8K.mp4" -t 20 -map 0:a:0 -ac 1 -ar 48000 -f f32le virtual_audio_600.f32
```

如果 virtual URL 解码出的音频仍和源音频只差约 5ms,则服务端 audio payload/timing 基本正常,问题更偏向播放器兼容性或视频时间轴解释。

### 8.3 dump 实际 virtual moov

需要确认实际服务出去的 rewritten `moov`:

- audio trak 是否来自 `b6a4e5...` 或实际 digest 对应 sidecar。
- `mvhd/tkhd/mdhd/elst/stts/ctts/stsz/co64` 是否与专家检查一致。
- `co64` 是否单调、计数是否等于 sample count。

如果可能,在代码里增加临时 debug dump:

```text
debug_output/si_virtual_mp4_last_moov.mp4box.bin
```

或输出 top-level init segment:

```text
ftyp + moov + mdat header
```

供 Bento4/MP4Box/pymp4 独立检查。

### 8.4 不建议现在改 timing filter

不要因为旧 summary 提到的方向就贸然加入:

- `asetpts=PTS-STARTPTS`
- `aresample=async=1:first_pts=0`
- `-avoid_negative_ts make_zero`

理由:

- PCM 相关性已显示 sidecar 中原声没有明显偏移。
- 贸然重写 PTS/priming 可能破坏当前 sidecar 的 AAC priming/edit list 处理。

## 9. 当前结论

当前更准确的结论是:

1. `DEFAULT_SI_DELAY_SECONDS=1.0` 是一个会影响 SI 翻译声相对原声的默认值问题,改成 `0.0` 合理。
2. 但用户澄清后,本次实测问题是“原声相对画面错位”,因此不能把它归因为 SI delay。
3. 两份 sidecar 中的原声成分都与源原声在约 `5ms` 内对齐,没有可感知固定偏移或漂移。
4. 如果 `/media_si` 仍出现原声 vs 画面明显错位,下一步应验证 actual HTTP virtual MP4 输出和目标播放器行为,而不是再改 SI delay 或 audio filter timing。

一句话:

**SI delay 修正只解决 SI 翻译声偏移；原声音画错位仍是开放问题。当前证据更支持 sidecar/filter 正常,需要继续检查 virtual MP4 实际输出和播放器解释。**

## 10. 2026-06-18 追加实验实现: audio edit-list A/B

根据专家最新反馈,当前最值得实机验证的假设是:

> 播放器对 source video 的 edit-list/负 DTS/ctts 和 sidecar audio 的 AAC priming edit-list 处理不对称,导致 remux 后原声相对画面固定偏移。

为了把这个假设变成可测 A/B,已加入一个很窄的实验开关:

```text
PT_SI_AUDIO_EDIT_MODE=remove|preserve
```

当前默认:

```text
PT_SI_AUDIO_EDIT_MODE=remove
```

模式含义:

- `preserve`: 保持旧行为,导入 sidecar audio trak 时保留 `edts/elst`。
- `remove`: 只在 virtual moov 中移除导入 audio trak 的 `edts/elst`。不改 sidecar 文件,不改 video trak,不改 sample payload,不改 SI delay。

实现点:

- `config.py`
  - 新增 `SI_AUDIO_EDIT_MODE`,默认 `remove`。
- `pipeline/si_virtual_mp4.py`
  - `_rewrite_trak(..., drop_edts=True)` 支持跳过 `edts`。
  - `_build_moov(..., audio_edit_mode=...)` 只对 imported audio trak 应用该策略。
  - layout cache digest / ETag 已纳入 `audio_edit_mode`,避免同进程切换 A/B 时复用旧 moov。
  - `ProgressiveSIVirtualMp4.audio_edit_mode` 记录当前模式。
- `http_app/routes_media.py`
  - 响应头新增:
    - `X-SI-Audio-Edit: remove|preserve`

真实 `SI_TEST_8K.mp4` 验证:

```text
audio=b6a4e5fc94f1f98a15f57e2abeddbd39a12f1711a82dbb42bd04ac9debd7b0ab.audio.mp4
audio_edit=remove
size=7449678144
moov=6173403
samples=180240+140959
regions=321200
etag=be9eb7fa1da278feee64a2ee1ec2aa00
```

实际 init segment 检查:

```text
audio_edit remove
video_edts True
audio_edts False
moov_size 6173403
```

这确认实验只移除了 imported audio trak 的 `edts`,源 video trak 的 `edts` 仍保留。

测试:

```text
tests\test_si_mix.py tests\test_si_virtual_mp4.py tests\test_config_defaults.py tests\test_routes_media_cache.py
74 passed

tests\test_content_directory_modes.py -k "not versioned_live_id_resolves"
34 passed, 1 deselected, 4 subtests passed

git diff --check
passed, only CRLF warnings
```

下一次实机测试判断点:

- 默认已经是 `remove`,重启 server 后测试 `/media_si`。
- 看响应头或日志是否出现:
  - `X-SI-Audio-Edit: remove`
  - `built SI progressive virtual MP4 ... audio_edit=remove ...`
- 如果 `remove` 下原声对画面同步恢复,则说明播放器对 audio priming edit-list 与 video edit/ctts 的处理存在不对称。
- 如需回退旧行为做 A/B,设置:

```powershell
$env:PT_SI_AUDIO_EDIT_MODE="preserve"
```

然后重启 server、重新 Browse `[SI]` item 再测。
