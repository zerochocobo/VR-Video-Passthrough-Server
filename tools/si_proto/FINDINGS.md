# SI 虚拟 remux MP4 — PyAV 原型结论（2026-06-17）

独立预研产物，配合 `summary/summary_20260617_REALTIME_TS_DEV_PLAN_CN.md` 使用。
脚本：`tools/si_proto/pyav_prototype.py`（一次性研究原型，非生产代码）。
测试对象：`videos/SI_TEST_8K.mp4`（HEVC 8K, 50min, 7.47GB）+ `SI_TEST_8K.si.wav`。

## 已验证（4/4 通过）

| # | 验证项 | 结果 |
|---|---|---|
| 1 | PyAV 读源视频样本表（pos/size/pts/keyframe + hvcC extradata） | ✅ codec=hevc 8192x4096，hvcC=2577B，逐样本 pos/size/key 可得 |
| 2 | **`源文件[pos:pos+size]` == demux 出的 packet 字节** | ✅ True —— **视频字节可直接按偏移从源文件切，无需落整片(避开 8GB)** |
| 3 | PyAV 把 视频copy + 混音 mux 成 fMP4 到 file-like，并拦截顶层 box | ✅ init 段(ftyp+moov)仅 **3838B**；moof/mdat 分片；自动产出 **mfra**(随机访问索引) |
| 4 | 同输入两次生成字节完全一致 | ✅ sha256 相同、大小相同 —— **字节稳定，可支撑稳定 Content-Length / ETag** |

round-trip：PyAV 产物经 ffprobe 校验为合法 `hevc + aac`。

## 关键数字（20s 8K 切片）
- init segment = **3838 B**（fMP4 的 moov 极小，无逐样本 stbl，印证专家判断）。
- 分片 = 一关键帧一片（`frag_keyframe`），mdat 约 **13.5 / 16 / 15.8 MB** 一片（8K 关键帧稀疏 → 片很大）。
- libav 自动在尾部写 `mfra`（time→moof 偏移映射，利于 seek）。

## 对 Phase 1 的意义
- **lean 路线成立**：不落整片。一次性扫描得到每个 fragment 的确定大小 + 组成它的源视频样本范围；服务 Range 时**按需重生成该 fragment**（copy+固定音频→确定性→字节稳定，已由 #4 证明），或直接按 #2 从源文件切视频字节。8GB 中间文件可避免。
- **PyAV 可担此任**：muxer + box 拦截 + 进程内样本元数据，一套 API 搞定，免 subprocess/管道。

## PyInstaller 打包探针（已补测）

脚本：`tools/si_proto/pyinstaller_av_probe.py`。

已执行隔离 onedir 构建并运行冻结 exe：

```powershell
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean `
  --name si_pyav_probe --onedir `
  --distpath debug_output\si_proto\pyinstaller_dist `
  --workpath debug_output\si_proto\pyinstaller_build `
  --runtime-hook packaging\runtime_hook_cuda_dlls.py `
  tools\si_proto\pyinstaller_av_probe.py

debug_output\si_proto\pyinstaller_dist\si_pyav_probe\si_pyav_probe.exe `
  debug_output\si_proto\B_fmp4.mp4
```

结果：✅ 冻结 exe 可加载 `av==17.1.0`，可打开 `B_fmp4.mp4` 并识别 `hevc 8192x4096 + aac`。PyInstaller hook 自动收集 `av.libs`，并与现有 `packaging/runtime_hook_cuda_dlls.py` 共存。该探针不是完整主程序打包，但已覆盖 PyAV/libav DLL 的最小运行风险。

## 原型的简化 / 生产待办（开发注意）
1. 本原型把"全部视频包 + 全部音频包"顺序写入；**生产需在每个 fragment 内按 dts 正确交错音视频**（否则播放器缓冲/同步受影响）。
2. 本原型把整段 fMP4 **物化在内存**（20s=61MB）；生产应只算布局、按需重生成 fragment，不物化整片。
3. fragment 粒度：`frag_keyframe` 使一片≈一 GOP（8K 下可达十几 MB）。如需更细 Range 命中，评估 `frag_duration` 或自定义分片；注意片越细 moof 开销越大。
4. **Content-Length 必须由一次性布局扫描得到精确值**（fMP4 全局 sidx / 各 fragment 大小），不能估算。
5. A/V 时长对齐 + SI 延迟由 `build_si_mix_filter` 的 `adelay` 承担（音频已预混）。
6. >4GB：fMP4 用 `default-base-is-moof` 相对偏移，天然规避 co64 绝对偏移问题。

## Phase 0 SKYBOX 抓包结论（2026-06-17 补）

抓包：`debug_output/si_proto/skybox.pcapng`；服务日志：`debug_output/server_skybox.log`。

| 文件 | SKYBOX 体验 | HTTP Range 模式 | 服务端实际发送 |
|---|---|---|---|
| `A_progressive.mp4` | 缓存条快速读满，播放流畅 | 1 个 `bytes=0-` 顺序请求 | 约 92.8MB，1.75s 拉完整文件 |
| `B_fmp4.mp4` | 可显示进度条、可拖动，但缓存条追着进度条，体感卡顿 | 856 个 open-ended Range，先扫 `moof/mfra`，随后大量重叠 `mdat` 请求 | 约 343MB（文件本身约 92.8MB） |
| `C_fmp4_dash.mp4` | 可显示进度条、可拖动，但体感同样卡顿 | 807 个 open-ended Range，先扫尾部/`sidx`，随后大量重叠 `mdat` 请求 | 约 333MB（文件本身约 92.8MB） |

结论：SKYBOX **认 fMP4，也能 seek**，但 fMP4 触发了高频重叠 Range 风暴，网络读取量约为文件大小的 3.6 倍，解释了 B/C 的卡顿。对 SKYBOX 的 Phase 1 首选应改为 **progressive virtual remux**，而不是 fMP4；fMP4 仅保留为其他播放器实测更优时的备选。

限制：A 文件只有 30s/约 92.8MB，LAN 下 1.75s 已整片缓存，所以 A 的拖动没有充分证明"未缓存大文件远端 seek"行为；但它已经证明 SKYBOX 对 progressive MP4 的文件式播放/缓存路径明显最稳。后续接 `/media_si` 时应优先按 progressive 布局实现，并用真实长片再验证拖动。

## 仍未解决（其他播放器）
SKYBOX 已给出方向：progressive 优先。nPlayer 与 4XVR 后续也确认不推翻该判断。HereSphere/DeoVR 是否同样偏好 progressive，仍需用 Phase 0 三个静态文件继续抓包确认。

## Phase 0 nPlayer 抓包结论（2026-06-17 补）

抓包：`debug_output/si_proto/nPlayer.pcapng`；服务日志：`debug_output/server_nplayer.log`。

| 文件 | nPlayer 行为 | HTTP Range 模式 | pcap 服务端 TCP 实发 |
|---|---|---|---|
| `A_progressive.mp4` | 能播、能拖；多次播放/拖动混在同一日志里 | 4 个无 Range `200` 全文件请求 + 8 个 open-ended Range；常见尾部 `bytes=92667904-` 和 50%/66% 位置 | 约 185MB（包含多次重复播放/拖动，单次完整 full-file 约 92.8MB/1.8s） |
| `B_fmp4.mp4` | 能播、能拖 | 2 个无 Range `200` + 36 个 open-ended Range；多数起点落在 `mdat` 内部，不贴 `moof/mfra` 边界 | 约 94MB |
| `C_fmp4_dash.mp4` | 能播、能拖 | 2 个无 Range `200` + 34 个 open-ended Range；多数起点落在 `mdat` 内部，不贴 `sidx/moof` 边界 | 约 99MB |

结论：nPlayer 对 progressive 和 fMP4 都能 seek；fMP4 **没有**出现 SKYBOX 那种 800+ Range / 3.6x 重叠读取风暴。但 nPlayer 也没有明显利用 fMP4 的 `sidx/moof` 边界，`C_fmp4_dash` 相比 `B_fmp4` 没看到收益。nPlayer 不推翻 SKYBOX 的路线判断：**progressive virtual remux 仍是默认首选**；fMP4 可保留为播放器专项备选。

## Phase 0 4XVR 抓包结论（2026-06-17 补）

抓包：`debug_output/si_proto/4xvr.pcapng`；服务日志：`debug_output/server_4xvr.log`。4XVR UA 为 Quest 3 上的 `Dalvik/2.1.0`。本轮日志开头曾点到 `/passthrough_live/...A_progressive...` alpha 条目，该请求不纳入静态 `/media` 判定。

| 文件 | 4XVR 行为 | HTTP Range 模式 | pcap 服务端 TCP 实发 |
|---|---|---|---|
| `A_progressive.mp4` | 能播、能拖动 | 静态 `/media` 中 2 个无 Range `200` + 1 个闭区间 Range `bytes=45088768-92795440` | 约 116MB，约 1.25x 文件大小 |
| `B_fmp4.mp4` | 能播、能拖动，但请求压力大 | 1 个无 Range `200` + 45 个闭区间 Range，均为 `bytes=N-file_end` | 约 589MB，约 6.35x 文件大小 |
| `C_fmp4_dash.mp4` | 能播、能拖动，但请求压力仍大 | 1 个无 Range `200` + 32 个闭区间 Range，均为 `bytes=N-file_end` | 约 439MB，约 4.73x 文件大小 |

和 SKYBOX/nPlayer 不同，4XVR 不发 open-ended Range，而是发**有限闭区间**到文件尾，Range 起点大量按 1MiB 对齐。用 `BoxSplittingSink` 对 B/C 映射后确认：B 的 45 个 Range 起点、C 的 32 个 Range 起点**全部落在 `mdat` 内部**，没有命中 `moof`、`mfra` 或 `sidx` 边界。`C_fmp4_dash` 的 `sidx` 没带来收益。

结论：4XVR 同样认可三种 MP4 文件并可 seek，但 fMP4 的实际网络读量明显高于 progressive。它不推翻 Phase 1 的默认路线，反而继续支持 **progressive virtual remux MP4 优先**；fMP4 仅保留为未来播放器专项 fallback。

## 字节稳定不变量已被测试钉死（2026-06-17 补）
原型只手验过"两次生成字节一致"。现已用两层测试锁定 `iter_virtual_range` 的字节稳定+一致性：
- **CI 永久单测**（无需 PyAV）：`tests/test_si_virtual_mp4.py::...random_ranges_match_ground_truth_and_are_stable` —— 混合 memory/file 区域布局，300 个随机 Range × 多种 chunk 大小，断言"切片==整文件对应区间"且"重复切片字节恒等"。
- **真实 fMP4 自测脚本**：`tools/si_proto/virtual_range_selftest.py` —— ffmpeg 生成合成 fMP4 → `BoxSplittingSink` 拆 box → 构建 `1×memory(init)+N×file(fragment)` 布局 → 500 个随机 Range 断言一致+稳定+整文件逐字节还原。实测 PASS（260KB fMP4，14 区域）。
> 意义：虚拟文件的"同一字节 N 恒等 / 稳定 Content-Length·ETag / Range 可靠"这一地基已可复跑验证，真机放行后接 `/media_si` 的风险更低。

## 产物清单
- `tools/si_proto/pyav_prototype.py` —— 可复跑的原型。
- `tools/si_proto/virtual_range_selftest.py` —— 对真实 fMP4 的虚拟 Range 字节稳定/一致性自测（可复跑）。
- `tools/si_proto/build_phase0_samples.py` —— 可复跑生成 Phase 0 三个静态测试 MP4（默认输出 `debug_output/si_proto/`）。
- `tools/si_proto/pyinstaller_av_probe.py` —— PyAV 冻结运行时探针。
- `debug_output/si_proto/A_progressive.mp4` / `B_fmp4.mp4` / `C_fmp4_dash.mp4` —— Phase 0 真机测试素材（30s，视频copy+混音）。
- `debug_output/si_proto/pyav_fmp4_out.mp4` —— PyAV 生成的 fMP4 样本（可对照 `mp4dump`/`MP4Box -info`）。
