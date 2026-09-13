# VSR 原生 1x 增强 + 6K 中间档 实现记录（中文）

日期：2026-09-11
分支：`DLSS`
状态：已实现并实测通过

对应 [调研报告](summary_20260911_DLSS5_NR_AND_RTX_VSR_REVIEW_CN.md) 建议执行顺序的第 2、3 项。第 1 项 TrueHDR 见 [summary_20260911_TRUEHDR_OFFLINE_SUPERRES_CN.md](summary_20260911_TRUEHDR_OFFLINE_SUPERRES_CN.md)。

---

## 1. 最重要的发现：NGX VSR 的耗时由输入分辨率决定，不是输出

这条实测结果推翻了调研报告里的推断（那份报告假设「compute-bound → 帧率随输出像素线性下降 → 6K 能跑到 40 FPS」）。

RTX 5060 Ti，quality 3，同一段素材，`PT_RTX_VSR_STAGE_TIMING=1` 测得的每眼 NGX 耗时：

| 输入（每眼） | 输出 | NGX ms/眼 | 整链 FPS |
|---|---|---|---|
| 1920×1920 | 8192×4096 | 17.2 | 25.3 |
| 1920×1920 | 6144×3072 | 16.1 | 27.2 |
| 1920×1920 | 3840×1920（原生 1x） | 15.1 | 30.4 |
| **960×960** | 8192×4096 | **6.2** | 34.9 |

输出像素从 8K 降到 6K 是 −44%，耗时只降 7%；输入像素降到 1/4，耗时降到 1/2.8。**决定成本的是输入。**

直接推论：
- 4K VR 源（每眼 1920×1920）无论输出多大，NGX 都要约 17 ms × 2 眼 ≈ 34 ms/帧，也就是**上限约 29 FPS**。实时 60 FPS 只有降低输入才可能。
- 6K 档不是性能档。它的价值是**输出像素只有 8K 的 56%**，码率和头显解码负担显著降低 —— 给解 8K HEVC 吃力的一体机用。
- 原生 1x 对 VR 同样省不下多少 NGX 时间（15.1 vs 17.2 ms），省的是输出体积；对 2D 则很快（见下）。

---

## 2. 实现的两件事

### 原生 1x 增强（目标档 = 0）

VSR 在源分辨率上评估，不放大，只做锐度/细节增强。实测有效：低码率 720p 素材上，头发丝、轮廓、边缘明显更干净（PSNR 反而从 40.5 降到 39.9 —— 锐化就是这样，**不能宣称它是去压缩伪影的保真恢复**）。

关键门控变化：放大模式的输入上限（`PT_RTX_VSR_INPUT_MAX_HEIGHT`，默认 1440）对 1x **不适用** —— 1x 的成本就是源分辨率的成本。1x 改用自己的上限 `PT_RTX_VSR_NATIVE_MAX_HEIGHT`（默认 4096）加 NVENC 的 8192 边长封顶。因此原先被「超过 1440p」挡住的 4K 2D、2160p 竖屏素材现在可以处理。

### 6K 中间档（目标档 = 3072）

VR 源输出 6144×3072（分眼各 3072×3072），2D 源与 4096 档一样收敛到 3840×2160。

---

## 3. 改动清单

| 文件 | 改动 |
|---|---|
| `utils/rtx_vsr.py` | `NATIVE_TARGET_HEIGHT=0`、`ENCODER_MAX_SIDE=8192`、`resolve_target_height()`、`is_native_target()`；`target_resolution()` 加 6K 分支与 native 分支；`target_dimensions()`/`source_exceeds_target_resolution()`/`effective_offline_target_height()`/`source_block_reason()` 全部支持 native |
| `config.py` | `PT_RTX_VSR_TARGET_HEIGHT` 允许 0（原先 `max(2, …)` 会把 0 变成 2）；新增 `PT_RTX_VSR_NATIVE_MAX_HEIGHT` |
| `utils/vr_naming.py` | 输出后缀加 `_1X` 与 `_6K`，重复后缀正则一并更新 |
| `offline/convert.py` | 目标解析改用 `resolve_target_height()`（`or` 写法会把 0 吞成默认值）；批量发现跳过 `_1x`/`_6k` |
| `offline/rtx_vsr.py` | 同上；6K 与 8K 一样禁用整帧 FFmpeg 回退（VR 专属档必须走分眼 GPU 路径） |
| `offline/rtx_vsr_pynv.py` | VR 源在 1x 下同样分眼评估 |
| `pipeline/pynv_stream.py` | 实时路径同上 |
| `dlna/content_directory.py` | 目录可见性：1x 时输入高度改用 native 上限 |
| `ui/superres_targets.py`（新增） | 档位与文案 key 的唯一来源，三处 UI 共用 |
| `ui/pages/superres_page.py`、`ui/dialogs/feature_dialogs.py`、`ui/pages/dashboard_page.py` | 目标下拉从 3 档扩到 5 档（原生 / 2K / 4K / 6K VR / 8K VR），首页摘要文案共用同一映射 |
| `ui/translations/*.json` | 三语新增 `superres.target_native`、`superres.target_6k_vr` |

### 一个容易踩的坑

`0` 是合法的目标高度（原生），但项目里到处是 `int(target_height or config.RTX_VSR_TARGET_HEIGHT)` 这种写法，会把 0 静默换成默认值。所有这类位置都改成了显式的 `resolve_target_height()`，并在 `utils/rtx_vsr.py` 顶部写明了这条约束。UI 侧的 `int_setting()` / `_setting_value()` 用的是 `is None or == ""`，对 0 是安全的。

---

## 4. 实测

RTX 5060 Ti，quality 3，8 秒素材，整链（NVDEC → VSR → NVENC → mux）：

| 场景 | 输入 → 输出 | FPS |
|---|---|---|
| 2D 原生 1x | 1216×2160 → 同 | **75.3** |
| VR 原生 1x | 3840×1920 → 同（分眼 1920×1920） | **30.4** |
| VR 6K | 3840×1920 → 6144×3072 | **27.2** |
| VR 8K（对照） | 3840×1920 → 8192×4096 | **25.3** |

输出抽帧检查：原生 1x 的 VR 输出左右眼拼接正确、无接缝错位；2D 1x 与源对比锐度提升明显。

命名：`[SuperRes]<名字>_1X.mp4` / `_6K.mp4`，重复运行会替换旧后缀，批量发现会跳过这两类成品。

---

## 5. 测试

- `tests/test_rtx_vsr.py` 新增 6 项：native 不被 `or` 吞掉、native 保持源尺寸、6K 是 VR 专属档且 2D 回落 4K、native 门控替换放大门控（含编码器边长上限）、`_1X`/`_6K` 命名、UI 档位与文案 key 一一对应。
- `tests/test_offline_convert.py`：批量发现跳过 `_1X`/`_2K`/`_4K`/`_6K`/`_8K` 成品。
- 全量：`648 passed, 2 skipped`。

---

## 6. 边界与后续

- 实时 VR 的 NGX 上限就是每眼约 17 ms（4K VR 源），**6K 档不会让实时 VR 超分变成 60 FPS**。真要提实时帧率，唯一方向是降低送进 NGX 的输入分辨率（例如从 2K VR 源放大），或者接受 30 FPS 输出。
- 8K VR 源在实时侧仍然隐藏（DLNA 层 `width > 4096` 的判断未变）；离线的 1x 允许到 8192 边长。
- 原生 1x 是锐度增强，不是保真恢复；对已经很干净的源提升有限。


---

## 7. 后续：虚拟文件（可拖动进度条）链路接入超分

### 接之前的真实状态

不是「分辨率声明错」那么简单 —— 虚拟文件用的帧生成器开头就是：

```python
if mode not in {"green", "alpha"}:
    raise RuntimeError(f"slot PyNv GOP builder does not support output_mode={mode!r}")
```

所以 SuperRes 走虚拟文件**根本跑不起来**，会在生成第一帧时抛错。同时 `_vmp4_slot_output_size()` 里 `alpha` 和 `two_dvr` 各有分支、`superres` 没有，外壳会声明源分辨率。两处都要补。

字节预算反而不是障碍：那一层早就为 superres 特判过（不继承源预算，退回固定预算），8K 溢出问题也在 alpha 的 8K VR 上解决过（自适应 headroom 闭环）。实测 240 帧**没有一帧超预算**。

### 做法

1. VSR 处理抽到 `pipeline/superres_stage.py`，实时 worker 和虚拟文件帧生成器共用一个 stage —— 分眼、缓冲复用、NVENC 输入环、SDR 观感只有一份实现。
2. 帧生成器接受 `superres` 模式，输出尺寸取自 stage。
3. `_vmp4_slot_output_size()` 补上 superres 分支，和 stage 问同一个函数要目标，外壳声明的分辨率必然等于实际编码的分辨率。
4. 虚拟文件链路调用 `seek_target_height()` 而不是配置的档位：默认降级为原生 1x。

### 为什么降级到 1x

这条链路是**按播放速度被拉流**的，stage 必须跟得上播放。而 NGX 耗时看输入，放大与否几乎不影响成本 —— 4K VR 源无论输出 6K 还是 8K 都是 25～27 FPS，撑不起 60fps 的片子；原生 1x 至少把输出体积和编码压力降下来了。

逃生开关 `PT_RTX_VSR_SEEK_ALLOW_UPSCALE=1` 保留配置档位，但**未在真实播放器上验证过**。

### 实测（RTX 5060 Ti，虚拟文件链路，原生 1x，默认质量档）

| 源 | 输出 | FPS | 超预算帧 |
|---|---|---|---|
| 1216×2160 2D | 同尺寸 | 37.4 | 0 / 240 |
| 3840×1920 SBS VR | 同尺寸 | 27.2 | 0 / 240 |

比离线同源慢（离线 2D 75 FPS、VR 30 FPS），因为这条链路是逐帧 yield + Annex-B 拆分，没有离线那样的流水线。

**含义**：30fps 的片子 2D 够用、VR 略紧（27 < 30，靠后台预生成缓冲补）；60fps 的片子两者都不够。这个判断来自吞吐数字，**真机拖进度条的表现仍未验证**。

### 还没做

- UI 不提供任何新开关，虚拟文件/直播流仍然是 Alpha 和绿幕那个全局选择。
- 放大档在虚拟文件下的表现未验证（默认走不到）。
- 真实播放器的拖动、跳转、剩余时间显示未验证。

### 性能追查：修掉一个真 bug，但主因仍未定位

虚拟文件链路比离线慢约 1.9 倍（同源同档：离线 75.7 FPS，虚拟文件 39.9 FPS）。追查过程与结论：

1. **每帧重建 NGX（已修）**。桥把 CUDA stream 也算进会话身份，而调用方在 CuPy stream 上下文里跑，每帧 stream 指针都不同 → 每帧 shutdown+create。240 帧的运行里 native create 被调用了 **241 次**。改为只按 CUDA context 判身份后降到 **1 次**，离线无回归（75.3 → 75.7），虚拟文件 37.4 → 39.9 FPS（**+7%**）。这个 bug 对实时链路同样存在，修复后实时也应受益（未单独实测）。
2. **CBR 每帧预算不是主因**。关掉每帧字节预算（退回 VBR）只从 38.5 升到 41.2 FPS，同样约 7%。
3. **profile 的表面读数是伪影**。cProfile 和 perf_counter 都把约 3.9 秒记在 `initialize` 上，但隔离基准测出它只要 **0.002 ms/次** —— 那段时间是被计到该函数头上的调度/等待，不是它自己的开销。

**剩下的约 1.9 倍差距没有定位。** 下一步应该用 CUDA event 而不是 CPU 计时来分解这条链路，先判断是 GPU 侧真的更慢，还是 Python 侧的逐帧 yield / AU 拆分在串行化流水线。在有测量依据之前，不应该声称原因。

### 质量档的代价（实测，离线链路，4K VR 源原生 1x）

| 质量档 | NGX ms/眼 | 整链 FPS |
|---|---|---|
| 2 中 | 6.5 | **62.2** |
| 3 高 | 15.2 | 30.3 |
| 4 超高 | 20.8 | 22.5 |

**中档到高档是 2 倍代价，高档到超高档再加 35%。** 离线批量处理如果不追求极限画质，质量档选「中」能直接把吞吐翻倍。

但同样的质量档在**实时和虚拟文件链路上几乎不起作用**：4K VR 源上 quality 1 是 29.8 FPS、quality 2 是 27.6 FPS，只差 7%，而离线同档是 62 FPS。这说明那两条链路的瓶颈不在 NGX。已排除的候选：解码器（simple vs threaded_serial 差 7%）、CBR 每帧预算（关掉差 7%）。**瓶颈仍未定位**，下一步应当用 CUDA event 分解，而不是继续用 CPU 计时猜测。

### 虚拟文件 + 放大档的实测（含一个真 bug）

开 `PT_RTX_VSR_SEEK_ALLOW_UPSCALE=1` 走 6K，第一次测试**播不了**：

```
Invalid NAL unit size (267026 > 262140)
服务端日志：VMP4 frames oversize idx=3 267030>262144 (clamped)
```

每帧 256KB 的固定预算是按绿幕/Alpha 调的 —— 那两种输出要么是源几何、要么是抠像后大面积平坦的画面，都好压。超分放大后的画面细节密度高得多，第 3 帧就超了预算被截断，自适应 headroom 是逐 GOP 的事后反应，救不了开头几帧。

修复：**仅对超分**按输出像素相对 4K 基准缩放预算（上限 4 倍）。缩放后实测：

| 场景 | 每帧预算 | 带宽 @30fps | 结果 |
|---|---|---|---|
| 绿幕/Alpha 任意尺寸 | 256 KiB | 63 Mbps | 不变 |
| 超分 原生 1x（3840×1920） | 256 KiB | 63 Mbps | 240 帧解码通过 |
| 超分 6K（6144×3072） | 704 KiB | 173 Mbps | 240 帧解码通过，零超预算 |

6K 输出的左右眼拼接正确，中段 Range 也能落到正确 GOP。但带宽是 1x 的 2.7 倍，且服务端仍然只有约 27 FPS 的产出能力，所以 `PT_RTX_VSR_SEEK_ALLOW_UPSCALE` 保持默认关闭 —— 能用不等于该用。

### 最终设计：档位决定播放形态（取代静默降级）

早期实现是"虚拟文件链路一律按原生 1x 渲染"，用户选 6K 也照播 1x。这是静默降级，而且播放器（Skybox）会缓存条目名称，想靠标题标注 `[SUPERRES 1X]` 来提示也行不通 —— 名字一旦被缓存就再也改不了。

改成由档位决定形态：

| 超分档位 | DLNA 条目形态 | 播放模式可选 |
|---|---|---|
| 原生 1x | 虚拟文件 或 直播目录 | 是，与 Alpha 共用同一个全局选择 |
| 2K / 4K / 6K / 8K | 仅直播章节目录 | 否，对话框显示原因说明 |

- `seek_target_height()` 不再替换档位，只返回配置值；是否提供 seek 形态由 `seek_supported_target()` 决定。
- DLNA 浏览侧据此只为原生档发布 seek 条目；seek 路由对不可 seek 的档位返回 409，不会半途播出别的东西。
- 实时超分对话框加入与 Alpha 相同的 `PlaybackModeChooser`，仅在原生档位显示。
- 条目标题保持 `[SUPERRES]`。

实测（运行中的服务，4K SBS VR 源）：6K 档请求 seek → `409 SuperRes virtual-file playback requires the native 1x target`；1X 档 → 200，输出 3840×1920，240 帧完整解码通过。

### 默认值与界面提示

- 实时超分默认档位改为**原生增强**（`ui/settings.py` 的 `superres_target_height` 与 `config.py` 的 `PT_RTX_VSR_TARGET_HEIGHT` 同步为 0）—— 它是唯一能同时提供虚拟文件形态的档位。
- 离线超分默认仍是 8K VR 档：离线不受播放速度约束，放大才是它的价值所在。两个默认值在 `ui/superres_targets.py` 里分开定义（`REALTIME_DEFAULT_TARGET` / `OFFLINE_DEFAULT_TARGET`）。
- 实时对话框里那条「只有原生增强才能用虚拟文件播放」的说明**在任意档位下常驻**，而不是只在失去选择时才出现 —— 它是决定上方选择器存不存在的规则，应该在选档位之前就被读到。
- 已存的用户设置不会被覆盖：改的是默认值，之前手动选过档位的用户保持自己的选择。
