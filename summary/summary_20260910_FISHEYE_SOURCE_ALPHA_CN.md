# 原生鱼眼片源（_fisheye190 / MKX200 / VRCA220）走 Alpha 透视 —— 方案研究

> 纯研究，未改任何生产代码。结论与数字都来自本次跑的离线原型（合成网格图 +
> 真实素材 `[180 VR] STAYC - RUN2U.mp4` 转出的鱼眼测试帧），脚本在会话
> scratchpad，未落库。

---

## 0. 一句话结论

**现在给鱼眼源开 alpha 会出几何错误**（平均角度误差 7.4°、最大 27°、
对比参考画面 PSNR 只有 11.4 dB），因为 `is_sbs_vr_size()` 把双鱼眼 SBS 当成了
half-equirect SBS。修法很便宜：鱼眼→鱼眼只是**沿半径的线性缩放**，
新增一个 kernel 分支即可，实时几乎零额外成本；源是 F180 时甚至可以完全不重采样。
真正的难点不在几何，而在**识别源投影 + 知道源的 FOV**。

---

## 1. 现状：这条链路今天怎么走

alpha 透视的输出格式是固定的：**SBS 鱼眼 180（每眼方形、圆内切）+ 六块 alpha 布局**，
文件名后缀 `_LR_180_FISHEYE_F180_alpha`（`utils/vr_naming.py:35`），
播放器按 F180 网格渲染并从角块里取 matte。

输入侧只认三种源，判定全靠尺寸（`pipeline/alpha_packer.py:665`
`projection_mode_static`）：

| 判据 | projection_mode | kernel |
|---|---|---|
| `w >= 2*h` | `sbs_half_equirect` | `project_fisheye_nv12_alpha`（`alpha_packer.py:371`） |
| 非 2:1 + `ALPHA_2D_PROJECTION=fisheye` | `flat2d_fisheye` | `project_flat2d_fisheye_nv12_alpha` |
| 非 2:1 + `flat3d` | `flat2d_3d` | `project_flat2d_3d_nv12_alpha` |

**问题就在第一行**：一部 `xxx_fisheye190_LR.mp4`（7680x3840，每眼 3840 方形鱼眼）
同样满足 `w >= 2*h`，于是被 `fisheye_to_half_equirect()`（`alpha_packer.py:112`）
当成等距圆柱图去采样 —— 鱼眼画面被再投影一次，等于叠了两层畸变。

## 2. 错到什么程度（实测）

### 2.1 合成网格图（几何量化）

源：等距 fisheye190，单眼 768²；输出按 F180 网格解释。
"角度误差" = 播放器把某像素画在的方向 与 该像素内容真实来自的方向 之间的夹角：

| 路径 | 平均 | p95 | 最大 |
|---|---|---|---|
| A 今天的 hequirect kernel | **7.37°** | 19.67° | **27.06°** |
| B 鱼眼→鱼眼（FOV 给对，190） | 0.00° | 0.00° | 0.00° |
| B 但 FOV 猜成 180（源实为 190） | 3.33° | 4.87° | 5.00° |

最后一行正好等于 (190−180)/2 = 5° 的上限，可作为"FOV 猜错"的代价上界。

### 2.2 真实素材（端到端）

用 STAYC 的 equirect 母版分别生成 ①fisheye190 源 ②参考 F180 输出，
再把 ① 分别按两条路径转成 F180，与 ② 比：

| 路径 | PSNR vs 参考 F180 |
|---|---|
| 今天的路径（鱼眼当 hequirect 读） | **11.37 dB** |
| fisheye190 → F180 径向重映射 | **34.68 dB** |

11 dB 是"完全不同的画面"级别，肉眼一看就知道：见 `real_source_compare.png`
（源 / 今天 / 修好 / 参考 四联）。

## 3. 几何：为什么这件事其实很便宜

等距鱼眼里 `r = R · θ / (FOV/2)`。源与输出都是等距鱼眼时：

```
r_src / R_src = (FOV_out / FOV_src) · (r_out / R_out)
```

**纯半径线性缩放**，没有球面往返、没有三角函数。190→180 就是 `scale = 0.9474`，
200→180 是 `0.9000`。等价于"从源鱼眼圆里取一个同心圆放大到满幅"。

已用 ffmpeg 对账（项目里 `vr_reproject.py` 的老规矩）：
`v360=fisheye:fisheye:ih_fov=190:iv_fov=190:h_fov=180:v_fov=180` 的输出，
拟合出的最佳径向缩放是 **0.9460**，理论值 0.9474，吻合到 0.15%（拟合网格步长
0.002 就有 0.0014 的量化），PSNR 47.9 dB。
**结论：v360 的 fisheye 就是等距模型，离线用 ffmpeg 预处理和自写 kernel 数学一致**，
两条路可以互为验证/互为兜底。

顺带一个和本题无关但值得记的观察：六块 alpha 布局在 `PACK_SCALE=0.4` 下，
**每块有 13.7% 的面积压在可见鱼眼圆内**（圆边缘靠四角处）。要做到零重叠得把圆缩到
`radius_scale ≤ 0.85`，代价太大。这是格式的既有代价，与源投影无关，鱼眼源不会让它变差。

## 4. 三条落地路线

### 路线 1（推荐）：新增 `fisheye_src` 投影模式，输出仍是 F180

- `AlphaPacker` 增第四个 kernel `project_fisheye_src_nv12_alpha`：
  逐输出像素算 `rr`，`r_src = rr · (FOV_out/FOV_src)`，越界返回黑；
  alpha 在源鱼眼空间采（和现有 kernel 同一套 `sample_alpha_lr`）。
- **`FOV_src == FOV_out` 时退化成恒等映射** —— 对本来就是 F180 的源，
  可以再走一个 `fisheye_passthrough` 分支：RGB 一个字节都不动，只贴六块 alpha。
  这比今天任何一条 alpha 路径的画质都好，成本也最低。
- 输出命名、DLNA 条目、`alpha_output_size`（2:1 源返回源尺寸）全部不用动。
- 代价：190 源丢掉最外 5° 一环、200 源丢 10°。这一环基本是相机脚下/身后的地面，
  而且透视场景下多半正好是被抠掉（alpha=0）的背景，实际损失比数字看着小。

### 路线 2：保留源 FOV，输出打 `_FISHEYE190_alpha`

`utils/offline_outputs.py:16-22` 里已经列了 `_FISHEYE190_alpha`、`_SBS_F180_alpha`
等后缀，说明历史上试过这个方向。做法是 RGB 直通、只贴 alpha 块，命名交给播放器的
fisheye190 网格。**风险是无法从代码验证**：alpha packing 是播放器私有特性，
它是否与非 F180 网格组合生效，只能真机试（DeoVR / HereSphere / Skybox / 4XVR，
见 `resources/player_support.json`）。建议作为路线 1 的一个开关，而不是默认。

### 路线 3（**已选定并实现**，见 §11）：全视场装进 F180 容器（不裁）

**源圆 1:1 铺满输出圆**，什么都不缩。播放器按 F180 网格渲染，真实 95° 的边缘被画在
90° 上，误差 = 5·rr 度，最大 5°（§2.1 最后一行那 3.33°/5° 就是这个）。视野一点不丢。

> 这里要更正本文初稿的一句话：初稿写的是"把整圆缩到 0.947 R 放进 F180 帧"，**那是错的**。
> 缩到 0.947 R 之后，输出半径 rr 处的内容真实角度变成 95·rr/0.947 = 100.3·rr，
> 而播放器仍按 90·rr 渲染，误差翻倍到 **9.8°**。"圆变小 → 六块布局重叠降到 0"这个
> 附带好处要用双倍几何误差去换，不值得。F180 网格最多只能渲染到 90° 离轴，
> 所以在 F180 容器里几何正确地显示 >180° 的内容本来就不可能：要么裁（crop，角度精确），
> 要么 1:1 铺满（fit，视野完整、误差 ≤5°）。没有第三种。

1:1 还有一个实现上的好处：**`src_radial_scale = 1.0` 就是恒等映射，RGB 一个像素都不动**
（采样落在整数索引上，双线性权重为 0），等于零重采样直通。

## 5. 难点在识别，不在几何

### 5.1 尺寸没有信息量

双鱼眼 SBS 和 half-equirect SBS **尺寸完全一样**（每眼方形、整体 2:1）。
`is_sbs_vr_size()` 永远分不开这两者。

### 5.2 像素能可靠分开（实测）

鱼眼圆外是编码器都压不出内容的死黑，equirect 铺满整幅。取一帧、缩到每眼 256²，
量"内切圆外（rr>1.03）的亮度 p99"和"圆内环带（0.75<rr<0.95）的 p99"：

| 测试帧 | corner p99 | ring p99 | 判定 |
|---|---|---|---|
| equirect SBS（真实素材） | **157.0** | 214.0 | 非鱼眼 ✓ |
| fisheye190 SBS | **0.0** | 192.0 | 鱼眼 ✓ |
| fisheye180 SBS | **0.0** | 209.0 | 鱼眼 ✓ |

判据 `corner_p99 < 16 且 ring_p99 > 64` 三个样本全对。
第一版我用的是"角落均值/圆内均值"，**判反了**：那部 VR180 素材的 equirect
四角本来就很暗（均值 10.4），比值 0.078 直接落进鱼眼区间。**用 p99 不要用均值**
——鱼眼圆外是"没有任何亮像素"，equirect 暗角只是"平均暗"。

真实鱼眼片源角落会有编码噪声/块效应，阈值 16 有余量；再叠 2~3 帧投票更稳。
这个探测足够便宜，可以挂在 DLNA 列目录时那次 probe 上。

### 5.3 但 FOV 探测不出来

190 和 200 的圆都是内切满幅的，**像素上完全不可区分**。FOV 只能来自：

1. 文件名标记 —— `utils/vr_naming.py:22-27` 的正则**已经认识**
   `fisheye190 / mkx200 / vrca220 / rf52 / f180`，但只用来回答"有没有 VR 标记"，
   没有把 FOV 解析出来。需要一个 `parse_source_projection(stem) -> (kind, fov)`。
   映射表：`f180/fisheye180 → 180`、`fisheye190/rf52 → 190`、`mkx200 → 200`、
   `vrca220 → 220`。
2. 用户在设置里给默认值（探测到是鱼眼但文件名没标时用）。
3. 猜错的代价已量化在 §2.1：≤5°，不致命，但会让物体显得略近/略小。

半径检测（圆是否小于内切）值得一起做：有些片源上下有黑边、圆并不满幅，
那时 `radius_scale` 必须跟着改，否则整幅偏移。我的原型里也一并测了
（合成的 190 素材因为母版只有 180 内容，测出 0.945，正好等于 180/190 —— 这恰恰
说明**半径反映的是"圆多大"，不是"FOV 多少"**，两者不能互推）。

## 6. 抠像质量：鱼眼源不会更差，广角处反而更好

matting 是在源空间跑的（`matting.py:2608` 起，SBS 会左右眼各跑一次），
所以模型看到的是什么投影很重要。等距鱼眼 vs hequirect 的局部拉伸比：

| 离轴角 | hequirect（最坏方向） | 等距鱼眼 |
|---|---|---|
| 30° | 1.15x | 1.05x |
| 60° | 2.00x | 1.21x |
| 85° | 11.47x | 1.49x |

人基本在正前方 ±30° 内，两者都还好；**越往边缘鱼眼越占优**。
所以"直接在鱼眼上抠"是安全的起点，不需要为了质量先转 equirect。

如果以后要再提一档，可以照搬 RM 的 vr2flat 思路（`pipeline/vr_reproject.py`）：
把人所在区域 gnomonic 投成平面 → 抠像 → alpha 投回鱼眼。属于阶段 3，不是前置条件。

## 7. 性能

- 实时：新 kernel 与现有 kernel 同为"每输出像素一次双线性采样"，**成本持平**；
  F180 直通分支反而更省（省掉一次重采样）。
- 输出尺寸不变：2:1 源在 `alpha_output_size()`（`alpha_packer.py:41`）原样返回源尺寸，
  8K 鱼眼源 → 8K 输出，和今天的 8K equirect alpha 完全同档，
  码率估算（`utils/bitrate_estimator.py:73`）也不用动。
- 离线：`tools/offline_alpha_passthrough.py:475` 的 `_alpha_packer_from_args` 复用同一个
  packer，加参数即可；也可以选择在解码前用 `v360` 滤镜预处理（§3 已验证等价）。

## 8. 代码落点清单

| 位置 | 要做的事 |
|---|---|
| `utils/vr_naming.py:22` | 新增 `parse_source_projection()`，从 stem 解析 kind/FOV |
| `pipeline/alpha_packer.py:37,665` | `is_sbs_vr_size` / `projection_mode_static` 增加鱼眼分支（尺寸之外还要吃探测结果与文件名） |
| `pipeline/alpha_packer.py:112` 附近 | 新增 `fisheye_to_fisheye()` + `project_fisheye_src_nv12_alpha()`，以及 FOV 相等时的直通分支 |
| `pipeline/pynv_stream.py:2502-2505, 2692-2710` | 把探测出的 kind/FOV 传给 packer，并打进那行 alpha 日志 |
| `config.py:549-582`（ALPHA_2D 块旁） | `PT_ALPHA_SRC_PROJECTION=auto\|hequirect\|fisheye`、`PT_ALPHA_SRC_FOV`、`PT_ALPHA_SRC_RADIUS_SCALE`、`PT_ALPHA_SRC_FIT=crop\|fit` |
| `ui/settings.py:119-120,369-370` + `ui/dialogs/feature_dialogs.py:291` | 复用 `Alpha2DSettingsDialog` 的写法加一组"鱼眼源"选项 |
| `dlna/content_directory.py:1181`、`http_app/routes_media.py:2576` | 尺寸逻辑不变（2:1 原样返回），只需确认鱼眼源不会掉进 2D 分支 |
| `tools/offline_alpha_passthrough.py:475` | 加 `--src-fisheye-fov` / `--src-radius-scale` 透传 |

## 9. 建议的推进顺序

1. **先补探测**：`parse_source_projection()` + 像素探测（§5.2 的判据），
   只打日志不改行为，拿几部真实鱼眼片源验证判定与半径估计。
2. **再加 kernel**：路线 1，默认 `crop` 到 F180；F180 源走直通。
   用 §2.2 的方法（v360 生成参考帧）做逐帧 PSNR 回归，进 `tests/test_alpha_packer.py`。
3. **真机试路线 2/3**：`_FISHEYE190_alpha` 命名在四个支持 alpha 的播放器上到底认不认，
   这一步只能实测，代码上把它做成开关即可。
4. 可选阶段 3：gnomonic 平面化后抠像。

## 10. 还没有答案的

- **DeoVR/HereSphere 的 alpha packing 是否只在 F180 网格下生效**（决定路线 2 是否成立）。
- **MKX200 / VRCA220 是不是等距**。这两个是厂商镜头，很可能是等立体角或多项式畸变。
  若不是等距，径向映射要换成 1D LUT（256 项就够，运行时成本仍然是零）。
  等距假设用在非等距源上，误差会集中在最外圈，中心几乎不受影响。
- 真实鱼眼片源的圆是否总是内切满幅（决定半径探测要不要做成默认开启）。

---

## 11. 实现记录（2026-09-11）

按路线 3 落地，`fit` 为默认。实测源：`videos/TEST_4096p_91983_FISHEYE190_x265.mp4`
（SLR 的 190° 样片，8192x4096、10bit、60fps、64 分钟）。

### 11.1 改了什么

| 文件 | 改动 |
|---|---|
| `utils/vr_naming.py` | `parse_fisheye_fov()` 从文件名读 FOV；`strip_projection_markers()` 剥掉源的投影/生成标记；`alpha_passthrough_stem()` 改为"先剥离再加统一后缀" |
| `config.py` | `PT_ALPHA_SRC_FISHEYE`(auto/off)、`PT_ALPHA_SRC_FOV`(0=看文件名)、`PT_ALPHA_SRC_FIT`(fit/crop)、`PT_ALPHA_SRC_RADIUS_SCALE` |
| `pipeline/alpha_packer.py` | `alpha_src_fisheye_fov()` / `fisheye_src_radial_scale()`；device `fisheye_src_to_src()`；kernel `project_fisheye_src_nv12_alpha`；`projection_mode_static()` 增 `fisheye_src`；字幕 blend kernel 跟着走同一套源映射 |
| `pipeline/pynv_stream.py` | 实时两处（slot GOP builder、worker）都从源 stem 取 FOV 并传给 packer，日志加 `src_fisheye_fov/src_fit/src_radial` |
| `tools/offline_alpha_passthrough.py` | 从文件名**自动**识别，无需加参数；`--src-fisheye-fov` / `--src-fit` 可覆盖 |
| `utils/offline_outputs.py` | 输出名不再以源 stem 开头，匹配时多试一个"剥离标记后的源 stem" |

### 11.2 命名

`TEST_4096p_91983_FISHEYE190_x265` → `TEST_4096p_91983_x265_LR_180_FISHEYE_F180_alpha`。
源的 `FISHEYE190` 必须剥掉：留着的话一个文件名里会同时有 `FISHEYE190` 和 `FISHEYE_F180`
两个投影标记，播放器读到哪个是碰运气。源文件名里的 `_alpha` 同样被忽略/剥离。

### 11.3 实测（离线，4 秒 / 240 帧）

```
输出 8192x4096 HEVC，rc=0，240 帧，0.93 fps（8K RVM 的固有速度，与本次改动无关）
rvm_ort_avg = 991 ms   rvm_alpha_pack_avg = 16.9 ms   encode_avg = 0.47 ms
```

抽首帧与源首帧对比（都经 ffmpeg 缩到 2048x1024）：

| 检查项 | 结果 |
|---|---|
| 圆内画面（排除六块区域） | PSNR **34.63 dB** —— 几何完全对齐（几何错位会掉到 11 dB 量级，见 §2.2） |
| 圆外、块外 | 源 max=255（SLR 水印），输出 max=6 —— 圆外一律置黑，**水印顺带被清掉** |
| 六块区域 | 红通道占比 0.857，是 matte 不是画面 |

34.63 dB 而不是无损，来自 10bit→8bit、HEVC 编码和两次 ffmpeg 缩放，不是几何误差。

### 11.4 实测（实时）

`tools/debug_realtime_alpha_capture.py` 抓 20MB mpegts，解首帧：

- 输出 4096x2048（实时把 8K 缩了一档，仍是 2:1），画面圆内 vs 源 **PSNR 35.75 dB**；
- 圆外源 max=255（水印）、实时输出 max=3。

几何走错的话这里会是 11 dB 量级（§2.2），所以实时确实进了 `fisheye_src` 分支。

### 11.5 还没做/还没验的
- UI 没有加开关，走 `PT_ALPHA_SRC_*` 环境变量。默认 `auto` + `fit` 就是选定的行为。
- MKX200 / VRCA220 是否等距仍未验证（§10）。本次按等距处理；若不是，
  `fisheye_src_to_src` 的那一行径向缩放换成 256 项 1D LUT 即可，运行时成本不变。
