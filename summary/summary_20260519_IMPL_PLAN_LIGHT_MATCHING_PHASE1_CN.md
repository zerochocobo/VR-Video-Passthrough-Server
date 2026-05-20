# 光照匹配 Phase 1 实施计划（中文）

- 日期：2026-05-19
- 范围：在 NV12 合成之前，对前景（抠出的人物）应用一组可由用户配置的"环境光匹配"滤镜，让视频主体在 Quest3 透传所看到的家居环境光下融合得更自然。绿幕模式和 alpha 模式均覆盖。
- 不在范围：神经网络重打光（IC-Light / SwitchLight）、方向性阴影投射、边缘 bounce light、参考照片色彩迁移。这些放入 Phase 2 及以后。

---

## 1. 背景

戴 Quest3 开透传时，用户看到的是：
- 自己家的真实环境（被家里灯影响：常见 3000K 暖白炽 / 4000K LED）；
- VR 播放器抠绿或解 alpha 后合成上去的视频主体。

如果视频是棚光环境（5500K 冷白）拍的，而房间是暖光，视频主体就会"飘在画面上"，俗称"贴片感"。要在不上重型 relighting 的前提下显著缓解这种感受，最低成本就是：**只对前景，每帧做一组色温/曝光/对比/gamma 校正**，并且在视频离开服务器之前就做完。

## 2. 目标

- 对每一帧 passthrough 输出的前景部分，应用用户可配置的色温、色调、曝光、对比度、gamma、饱和度，绿幕和 alpha 模式都覆盖。
- 在 RTX 20 系级别 GPU 上，8K 单帧额外开销控制在 0.5 ms 以内，**不影响现有 8K FPS 目标**。
- 桌面 UI 提供三档预设 + 完整手动控制。
- 设置存到 `ui_settings.json`，通过 `PT_*` 环境变量传给后端进程。

## 3. 非目标

- 不改背景（alpha 模式没有真正的背景；绿幕模式背景由播放器替换，改了也没用）。
- 不投阴影。
- 不自动检测家里灯光（Quest3 不开放透传摄像头给第三方应用）。
- 不导入 3D LUT（`.cube`），延后到 Phase 2。

## 4. 现有 Pipeline 参考点

- `pipeline/matting.py` 已经有 GPU 合成 kernel：
  - `composite_green_nv12_to_nv12`（`_COMPOSITE_NV12_TO_NV12_KERNEL_SRC`，约第 520 行）
  - `composite_green_nv12_upsample`（约第 446 行）
  - `composite_green_upsample`（约第 123 行）
- `pipeline/alpha_packer.py` 负责 alpha 模式 NV12 前景打包。
- `config.py:COMPOSITE_BG_RGB_HEX` 是目前唯一与颜色相关的运行时变量。
- 实时直通走 `pipeline/pynv_stream.py`，每帧调用 Matter 的 composite 方法。
- UI 设置流向：`ui/settings.py::server_env()` → 子服务进程环境变量 → `config.py` 读取 `PT_*`。

## 5. 设计

### 5.1 在 YUV (NV12) 域做颜色变换

现有合成 kernel 已经在 NV12 limited-range BT.709 域内工作。为压低成本，**直接在 YUV 域**做校正，再混入输出 NV12：

- **色温 / 色调**：(U, V) 上 2x2 仿射 + Y 上一个小偏置。主机端用预计算的 RGB 域 gain 矩阵，通过 BT.709 转换常数折算为 YUV 域的 9 个系数，作为 `__constant__` 上传：

  ```
  Y' = Y * y_gain + y_bias
  U' = (U - 128) * uu + (V - 128) * uv + u_bias + 128
  V' = (U - 128) * vu + (V - 128) * vv + v_bias + 128
  ```

- **曝光 (EV)**：Y 上乘 `2 ** ev`，折进 `y_gain`。
- **对比度**：`Y' = (Y - 128) * contrast + 128`，折进 `y_gain, y_bias`。
- **gamma**：分段非线性，只对 Y 做。每次设置变更时上传一张 256 字节 uint8 LUT。
- **饱和度**：`U' = (U - 128) * sat + 128; V' = (V - 128) * sat + 128`，折进 `uu, vv`。

所有线性操作合并进同一组 9 系数仿射，只有 gamma 需要 LUT 分支。

### 5.2 kernel 改动

- 在 kernel 源码字符串里新增一个公共 device function：`apply_light_match(y, u, v, coeffs, gamma_lut)`。
- 在 `composite_green_nv12_to_nv12` 和 `composite_green_nv12_upsample` 内部调用它：**采到源 Y/UV 之后、alpha 混合之前**。这样校正只作用于前景，不会污染绿色背景填充，绿幕键控仍然干净。
- `alpha_packer.py` 的 NV12 路径同样调用，只作用在前景 tile（alpha 打包的角落区域存的是 mask 数据，不动）。
- 当 `light_matching.enabled == false`，主机传 identity 系数 + identity LUT。kernel 入口加一个 uniform 分支 `if (!identity)` 短路，关闭状态下不产生测量得到的开销。

### 5.3 系数计算（主机端）

```
def build_light_match_coeffs(temp_k, tint, exposure_ev, contrast, gamma, saturation):
    # 1. temp_k + tint -> RGB gain（Bradford CAT 或简化的 Planckian -> sRGB）
    # 2. 曝光 -> RGB gain *= 2 ** ev
    # 3. RGB gain -> 3x3 RGB 变换
    # 4. RGB 变换 -> 2x2 UV 仿射 + Y 增益（用 BT.709 系数推导）
    # 5. 对比/饱和折进 Y/UV 仿射
    # 6. gamma -> Y 通道 256 项 uint8 LUT
    return Coeffs(...), gamma_lut_u8
```

纯 Python+numpy，< 1 ms。只在 UI slider 变化时触发，不在每帧路径。

### 5.4 配置项

| 环境变量 | 默认 | 范围 | 用途 |
|---|---:|---|---|
| `PT_LIGHT_MATCH_ENABLED` | `0` | 0/1 | 总开关 |
| `PT_LIGHT_MATCH_TEMP_K` | `5500` | 2700-9000 | 目标色温 (K) |
| `PT_LIGHT_MATCH_TINT` | `0` | -50..+50 | 绿-品红偏移 |
| `PT_LIGHT_MATCH_EXPOSURE_EV` | `0.0` | -2.0..+2.0 | 曝光档位 |
| `PT_LIGHT_MATCH_CONTRAST` | `1.0` | 0.5..1.5 | Y 域对比 |
| `PT_LIGHT_MATCH_GAMMA` | `1.0` | 0.7..1.4 | Y 域 gamma |
| `PT_LIGHT_MATCH_SATURATION` | `1.0` | 0.0..2.0 | UV 域饱和 |
| `PT_LIGHT_MATCH_PRESET` | `custom` | `home_warm`/`daylight`/`night_cool`/`custom` | UI 预设 key，决定哪些 slider 被锁 |

默认关闭。老用户体验完全不变。

### 5.5 UI 改动（`ui/pages/home_page.py`）

- 在现有性能区下方新增可折叠区块 "光照匹配 / Light Matching"。
- 一个开关 + 三个预设按钮（暖光客厅 3000K、日光 5500K、夜间冷光 6500K）+ 六条 slider。
- "重置" 按钮恢复默认。
- 一个 "?" tooltip 说明："调整视频主体的色彩以更好融合家中环境光。不会改变透传所看到的房间本身。"
- 三语 (`zh_CN`/`en_US`/`ja_JP`) 翻译 key 放在 `light_match.*`，文件保持 UTF-8 BOM（项目约定）。

### 5.6 后端串联

- `config.py`：新增 8 个常量，通过 `_env`/`_env_any` 读取，加一个 `LIGHT_MATCH_DICT` 快照辅助函数。
- `pipeline/matting.py`：
  - `Matter.__init__` 构造初始系数 + gamma LUT，上传到一个 12 float 的 CuPy device 数组和一个 256 字节 LUT。
  - 增加 `Matter.update_light_match(...)`（为后续 Phase 2 live reload 预留，Phase 1 不调用）。
  - 把两个 device 指针作为 kernel 启动参数的最后两位，最小化已有 kernel body 的 diff。
- `pipeline/alpha_packer.py`：同样的指针模式，在前景采样步骤应用。
- `pipeline/pynv_stream.py`：流启动时读一次配置，打日志记录生效值，构造 Matter 时传入。每帧 Python 路径无新增工作。

### 5.7 实时热加载

Phase 1 **不做** live reload。设置在流启动时读取，UI 改动要重启会话才能生效，避免跨线程同步问题。Live reload 放 Phase 2。

## 6. 性能预算

- 系数构造：设置变更时 < 1 ms（主机 CPU）。
- 单帧 8K kernel 增量：
  - Y 像素 9 个 FMA + UV 像素 4 个 FMA + Y 像素 1 次 LUT 查表。
  - RTX 2080 8K 估算 < 0.4 ms。
- 内存：每流 48 字节（12 floats）+ 256 字节 (gamma LUT)。可忽略。
- 关闭路径：每次 kernel 启动多一次 uniform 分支判断，无可测开销。

## 7. 验证计划

1. **数值单元测试**（`tests/test_light_match_coeffs.py`）：
   - identity 输入 (`temp_k=6500, tint=0, ev=0, contrast=1, gamma=1, sat=1`) 产生 identity 系数（float epsilon 内）。
   - `temp_k=3000` 让纯白向暖色偏移（gain 后 R > G > B）。
   - `saturation=0` 让 U,V 塌缩到 128。
   - `gamma=0.5` 抬中间调（LUT 第 64 项 > 64）。

2. **Kernel 参考测试**（`tests/test_light_match_kernel.py`）：
   - 在 4x4 NV12 fixture 上跑 identity 系数 kernel；断言字节级相等。
   - `temp_k=3000` 跑同样 fixture；断言 U/V 漂移方向正确。

3. **集成 smoke**：
   - 用 `PT_LIGHT_MATCH_ENABLED=1 PT_LIGHT_MATCH_TEMP_K=3000` 在 `videos/test_8k.mp4` 跑直通。
   - 与现有 baseline (`baseline/auto_tune_8k_phase1_*`) 对比 10 秒平均 FPS。
   - 验收：平均 interval FPS 与基线差 < 1 fps。

4. **VR 真机校验**：
   - Quest3 上对一段暖光房间环境，连续切换三档（关闭 / 暖光 3000K / 日光 5500K），用户主观确认"贴片感"明显减弱。

## 8. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| YUV 域仿射推导错误导致颜色不对 | 低 | 单元测试覆盖 identity + 3 个已知点。 |
| limited range 越界 | 中 | kernel 内 clamp Y 到 [16,235]，UV 到 [16,240]。 |
| 老 GPU 出现帧率退化 | 低 | 现有 composite 已经是 GPU 主开销，新增运算被融合进同一 kernel；以 1 fps 容差作为 gate。 |
| 用户疑惑（"为什么背景没变"） | 中 | tooltip + 预设描述都明确写明"仅作用于视频主体"。 |
| 默认开启后悔 | 不适用 | 默认关闭。老用户无感知。 |

## 9. 暂不实现但值得记录（Phase 2 候选）

- 边缘 bounce light（前景边缘吸取一点环境色）——是 Phase 1 之后感知收益最大的一项。
- Drop shadow（平面投影 + alpha blur，用户指定方向）。
- 3D LUT（`.cube`）导入。
- 流运行时的 live reload。
- 用户上传参考照片做 color transfer（Reinhard 算法）。
- 按片源记忆设置（per-file / per-directory）。

## 10. 文件改动清单

| 文件 | 改动 |
|---|---|
| `config.py` | 新增 8 个 `PT_LIGHT_MATCH_*` 常量及辅助。 |
| `pipeline/light_match.py`（新增） | 纯 Python 的系数 + LUT builder。 |
| `pipeline/matting.py` | NV12 合成 kernel 加 `apply_light_match`；新增上传与更新接口。 |
| `pipeline/alpha_packer.py` | 在前景 tile 上调用 `apply_light_match`。 |
| `pipeline/pynv_stream.py` | 流启动时读取配置快照、记日志，传给 Matter。 |
| `ui/settings.py` | 持久化 8 个字段；导出到 `server_env()`。 |
| `ui/pages/home_page.py` | 新增 "光照匹配" 面板 + 预设按钮。 |
| `ui/translations/{zh_CN,en_US,ja_JP}.json` | 新增 `light_match.*` 键（UTF-8 BOM）。 |
| `tests/test_light_match_coeffs.py`（新增） | 系数单元测试。 |
| `tests/test_light_match_kernel.py`（新增） | kernel 参考测试，CUDA 可用性 gate。 |
| `summary/summary_20260519_IMPL_PLAN_LIGHT_MATCHING_PHASE1_EN.md` | 英文版（本文档对应）。 |
| `summary/summary_20260519_IMPL_PLAN_LIGHT_MATCHING_PHASE1_CN.md` | 本文档。 |
| `prompt/HANDOVER_20260519.md` | 实施后追加交接记录。 |

## 11. 验收标准

- 全部 4 项单元 / 集成测试通过。
- 60 秒 8K 绿幕 + alpha 双模式跑分相对改动前基线差异 < 1 fps。
- UI 切换三档预设，在播放器里能明显看到画面变化。
- 设置在 UI 重启后保持。
- 关闭状态（默认）相对改动前输出字节等价（用 `/passthrough_live` 响应前 256 KB 哈希比对）。

## 12. 工作量估算

- 系数 + LUT builder + 单元测试：0.5 天。
- kernel 扩展 + alpha_packer + pynv_stream 串联：1 天。
- UI 面板 + 翻译 + 持久化：0.5 天。
- VR 真机调试 + 默认值调整：0.5 天。
- **合计：单人约 2.5 个工作日**。
