# TensorRT 加速 + 引擎缓存 UI 实施计划（中文）

- 日期：2026-05-20
- 范围：在桌面 UI 首页"性能配置"面板中新增 TensorRT 加速开关与引擎缓存管理；提供首次缓存的进度反馈与驱动/版本变更检测；保证关闭加速时可随时无损回退到 CUDA 路径。
- 不在范围：动态切换 provider（不重启服务）、跨机器分发预编译 engine、为 offline 工具（`offline/sam3_matanyone2.py`、`offline/yoloworld_efficientsam.py`）做 TRT 缓存。

---

## 1. 背景

8K 实时透传在 v0.1.0-beta.1 后从 40-50 FPS 掉到 30+ FPS，根因是为了画质把 `ALPHA_STRIDE` 从 3 改成 1。回退会引起黑十字与远景人物丢失，因此画质决策保留，**性能损失改由 TensorRT 加速 RVM 推理来弥补**。

本期使用 `rvm_mobilenetv3_fp32.onnx` 作为唯一推理模型（fp16 变体经验证在 VR 场景画质损失明显，已弃用）。**性能提升完全由 TensorRT FP16 后端从 fp32 ONNX 直接编译时落地**——TRT 在 build 阶段以 FP16 精度构造 engine（`trt_fp16_enable=1` 默认开启），同时保留 fp32 ONNX 的权重精度作为编译输入，在 RTX 2080 上对 RVM 的实测增益通常仍有 1.5-2×。但首次启动需要编译 engine（5-15 分钟），driver/CUDA/TRT/ORT/模型任何一项变化都会让缓存失效。我们要做的不是把 TRT "开起来"——`ONNX_TRT_FP16_ENABLE` 默认已经是 1，只是 `ONNX_PROVIDERS` 没把 `TensorrtExecutionProvider` 放进去——而是给用户一套**可见、可控、可撤销**的 UI 流程。

**TensorRT 加速只针对 `rvm_mobilenetv3_fp32.onnx` 这一个模型**。其他模型不在本期加速范围。

**输入策略**：`MATTING_INPUT_SIZE=1024` 固定方阵 + `RVM_DOWNSAMPLE_RATIO=0.5`。经验证，长边 2048 + 0.125 下采样会导致 VR 场景下人像抠不完整甚至丢失，因此保留 1024 方阵这一对 TRT 极友好的策略：只有两种固定 shape（`1×3×1024×1024` 和 `2×3×1024×1024`），TRT 各编一个 static engine，缓存最稳、kernel 最优、profile 无烦恼。

## 2. 目标

- 用户在 UI 首页性能配置中能看到一个"TensorRT 加速"开关，状态语义清晰。
- 首次启用前必须先缓存引擎，缓存过程显示阶段化进度和实时计数，可取消。
- 缓存完成后开关才能正常打开，重启服务后秒级加载。
- 驱动 / CUDA / TRT / ORT / 模型变更后自动识别失效，UI 显示"需重新缓存"。
- 关闭开关或缓存失效时，服务自动回退到 CUDA EP，不影响正常播放。
- 缓存可以独立于开关存在：关闭开关不删缓存，下次再开秒生效。

## 3. 非目标

- 不实现 ORT 的运行时 provider 热切换（ORT 不支持，硬做会引入大量边界情况）。
- 不打包预编译 engine 随安装包分发（engine 绑定具体 GPU/driver，跨机不可用）。
- 不让用户在 UI 上挑细粒度的 engine（batch=1 / batch=2 / 不同 recurrent state shape），暴露到模型粒度即可。
- 不引入新的 RVM 之外的模型。

## 4. 现有 Pipeline 参考点

- `pipeline/matting.py:880-882` 已有 `TensorrtExecutionProvider` 的 provider options 注入逻辑，吃 `ONNX_TRT_FP16_ENABLE` 和 `ONNX_TRT_CUDA_GRAPH_ENABLE`。
- `pipeline/matting.py:977-1003` 在 `InferenceSession` 创建时使用 `_filter_available_providers(ONNX_PROVIDERS)`。当前 `ONNX_PROVIDERS` 默认值在 `config.py:394-397` 是 `CUDAExecutionProvider,CPUExecutionProvider`，没有 TRT。
- `config.py:403-408`：`ONNX_TRT_ENGINE_CACHE_ENABLE=1`、`ONNX_TRT_ENGINE_CACHE_PATH=ROOT/runtime_cache/trt_engines`、`ONNX_TRT_FP16_ENABLE=1`、`ONNX_TRT_CUDA_GRAPH_ENABLE=1` 已就位。
- `ui/services/server_process.py` 已有服务进程的启动/停止/重启原语。
- `ui/settings.py::server_env()` 把 UI 配置以 `PT_*` 环境变量形式传给服务子进程；TRT 的开关从这里走最自然。
- `pipeline/matting.py:_supports_batch2`（约 line 1003）决定是否同时编 batch=2 的 SBS engine。

## 5. 设计

### 5.1 三方信息流

```
UI 性能面板（开关 / 缓存按钮 / 进度）
    │
    ├── 写 ui_settings.json（持久化 trt_enabled 等）
    ├── 读 runtime_cache/trt_engines/manifest.json（缓存状态）
    └── 触发独立的"缓存子进程"（一次性 warmup）
            │
            └── 调用 onnxruntime InferenceSession，让 TRT 编译并落盘
                完成后写 manifest.json
                出错则清理半成品 engine 文件

服务主进程 ptserver-server
    │
    └── 启动时根据 trt_enabled + manifest 的有效性，
        决定 providers 列表是否包含 TensorrtExecutionProvider
```

**关键约束**：缓存编译走**独立子进程**，与正在运行的服务隔离。理由：
- ORT InferenceSession 是阻塞调用，编译期间无法响应取消。
- 编译会占用大量 VRAM，与服务同进程容易抢资源。
- 子进程崩溃不影响服务，UI 可以 kill 子进程实现"取消"。

### 5.2 引擎缓存清单（manifest.json）

路径：`runtime_cache/trt_engines/manifest.json`

```json
{
  "version": 1,
  "fingerprint": {
    "gpu_uuid": "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "gpu_name": "NVIDIA GeForce RTX 2080",
    "driver_version": "560.94",
    "cuda_runtime": "12.4",
    "trt_version": "10.0.1.6",
    "ort_version": "1.20.0",
    "model_sha256": "ab12...",
    "matting_input_size": 1024,
    "rvm_downsample_ratio": 0.5,
    "trt_fp16": true,
    "trt_cuda_graph": true
  },
  "models": [
    {
      "key": "rvm_mobilenetv3",
      "label": "Robust Video Matting",
      "engines": [
        {"shape": "1x3x1024x1024", "size_mb": 38, "built_at": "2026-05-20T14:33:00Z"},
        {"shape": "2x3x1024x1024", "size_mb": 68, "built_at": "2026-05-20T14:39:30Z"}
      ],
      "total_build_seconds": 452,
      "status": "ready"
    }
  ],
  "built_at": "2026-05-20T14:39:30Z"
}
```

字段说明：
- `fingerprint`：环境指纹，任意一项变化都让缓存进入 `stale` 状态。
- `fingerprint.matting_input_size` / `rvm_downsample_ratio`：输入策略指纹，长边或下采样改了缓存失效。
- `models[].status`：`ready` / `stale` / `failed`。
- `models[].total_build_seconds`：用于下次预估耗时。

### 5.3 UI 设计

性能配置面板**最下方新增一行**：

```
TensorRT 加速    [○○]    [配置]    未缓存
```

构成（从左到右）：
1. 标签：`TensorRT 加速`
2. 开关 (`QCheckBox` 或自定义 toggle)
3. 配置按钮：`[配置]`
4. 状态文字（短文案，无图标）

状态文字 4 种：

| 缓存状态 | 文字 |
|---|---|
| 未缓存 | `未缓存` |
| 缓存中 | `缓存中…` |
| 已缓存最新 | `已缓存` |
| 已缓存失效 | `需重缓` |

**所有其他信息（上次耗时、驱动版本、引擎大小、阶段进度等）一律放到点击"配置"后弹出的对话框里**，主面板这一行只承担"是否启用 + 是否就绪"两个语义。

开关交互规则：
- 缓存状态为 `已缓存` 时，开关可正常切换，切换后底部 `保存` 按钮按现有性能面板流程触发重启服务。
- 其他三种状态下，开关被禁用为关；点击开关弹气泡 "请先在配置中完成缓存"。

配置按钮点击 → 弹出 5.4 描述的对话框。该对话框内承载所有详情、缓存触发、重缓、删除、进度展示。

### 5.4 配置对话框

点击主面板的 `[配置]` 按钮 → 弹出 modal 对话框。对话框是 TensorRT 相关所有详情和操作的唯一入口。

#### 5.4.1 未缓存 / 已缓存最新 / 已缓存失效 状态

```
┌─ TensorRT 加速配置 ───────────────────────────┐
│                                                │
│  模型：rvm_mobilenetv3_fp32.onnx               │
│  TRT 精度：FP16（由 fp32 ONNX 编译）             │
│  GPU：NVIDIA GeForce RTX 2080                  │
│  驱动：560.94                                   │
│  TensorRT：10.0.1.6                            │
│                                                │
│  缓存状态：已缓存                                │
│  上次编译耗时：7 分 32 秒                        │
│  引擎大小：106 MB                                │
│  缓存路径：runtime_cache/trt_engines/           │
│                                                │
│  ⓘ 启用 TensorRT 后，server 以 TensorRT FP16   │
│    后端从 fp32 ONNX 编译 engine 进行推理。      │
│                                                │
│  ⚠ 编译期间 GPU 会被占用，请勿同时播放视频。      │
│                                                │
│       [删除缓存]   [关闭]   [开始缓存]            │
└────────────────────────────────────────────────┘
```

底部主操作按钮按当前状态切换文案：

| 状态 | 底部主按钮 | 次按钮 |
|---|---|---|
| 未缓存 | `开始缓存` | — |
| 已缓存最新 | `重新缓存` | `删除缓存` |
| 已缓存失效 | `重新缓存`（高亮，提示"驱动已升级"） | `删除缓存` |

#### 5.4.2 缓存中状态

点击 `开始缓存` / `重新缓存` 后，**同一对话框**切换到进度视图（不再弹第二层 modal）：

```
┌─ TensorRT 加速配置 ───────────────────────────┐
│                                                │
│  正在编译 rvm_mobilenetv3_fp32.onnx (TRT FP16)  │
│                                                │
│  [████████████░░░░░░░░░░░░] 阶段 2 / 3         │
│                                                │
│  当前阶段：编译 SBS 双眼引擎                     │
│  已耗时：04:21 / 预计 08-12 分钟                │
│  已生成引擎：1 个                                │
│                                                │
│  ⚠ 关闭对话框等于取消编译，缓存将不可用。        │
│                                                │
│                          [取消]                  │
└────────────────────────────────────────────────┘
```

进度构成（**不承诺百分比精度，只承诺阶段**）：
1. 阶段 1/3：编译单眼引擎（batch=1, 1024×1024）
2. 阶段 2/3：编译 SBS 双眼引擎（batch=2, 1024×1024）
3. 阶段 3/3：固化运行时缓存（warmup runs）

"已生成引擎"用 `QFileSystemWatcher` 监听 `runtime_cache/trt_engines/` 目录的 `.engine` 文件数量。

编译完成后对话框自动切回 5.4.1 视图，状态显示 `已缓存`，主面板状态文字同步更新。

### 5.5 后端实现

#### 5.5.1 缓存子进程

新增 `ui/services/trt_warmup_process.py`，CLI 入口：

```bash
ptserver-trt-warmup --model rvm --input-size 1024 --downsample 0.5 \
                    --fp16 1 --cuda-graph 1 \
                    --cache-dir runtime_cache/trt_engines \
                    --progress-stdout
```

子进程行为：
1. 打印 `STAGE:1:start:编译单眼引擎`
2. 创建 batch=1 session（固定 shape `1×3×1024×1024`，`r1i~r4i` 按 stride 4/8/16/32 推导出对应固定 shape），运行一次推理触发 TRT 编译
3. 打印 `STAGE:1:done`
4. 创建 batch=2 session（如果 `_supports_batch2`），固定 shape `2×3×1024×1024`，运行一次推理
5. 打印 `STAGE:2:done`
6. 跑 `MATTING_WARMUP_RUNS` 次预热
7. 打印 `STAGE:3:done`
8. 写 manifest.json
9. 退出码 0 = 成功，非 0 = 失败

UI 用 `QProcess` 监听 stdout，按行解析。

#### 5.5.2 manifest 模块

新增 `utils/trt_manifest.py`：

```python
def collect_fingerprint() -> dict
def manifest_path() -> Path
def load_manifest() -> dict | None
def save_manifest(manifest: dict) -> None
def cache_status() -> Literal["missing", "ready", "stale", "failed"]
def stale_reasons(saved_fp: dict, actual_fp: dict) -> list[str]
def clear_cache() -> None  # 删 runtime_cache/trt_engines/ 全部内容
```

`collect_fingerprint()` 用 `pynvml` 拿 GPU UUID / name / driver；用 `onnxruntime.__version__`；TRT 版本走读 nvinfer DLL 文件版本号（已在 prompt/PYINSTALLER_PACKAGING_REPORT 里用过 ctypes 方案）。

#### 5.5.3 启动时的自动校验

`main.py` 启动前：
1. 读 `ui_settings.trt_enabled`。
2. 若为 true，调 `cache_status()`：
   - `ready` → 在 providers 列表头插入 `TensorrtExecutionProvider`。
   - 其他 → 写日志 `trt cache invalid (reason=...), falling back to CUDA EP`，不传 TRT，继续正常启动。
3. UI 主进程并行做同样检测，更新缓存状态徽章。

服务启动这一侧**永远不在主进程里触发 TRT 编译**。缓存只在 UI 主动触发的子进程里发生。

#### 5.5.4 settings.py 增项

`ui/settings.py` 新增：
- `inference_backend`：`"cuda" | "tensorrt"`，默认 `"cuda"`。
- `server_env()` 在 `inference_backend == "tensorrt"` 且 `cache_status() == "ready"` 时，注入 `PT_ONNX_PROVIDERS=TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`，否则维持原值。

注意：`config.py` 那一侧不需要改默认值，新加的环境变量直接覆盖。

### 5.6 状态机

```
                     ┌──────────────┐
                     │  CUDA (默认)  │
                     └──────┬───────┘
                            │ 切到 TRT，但未缓存
                            ▼
                     ┌──────────────┐
                     │ 阻塞：先缓存  │
                     └──────┬───────┘
                            │ 用户点开始缓存
                            ▼
                     ┌──────────────┐
                     │   编译中      │ ←──── 取消（kill 子进程）
                     └──────┬───────┘
                            │ 成功
                            ▼
                     ┌──────────────┐
                     │  已缓存 ready │
                     └──────┬───────┘
                            │ 应用并重启服务
                            ▼
                     ┌──────────────┐
                     │ TRT 服务运行  │
                     └──────┬───────┘
                            │ 检测到 driver/版本变更
                            ▼
                     ┌──────────────┐
                     │  stale 黄态  │
                     └──────────────┘
```

### 5.7 异常处理

| 场景 | 处理 |
|---|---|
| 编译时 GPU OOM | 子进程退出码非 0，UI 显示具体错误 + 建议（关掉其他占 VRAM 进程） |
| 用户点取消 | UI kill 子进程，清理 `runtime_cache/trt_engines/` 下 size=0 或没配套 `.profile` 的孤儿 engine 文件 |
| manifest 存在但 engine 文件丢失 | 启动时检测到，cache_status 返回 `failed`，自动清空目录并提示用户 |
| TRT EP 加载时报错（DLL 缺失） | 服务进程 catch，落日志 `trt provider load failed`，回退 CUDA EP 并通知 UI 把状态标为 `failed` |
| 用户在编译中关闭主程序 | 子进程作为孤儿进程，让它继续编完 → 主程序下次启动时通过 manifest 时间戳判断仍未完成 → 询问是否继续 |
| 编译完 verify 失败 | 删除整个缓存 + 清 manifest，报错让用户重试 |

### 5.8 PyInstaller 打包

新增需要在 `_internal/` 里出现：
- `nvinfer.dll` / `nvinfer_10.dll`（TRT 主库）
- `nvonnxparser.dll`
- `nvinfer_plugin.dll`（如果用到 plugin）
- `onnxruntime_providers_tensorrt.dll`
- `onnxruntime_providers_shared.dll`
- `cudnn*.dll` 已经在用（确认版本与 TRT 匹配）

打包后必做的 smoke test：在一台干净的 Windows 上启动 UI，确认勾选 TRT 后能跑完缓存。任意一个 DLL 缺失都会让 TRT EP 安静地不加载，回退 CUDA EP 用户不知道。

子进程 `ptserver-trt-warmup` 是独立 entry point，需要在 `build_exe.bat` 里加 PyInstaller 入口 spec。

### 5.9 子进程 stdout 协议

UI ↔ 子进程之间用文本行协议（简单稳定）：

```
STAGE:1:start:编译单眼引擎
STAGE:1:elapsed:01:23
STAGE:1:done:88
STAGE:2:start:编译 SBS 双眼引擎
STAGE:2:elapsed:03:01
STAGE:2:done:262
STAGE:3:start:固化运行时缓存
STAGE:3:done:8
DONE:total_seconds=452
```

或失败：
```
STAGE:1:start:编译单眼引擎
ERROR:GPU OOM
EXIT:1
```

字段：`STAGE:<n>:<event>[:value]`。UI 按 `STAGE:` 前缀解析其余日志原样落到 `runtime_cache/trt_engines/build.log` 供排查。

### 5.10 关于 driver 变更检测的实现

不需要在每一次 8K 帧处理时检测——成本不可接受。**只在服务进程启动时检测一次**：
- `main.py` 启动头部调用 `cache_status()`，如返回 `stale` 写一行日志 `trt cache stale due to driver_version change: 560.94 -> 561.09, falling back to CUDA`。
- UI 主进程在显示性能面板时同样调用一次，决定徽章颜色。

UI 不轮询。用户切到性能面板才查，开销可忽略。

## 6. 实现拆分（开发顺序）

1. **utils/trt_manifest.py + tests**
   纯函数，可独立单元测试。`collect_fingerprint` 在没有 GPU 的 CI 环境用 mock。

2. **ui/services/trt_warmup_process.py 子进程**
   独立 CLI，能脱离 UI 跑。先用现有 CUDA EP 跑通流程，再换 TRT EP 测真实编译。手测产物。

3. **ui/settings.py 增加 inference_backend 字段 + server_env() 拼接**
   纯字符串处理，单测。

4. **main.py / config.py 启动检测**
   读 manifest，决定是否注入 TRT provider。日志清楚。

5. **UI 性能面板新区块**
   先静态布局；再接 manifest 读取显示状态；最后接 QProcess 跑子进程 + 进度更新。

6. **PyInstaller spec 更新 + 干净机器 smoke**

7. **CHANGELOG + 用户文档**
   说明首次缓存耗时、缓存所在目录、driver 变更后行为。

## 7. 验收标准

- 全新机器（无 manifest）启动 UI：性能面板 TRT 单选可见但灰，缓存按钮显示"未缓存"。
- 点缓存 → 弹进度 → 5-15 分钟后 manifest 生成、状态变绿。
- 切到 TRT 后点"应用并重启服务"：服务重启，日志看到 `providers=[TensorrtExecutionProvider, ...]`。
- 切回 CUDA 后点"应用并重启服务"：服务重启，日志看到 `providers=[CUDAExecutionProvider, ...]`，TRT 不被加载，但缓存目录保留。
- 手动修改 manifest.json 的 driver_version 模拟升级 → 重启 UI → 状态自动变黄，提示需重缓。
- 编译中关闭 UI 主窗口 → 子进程能完成或被回收，不留半成品。
- 8K SBS 实测：开 TRT 后 FPS 比 CUDA 至少高 50%（在 RTX 20 系上，单 SBS 视频，ALPHA_STRIDE=1，input=1024，downsample=0.5）。

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| TRT 编译时间超预期，用户取消的概率高 | 阶段化进度 + 早期反馈（第一阶段几分钟内出现） |
| Driver 自动升级（Windows Update）让缓存失效，用户没看到提示就播放卡顿 | 启动时主动检测并在 UI 显示通知小红点 |
| 用户同时跑别的占 VRAM 进程导致 OOM | 子进程报错时给具体提示 + 重试按钮 |
| 缓存目录被杀毒软件清掉 | manifest 缺 engine 时进入 `failed`，UI 提示重缓 |
| PyInstaller 漏 DLL，TRT 静默不加载 | 启动日志检查 providers 与设置一致，不一致时弹一次 toast |
| 多用户/多 GPU 机器 | manifest 含 GPU UUID，换显卡自动失效；多 GPU 暂时只支持 0 号卡（与现有 server 行为一致） |

## 9. 后续可选优化（不纳入本期）

- 支持多个模型同时缓存（Phase 2 引入新模型时再做）。
- 缓存目录管理（清理过旧 engine、设上限大小）。
- 把"重启服务"做成不丢正在播流的"软切换"（需在 server 实现请求引流到新 session）。
- 一键导出/导入缓存（仅供同机器 reinstall 场景，跨机仍然无效）。

---

文档对应英文版：`summary_20260520_IMPL_PLAN_TENSORRT_CACHE_UI_EN.md`
