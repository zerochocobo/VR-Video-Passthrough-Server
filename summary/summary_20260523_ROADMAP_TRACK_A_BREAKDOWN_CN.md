# Track A 详细分解：首帧 ramp-up 收尾

日期：2026-05-23

> 父文档：`summary_20260522_ROADMAP_POST_TRT_WARMUP_CN.md`
> 前置完结：
> - `summary_20260522_TENSORRT_COLD_START_WARMUP_PATCH_STAGE1_CN.md`（singleton 复用 + static TRT 预加载）
> - `summary_20260522_TENSORRT_COLD_START_WARMUP_PATCH_STAGE2_CN.md`（`Matter.__init__(warmup_runs=…)` 形参隔离）
>
> 当前现状：首播 `alpha #1 ort_run = 18.6ms`（远低于 30ms 门槛），但 frame 30 fps 仅 ~32，frame 60 fps ~44；稳态 ~75-76fps。首帧 ramp-up 主要成本来源（已识别）：
> - `preprocess` 首次 ~39.3ms（怀疑 CuPy kernel JIT + GPU buffer 首次分配）
> - mux 首段 ~222ms（怀疑 NVENC encoder 首次创建 + bitstream header 写入）
>
> Track A 目标：frame 30 fps ≥ 40，frame 60 fps ≥ 55；冷启动后首播无明显视感 ramp-up。

---

## 0. Track A 概览

| 子步 | 内容 | 优先级 | 工作量 | 预期收益 | 风险 |
|---|---|---|---|---|---|
| A1 | 诊断首帧 preprocess=39.3ms 来源（细分计时） | P0 | 0.5 天 | 数据基线 | 无 |
| A2 | CuPy 预处理路径 warmup（composite_green / pack_uploaded JIT 预热） | P0 | 1 天 | preprocess 39.3ms → <5ms | 显存增量 ~100-200MB |
| A3 | NVENC encoder preflight（启动期暖通） | P1 | 1-1.5 天 | mux 首段 222ms → <50ms | preflight 几何/preset 选择 |
| A4 | `[WARMUP]` 结构化 JSON 日志 | P2 | 0.5 天 | 可观测性 | 下游解析改动 |
| A5 | UI 启动进度条扩展 | P2 | 1 天 | UX | 依赖 A3 + A4 |

依赖：A1 → A2 串行；A3 独立；A4 独立；A5 依赖 A4。

---

## 1. A1：诊断首帧 preprocess=39.3ms 来源

### 1.1 目标

在 server.log 中得到一份首帧（frame 0/1/2）的子段细分计时，回答以下问题：
- 39.3ms 中，CuPy kernel JIT、`upload_nv12_planes_gpu_scaled` H2D 拷贝、GPU buffer 首次分配各占多少？
- batch=2 SBS 路径与 batch=1 路径是否 JIT 行为不同？
- 第 1 帧之后哪几个子段仍偏慢（前 5 帧的 trace）？

A1 不引入功能改动，**只加诊断日志**，提供 A2 决策依据。

### 1.2 落地点

| 文件 | 函数 | 行号 | 改动 |
|---|---|---|---|
| `pipeline/matting.py` | `composite_green_gpu_p016_frame_to_gpu_nv12_profile` | ~2747 | 在首 N=5 次调用打印分段计时 |
| `pipeline/matting.py` | `composite_green_gpu_nv12_frame_to_gpu_nv12_profile` | ~2686 | 同上 |
| `pipeline/matting.py` | `_alpha_low_res_gpu_temporal` | (查找) | 首 N 次打印 preprocess 分段 |
| `pipeline/alpha_packer.py` | `pack_uploaded` | 713 | 首 N 次打印分段（仅在 alpha 路径触发） |

### 1.3 细分计时切面

每个 composite 入口预期插入的子段（不增加 hot-path 开销，仅前 N=5 帧打印）：

```
[DIAG][PREPROC] frame=<idx>
  upload_nv12_planes:    <ms>   # H2D + nv12→RGB CuPy kernel
  alpha_preproc:         <ms>   # _alpha_low_res_gpu_temporal 内 to_chw / normalize
  alpha_inference:       <ms>   # _run_rvm_iobinding_from_dev
  alpha_postproc:        <ms>   # resize/squeeze
  composite_kernel:      <ms>   # _composite_nv12_to_nv12_gpu_using_uploaded_frame
  total:                 <ms>
```

实现要点：
- 用一个实例计数器 `self._preproc_diag_count` 控制只前 N 帧打印
- 计时点之间需要 `stream.synchronize()` 才能拿到真实 GPU 时间（参考 `PASSTHROUGH_PYNV_SYNC_PROBE` 已有路径）
- 配置开关 `config.WARMUP_RAMPUP_DIAG_FRAMES`（默认 0，不打印）

### 1.4 验证

- 打开 `WARMUP_RAMPUP_DIAG_FRAMES=5`，跑一次冷启动 + 首播 60 帧
- 检查 server.log 中 `[DIAG][PREPROC]` 行：
  - 子段累加应 ≈ 已有 `preprocess_ms`
  - 找出 frame 0/1/2 与 frame 30/60 子段差异最大的 1-2 个项
- 把 diff 数据贴回本文档 1.5 节作为 A2 依据

### 1.5 数据落点

```
A1 代码已落地，实测待冷启动播放后填入。启用方式：
PT_WARMUP_RAMPUP_DIAG_FRAMES=5

Frame 0:  upload=__ alpha_preproc=__ alpha_inf=__ post=__ comp=__ total=__
Frame 1:  ...
Frame 5:  ...
Frame 30: ...
Frame 60: ...
```

### 1.6 风险

- 同步点引入会让计时本身偏慢（GPU stream 等待）；A1 仅作诊断，不放进稳态路径
- 多 worker 同时 warmup 时日志可能交错；按 `sid` 过滤即可

### 1.7 实施记录

2026-05-23：已增加 `PT_WARMUP_RAMPUP_DIAG_FRAMES`，默认 0。开启后 `pipeline/matting.py` 在 PyNv GPU NV12/P016 composite 入口打印前 N 次 `[DIAG][PREPROC]`，包含 upload、alpha_call、alpha_tail_sync、composite、total、mat_pre、mat_ort、mat_kernel。

---

## 2. A2：CuPy 预处理路径 warmup

### 2.1 目标

把 frame 0 的 preprocess 时延从 39.3ms 拉到 <5ms。手段：在 `utils/gpu_runtime_cache.py` 的 `_warmup_resident_matter_runtime` 中追加一段，用 zero frame 跑一次 composite / alpha 路径，让 CuPy kernel JIT + buffer 申请在启动期完成。

### 2.2 当前状态

`_warmup_resident_matter_runtime`（`utils/gpu_runtime_cache.py:467-527`）当前只做了：
1. `get_matter(warmup_runs=0)`
2. 对每个 `warmup_key.shape` 调 `_get_trt_static_session` 预加载 TRT engine
3. 对每个 shape 用 `cp.zeros` + `_run_rvm_iobinding_from_dev` 跑 N 次，验证 ORT IOBinding 路径
4. `matter.reset_state()`

**缺口**：从未跑过 `composite_green_*` / `_alpha_low_res_gpu_temporal` 上层入口，因此：
- `upload_nv12_planes_gpu_scaled` 的 nv12→RGB CuPy kernel 首次 JIT 未触发
- `_composite_nv12_to_nv12_gpu_using_uploaded_frame` 的 alpha 合成 kernel 未 JIT
- `pack_uploaded` 的 alpha 打包 kernel 未 JIT
- 各 staging buffer（`_g_chw`、`_g_frame`、`_g_out`、`Nv12OutputSlot`）未首次申请

### 2.3 修改方案

在 `_warmup_resident_matter_runtime` 现有 TRT 静态预加载之后、`reset_state()` 之前，追加以下伪代码：

```python
# === A2 patch: CuPy preprocess path warmup ===

# 构造代表性 nv12 zero frame（按 warmup_key 中最高分辨率 1 个 shape 即可）
biggest = max(warmup_key.shapes, key=lambda s: s[2] * s[3])
_, _, h_target, w_target = biggest  # 注意：shapes 给的是模型输入尺寸

# 选一个 8K 源几何作为代表（H,W 由 config 给）
src_h = config.WARMUP_COMPOSITE_SRC_H  # 默认 4096
src_w = config.WARMUP_COMPOSITE_SRC_W  # 默认 8192

# 1) 走 GREEN GPU NV12 路径 warmup
fake_nv12 = _build_zero_gpu_nv12_frame(src_h, src_w)  # 帮助函数（见 2.4）
try:
    out_slot = matter.acquire_nv12_output_slot(src_h, src_w)  # 强制 slot 首次分配
    matter.composite_green_gpu_nv12_frame_to_gpu_nv12_profile(
        fake_nv12, out_h=src_h, out_w=src_w, out_slot=out_slot,
    )
    if stream is not None:
        stream.synchronize()
except Exception:
    log.warning("[WARMUP] composite_green nv12 warmup failed", exc_info=True)

# 2) 走 GREEN GPU P016 路径 warmup（10bit 源）
fake_p016 = _build_zero_gpu_p016_frame(src_h, src_w)
try:
    matter.composite_green_gpu_p016_frame_to_gpu_nv12_profile(
        fake_p016, shift_bits=8, out_h=src_h, out_w=src_w, out_slot=out_slot,
    )
    if stream is not None:
        stream.synchronize()
except Exception:
    log.warning("[WARMUP] composite_green p016 warmup failed", exc_info=True)

# 3) 走 ALPHA pack 路径 warmup（SBS batch=2 typical）
try:
    from pipeline.alpha_packer import AlphaPacker
    packer = AlphaPacker(matter)
    # 用 batch=2 zero alpha，触发 packer kernel JIT
    fake_alpha = cp.zeros((2, 1, h_target, w_target), dtype=cp.float32)
    packer.pack_uploaded(fake_alpha, h_target, w_target, out_h=src_h, out_w=src_w)
    if stream is not None:
        stream.synchronize()
except Exception:
    log.warning("[WARMUP] alpha_packer warmup failed", exc_info=True)

matter.reset_state()
log.info("[WARMUP] composite/alpha CuPy kernels JIT warmed")
```

### 2.4 帮助函数

`_build_zero_gpu_nv12_frame(h, w)` / `_build_zero_gpu_p016_frame(h, w)` 需要构造一个 duck-typed 的 frame 对象（具有 `.height`/`.width`/`.y.as_cupy()`/`.uv.as_cupy()`）。

两个选择：
- **方案 A（推荐）**：在 `pipeline/matting.py` 暴露一个 `make_zero_gpu_frame(h, w, bit_depth=8 or 10)` 工厂，返回 dataclass，供 warmup 调用
- **方案 B**：在 `gpu_runtime_cache.py` 本地写个 `types.SimpleNamespace` + `cp.zeros`

倾向 A：单元可测；composite 路径未来若改 frame 接口，warmup 自动同步。

### 2.5 配置项

新增 `config.py`：

```python
# A2 warmup 控制
WARMUP_COMPOSITE_ENABLE = True
WARMUP_COMPOSITE_SRC_H = 4096
WARMUP_COMPOSITE_SRC_W = 8192
# 多个常见几何，循环 warmup（覆盖 4K / 8K）
WARMUP_COMPOSITE_GEOMETRIES = [(4096, 8192), (2048, 4096)]
```

`_warmup_resident_matter_runtime` 内循环 `WARMUP_COMPOSITE_GEOMETRIES`。

### 2.6 验证

- 完成 A1 后获得 frame 0 preprocess 子段基线
- 落地 A2 后再次冷启动 + 首播：
  - 期望 frame 0 `preprocess_ms < 5ms`
  - 期望 `[WARMUP]` 段出现新行：`composite/alpha CuPy kernels JIT warmed`
  - 期望 frame 30 fps 从 ~32 → ≥38
- 在 `tests/test_predict_warmup_state.py` 风格上增加一个测试：mock warmup，断言 composite 入口被调用 1 次

### 2.7 风险

- **显存**：8K NV12 帧约 48MB（Y）+ 24MB（UV）= 72MB；P016 加 2x = 144MB；slot + packer 中间张量再加 ~100MB。预计冷启动多 200-300MB 常驻。可接受。
- **正确性**：zero frame 走 alpha 模型会产生 alpha mask；但 reset_state 后影响清零。**确认**：A2 在所有 composite 调用结束后 `reset_state()`，与现有路径一致。
- **shape 不匹配**：实际播放几何可能不在 `WARMUP_COMPOSITE_GEOMETRIES` 内，CuPy 仍会 JIT 一次。可接受（kernel 已 JIT，只是 buffer 重新分配）。

### 2.8 完成判定

- `[WARMUP] composite/alpha CuPy kernels JIT warmed` 日志出现
- 冷启动首帧 `preprocess_ms < 5ms`
- frame 30 fps ≥ 38（短期目标）
- 16+ 测试通过；warmup 总时延增量 < 2s

### 2.9 实施记录

2026-05-23：已落地 A2 代码。

- 新增配置：
  - `PT_WARMUP_COMPOSITE_ENABLE`，默认 1；
  - `PT_WARMUP_COMPOSITE_GEOMETRIES`，默认 `4096x8192;2048x4096`。
- `pipeline/matting.py` 新增 `make_zero_gpu_frame(h, w, bit_depth=8|10)`，构造 duck-typed GPU NV12/P016 zero frame。
- `utils/gpu_runtime_cache.py` 的 resident warmup 在 static TRT 和 ORT IOBinding warmup 后，追加：
  - green NV12 composite warmup；
  - green P016 composite warmup；
  - alpha packer warmup；
  - 每个配置几何都会执行一轮，并在最后 `reset_state()`。
- 预热期间临时产生的 `Matter._call_count` / `_preproc_diag_count` 会恢复，避免污染首播 `alpha #1` 诊断。
- 已验证：
  - `.\.venv\Scripts\python.exe -m compileall config.py pipeline\matting.py utils\gpu_runtime_cache.py`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_config_defaults.py tests\test_predict_warmup_state.py tests\test_alpha_packer.py`

---

## 3. A3：NVENC encoder preflight

### 3.1 目标

首段 mux 222ms 来源主要是 NVENC encoder 首次 `CreateEncoder` + bitstream header 写入。当前已有 `PyNvStream.preflight(src, metadata)`（`pipeline/pynv_stream.py:370-423`），但它是 **per-request** 的，命中 `_preflight_ok` cache 才跳过；冷启动后第一次请求仍要付 222ms。

A3 把 preflight 抬到 **startup-time**，在 `_warmup_resident_matter_runtime` 完成后、HTTP 端点开放前，对若干常见几何做一次 NVENC create+release，让 NVENC SDK driver 完成进程级一次性初始化。

### 3.2 当前状态分析

`preflight` 函数（`pipeline/pynv_stream.py:370-423`）的核心动作：
1. 起一个临时 `PyNvSimpleDecoder(src, bit_depth)`
2. `nvc.CreateEncoder(width, height, "NV12", False, **kwargs)`
3. 调一次 `enc.EndEncode()`，`del enc`
4. 缓存 `(stat, geometry, fps, bitrate)` key 到 `_preflight_ok`（30 分钟 TTL）

**问题**：依赖具体 src 文件。启动期我们没有具体 src。

### 3.3 修改方案

在 `pipeline/pynv_stream.py` 增加新的 `class-method`：

```python
@staticmethod
def startup_preflight() -> None:
    """Pay NVENC SDK first-time init at startup, without a real src.

    Creates a small NVENC encoder, encodes nothing, releases. The goal is to
    force CUDA driver / NVENC SDK to load and JIT once per process, so the
    first real request does not eat 200+ms in CreateEncoder.
    """
    import PyNvVideoCodec as nvc

    for (w, h, fps_label, br) in config.NVENC_PREFLIGHT_GEOMETRIES:
        t0 = time.perf_counter()
        try:
            enc = nvc.CreateEncoder(
                int(w), int(h), "NV12", False,
                **_pynv_encoder_kwargs(bitrate=str(br), fps=fps_label),
            )
            try:
                end = getattr(enc, "EndEncode", None)
                if callable(end):
                    end()
            finally:
                del enc
                gc.collect()
            log.info(
                "[WARMUP] nvenc startup preflight ok %dx%d fps=%s br=%s elapsed_ms=%.1f",
                w, h, fps_label, br, (time.perf_counter() - t0) * 1000.0,
            )
        except Exception:
            log.warning(
                "[WARMUP] nvenc startup preflight failed %dx%d fps=%s br=%s",
                w, h, fps_label, br, exc_info=True,
            )
```

### 3.4 调用点

在 `main.py` startup 序列中，`warmup_gpu_runtime_cache(...)` 之后、`uvicorn` start 之前调用：

```python
# main.py 大概结构
warmup_gpu_runtime_cache(...)
set_startup_phase("nvenc_preflight", "warming NVENC encoder")
try:
    PyNvStream.startup_preflight()
except Exception:
    log.warning("nvenc startup preflight failed; first request will pay it lazily", exc_info=True)
set_startup_phase("listening", "ready")
```

### 3.5 配置

```python
# config.py
NVENC_PREFLIGHT_ENABLE = True
NVENC_PREFLIGHT_GEOMETRIES = [
    # (width, height, fps_label, bitrate_bps)
    (3840, 2160, "30.000000", "20000000"),   # 4K 30fps
    (7680, 4320, "30.000000", "60000000"),   # 8K 30fps
]
```

8192×4096 (SBS 8K) 也可加入，但每个 preflight 大约 100-300ms，列表别太长。

### 3.6 验证

- 冷启动观察：`[WARMUP] nvenc startup preflight ok ...` 行出现
- 第一次真实请求 `mux first segment` 时延 < 50ms（当前 ~222ms）
- 第二次起播 mux 时延无显著变化（NVENC 已暖）

### 3.7 风险

- **per-stream encoder 仍是新创建**：startup preflight 只解决进程级 SDK init；每个流仍要单独 CreateEncoder（实测应该已经 < 30ms）。如观察到仍有大跳延，需进一步研究 encoder pool。
- **几何/preset 覆盖率**：preflight 用 NV12 / P1 / 默认 GOP；如果真实请求用别的 preset，可能再花一次 init。当前所有路径都走 `_pynv_encoder_kwargs`，应一致。
- **错误处理**：preflight 失败不能阻断 startup，必须 best-effort。日志按 warning 记。

### 3.8 完成判定

- `[WARMUP] nvenc startup preflight` 日志 N 行（N = preflight 几何数）
- 冷启动后首次请求首段 mux < 50ms
- 16+ 测试通过

### 3.9 实施记录

2026-05-23：已落地 A3 代码。

- 新增配置：
  - `PT_NVENC_PREFLIGHT_ENABLE`，默认 1；
  - `PT_NVENC_PREFLIGHT_GEOMETRIES`，默认 `8192x4096@59.940060:50000000;4096x2048@59.940060:25000000`。
- `pipeline/pynv_stream.py` 新增 `PyNvPassthroughStream.startup_preflight()`：
  - 使用现有 `_pynv_encoder_kwargs()`，确保 preset/tuning/rc/gop/bf 与真实流一致；
  - 对配置几何调用 `PyNvVideoCodec.CreateEncoder(..., "NV12", False, ...)`；
  - 调 `EndEncode()` 后释放 encoder；
  - 成功日志：`[WARMUP] nvenc startup preflight ok ...`；
  - 失败只 warning，不阻断 server 启动。
- `main.py` 在 GPU warmup 后、HTTP/SSDP 启动前调用 `startup_preflight()`。
- 已验证：
  - `.\.venv\Scripts\python.exe -m compileall config.py main.py pipeline\pynv_stream.py tests\test_pynv_startup_preflight.py tests\test_config_defaults.py`
  - `.\.venv\Scripts\python.exe -m pytest tests\test_config_defaults.py tests\test_predict_warmup_state.py tests\test_alpha_packer.py tests\test_vr_naming.py tests\test_pynv_startup_preflight.py tests\test_main_args.py`

实测待下一轮 UI 冷启动后填入：

```
NVENC preflight 8192x4096 elapsed=__
NVENC preflight 4096x2048 elapsed=__
first request mux stage_max_ms=__
first stdout chunk delta=__
```

---

## 4. A4：`[WARMUP]` 结构化 JSON 日志

### 4.1 目标

当前 `[WARMUP]` 日志为 freeform 文本，UI / 监控端解析困难。把所有 `[WARMUP]` 行改成 JSON 结构化（保留 `[WARMUP]` 前缀方便 grep），同时维持 backward-compat（同一行可被人类阅读）。

### 4.2 改动落点

| 文件 | 行 | 改造 |
|---|---|---|
| `utils/gpu_runtime_cache.py` | 472, 499, 526 等 `[WARMUP]` 行 | 改 JSON |
| `pipeline/pynv_stream.py` | A3 新增 `[WARMUP] nvenc ...` 行 | 一开始就写 JSON |
| `pipeline/matting.py` | A2 新增的 `[WARMUP] composite ...` 行 | 一开始就写 JSON |

### 4.3 日志格式

```
[WARMUP] {"phase":"static_trt_preload","batch":1,"shape":[1024,1024],"loaded":true,"elapsed_ms":234.5}
[WARMUP] {"phase":"matter_singleton","id":140234567,"static_trt_available":true,"providers":["TensorrtExecutionProvider","CUDAExecutionProvider"]}
[WARMUP] {"phase":"composite_jit","kind":"green_p016","geometry":[4096,8192],"elapsed_ms":68.2}
[WARMUP] {"phase":"nvenc_preflight","geometry":[7680,4320],"fps":30.0,"bitrate":60000000,"elapsed_ms":312.4}
[WARMUP] {"phase":"reset","step":"after_warmup"}
```

### 4.4 实现要点

引入一个小帮助函数 `utils/logger.py`：

```python
def warmup_event(log, **fields):
    log.info("[WARMUP] %s", json.dumps(fields, separators=(",", ":")))
```

所有 `[WARMUP]` 调用统一通过 `warmup_event(log, phase=..., elapsed_ms=...)`。

### 4.5 兼容性

- 现有 `startup_status_poller` 不解析 `[WARMUP]` 行，只读 `/status` JSON；不受影响
- 如有外部 log 抓取（ELK / Loki / grep 脚本），可继续 grep `[WARMUP]`；JSON 化反而易解析

### 4.6 完成判定

- 所有 `[WARMUP]` 行 JSON 化
- 一个简单 test：mock log handler，断言 phase 关键字段都在
- 16+ 现有测试通过

### 4.7 实施记录

2026-05-23：已落地 A4。

- `utils/logger.py` 新增 `warmup_event(log, **fields)`，统一输出：
  - `[WARMUP] {"phase":"...","...":...}`
  - 保留 `[WARMUP]` grep 前缀。
  - `json.dumps(..., default=str)`，避免非 JSON 原生对象导致 warmup 日志影响启动。
- `utils/gpu_runtime_cache.py` 的 warmup 事件已改为结构化 JSON：
  - `matter_singleton`
  - `static_trt_preload`
  - `composite_jit`
  - `reset_state`
- `pipeline/pynv_stream.py` 的 NVENC startup preflight 已改为结构化 JSON：
  - `nvenc_preflight`
  - `status=ok|failed|disabled`
- 验证：
  - `rg "\[WARMUP\]" utils pipeline main.py` 只剩 `utils/logger.py` 的统一 helper。
  - `tests/test_predict_warmup_state.py` 增加 `warmup_event` JSON payload 测试。

---

## 5. A5：UI 启动进度条扩展

### 5.1 目标

`startup_status` 的 `step`/`step_index`/`step_total` 当前已支持，但实际 warmup 流程未细分 phase。Track A 完成后，启动序列应公开如下进度：

```
phase=warming  step="static_trt_preload"           step_index=1/N  progress=...
phase=warming  step="composite_jit"                step_index=2/N
phase=warming  step="nvenc_preflight"              step_index=3/N
phase=warming  step="reset_state"                  step_index=4/N
phase=listening
```

UI 启动 overlay (`ui/widgets/startup_overlay.py`) 据此显示进度条 + 当前 step 文案。

### 5.2 改动落点

| 文件 | 改造 |
|---|---|
| `utils/gpu_runtime_cache.py:_warmup_resident_matter_runtime` | 每个 phase 进入前调 `set_startup_phase("warming", message=..., step=..., step_index=..., step_total=N)` |
| `pipeline/pynv_stream.py:startup_preflight` | 同上 |
| `main.py` | 编排 `step_total` 与各 phase 顺序；最后调 `set_startup_phase("listening", ...)` |
| `ui/widgets/startup_overlay.py` | 新增 step 文案映射表（中文友好），渲染进度条 |
| `ui/services/startup_status_poller.py` | 透传 step/progress 给 UI |

### 5.3 step 文案映射（建议）

| step key | UI 中文 |
|---|---|
| `static_trt_preload` | 加载 TensorRT 引擎 |
| `composite_jit` | 预热绿幕合成内核 |
| `nvenc_preflight` | 预热视频编码器 |
| `reset_state` | 收尾 |
| `listening` | 就绪 |

### 5.4 完成判定

- UI 启动期看到 N 段进度条 + 中文文案，每段独立推进
- 总耗时与 server.log 中 `elapsed_ms` 合计一致（误差 < 200ms）
- 既有 `is_known_slow` cold-start UX 流程（sm_120 等）不破坏

### 5.5 依赖

- 必须先完成 A3（新 phase 存在）+ A4（structured 日志，便于诊断）
- A1/A2 无需绑定 A5；但同步落地能保证一次完整 cold-start 验收

### 5.6 实施记录

2026-05-23：已落地 A5。

- 后端 `/status` step 链已细分：
  - `predict`
  - `matter_singleton`
  - `static_trt_preload`
  - `ort_iobinding_runs`
  - `composite_jit`
  - `reset_state`
  - `nvenc_preflight`
  - `warmed`
- `main.py` 使用动态 `step_total`：
  - 默认 GPU warmup 5 步。
  - 若启用 PyNv + NVENC preflight，则总步数为 6。
- `utils/gpu_runtime_cache.py` 在 resident warmup 内部按阶段调用 `set_startup_phase(...)`，UI 不再只看到一个长时间不变的 `ort_session_and_runs`。
- `pipeline/pynv_stream.py:startup_preflight()` 进入时发布 `nvenc_preflight` step。
- `ui/widgets/startup_overlay.py` 新增 step label，显示类似：
  - `3/6  预热 GPU 推理`
  - `6/6  预热视频编码器`
- `ui/translations/{zh_CN,en_US,ja_JP}.json` 增加各 step 文案。
- 补齐网络启动/终态：
  - `firewall`
  - `ssdp`
  - `http_starting`
  - `listening`
- `main.py` 在 FastAPI startup event 中发布 `phase=listening`，UI poller 不再只依赖 stdout fallback 判断 uvicorn 已经可用。
- 验证：
  - 使用 UTF-8 BOM 兼容方式校验三套 JSON 翻译文件。
  - `tests/test_predict_warmup_state.py` 增加 step 覆盖测试。

---

## 6. 推荐落地顺序

| 序号 | 步骤 | 累计工作量 | 验收里程碑 |
|---|---|---|---|
| 1 | A1 | 0.5 天 | 拿到 frame 0/30/60 子段数据 |
| 2 | A2 | 1.5 天 | frame 0 preprocess < 5ms；frame 30 fps ≥ 38 |
| 3 | A3 | 3 天 | mux 首段 < 50ms |
| 4 | A4 | 3.5 天 | `[WARMUP]` JSON 化 |
| 5 | A5 | 4.5 天 | UI 进度条显示完整 phase 链 |

每一步建议独立 commit + 一次冷启动观察记录。

---

## 7. 与父文档的对接

完成 Track A 后，回写 `summary_20260522_ROADMAP_POST_TRT_WARMUP_CN.md` 第 8 节「完成判定」表中 Track A 行：

```
| A | frame 30 fps ≥ 40，frame 60 fps ≥ 55，首播无明显 ramp-up 视感 | ✅ 已完成 (YYYY-MM-DD)|
```

并在 HANDOVER_20260522.md 追加一段 Track A 收尾摘要。

---

## 8. 不在本子计划中（明确排除）

- CUDA Graph 评估（属 Track B1）
- alpha 通道真透明输出（属 Track C）
- 长时间压测（属 Track D2）
- ConnectionManager 完整化（属 Track E4）
- `pipeline/matting.py` 拆模块（E5，暂缓）

---

## 9. 备注：阶段 1+2 后已观察到的真实数据（基线）

引自 `prompt/HANDOVER_20260522.md` 末尾人工验证：

| 节拍 | fps |
|---|---|
| frame 30 | 32.12 |
| frame 60 | 44.17 |
| frame 120 | 56.61 |
| frame 600 | 72.65 |
| 稳态 | ~75-76 |

首帧 `alpha #1 ort_run = 18.6ms`（< 30ms 门槛）。

Track A 完成后目标重测节拍：

| 节拍 | 目标 fps |
|---|---|
| frame 30 | ≥ 40 |
| frame 60 | ≥ 55 |
| frame 120 | ≥ 65 |
| 稳态 | ≥ 75 |

---

文档维护：每完成一个子步骤，在本文档对应章节末尾追加「实测：YYYY-MM-DD …」一行，并把 frame N 实测 fps 填到 1.5 节表格。

---

## 10. A6/A7 首包延迟收尾

2026-05-23 Track A 已完结，终档见 `summary_20260523_TRACK_A_FINAL_ARCHIVE_CN.md`。A6/A7 初版之后又经过 A8.1/A8.P1/A8.P2.A 修正，以下为最终状态：

- A6：`PT_MUX_LATENCY_DIAG=1` 默认开启，输出 `[DIAG][MUX] first_chunk_breakdown`，拆分 mux spawn、首写、首 stderr、reader 首包。
- A6.5：`PT_FORCE_AUDIO_OFF=1` 可用于 video-only 单段 mux 对照。
- A7.1 初版的 raw HEVC `probesize=32/analyzeduration=0` 已证伪，会导致严格播放器 audio-only；最终 raw HEVC 锁定 `PT_MUX_RAW_VIDEO_PROBESIZE=1000000`、`PT_MUX_RAW_VIDEO_ANALYZEDURATION=1000000`。
- A8.P2.A.1：pipe_ts final mux intermediate TS stdin 锁定 `PT_MUX_INTERMEDIATE_TS_PROBESIZE=16384`、`PT_MUX_INTERMEDIATE_TS_ANALYZEDURATION=0`；8192 无收益并接近回归边界，不继续下探 4096。
- A8.2：`PT_PASSTHROUGH_AUDIO_MPEGTS_INTERLEAVE_DELTA=0` 单点无效，已恢复默认 `500000000`。
- A7.3：`PT_MUX_NOBUFFER_ENABLE=1` 默认将 `-fflags` 从 `+genpts` 扩展为 `+genpts+nobuffer+flush_packets`。
- A7.2：`PT_FMP4_FRAG_DURATION_US=100000` 替代 fMP4 硬编码 `250000`。

验证：

- nPlayer / Quest3 DeoVR / SkyBoxVR 三播放器均保持 video 模式，无 `Could not find codec parameters`、audio-only、PPS、Broken pipe、reader waiting 回归。
- 当前锁定 first chunk：nPlayer 1978.4ms，DeoVR 1939.1ms，SkyBoxVR 2028.9ms。
- `uv run python -m pytest tests\test_pynv_mux_latency.py tests\test_config_defaults.py tests\test_pynv_startup_preflight.py` -> 18 passed。

最终结论：

- Track A 完结；继续压低首块需新项目取消 ffmpeg 双段 mux，不在本 Track 范围内。
