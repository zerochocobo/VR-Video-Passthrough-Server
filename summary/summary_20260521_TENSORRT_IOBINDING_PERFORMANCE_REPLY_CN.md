# TensorRT 代码审核与 IOBinding 性能/Hang 问题排查方案

- 日期：2026-05-21
- 作用：对 commit `e21ed9f (tensorrt)` 做代码审核，并针对 `summary_20260521_TENSORRT_IOBINDING_PERFORMANCE_EN.md` 中遗留的 hang 与 4K alpha 性能不及 CUDA + IOBinding 的问题，给出**可执行的诊断步骤、修复方案与优先级**，供后续开发人员直接照做。
- 上游文档：
  - 实施计划：`summary/summary_20260520_IMPL_PLAN_TENSORRT_CACHE_UI_CN.md`
  - 问题报告：`summary/summary_20260521_TENSORRT_IOBINDING_PERFORMANCE_EN.md`

---

## 1. 代码审核结论（commit e21ed9f）

### 1.1 总评
- 完整实现了实施计划：manifest、子进程 warmup、UI 配置面板、provider 链注入、DLL 路径处理、静默回退到 CUDA。31 个测试通过。
- 计划文档中"FP16 1.5-2× 增益"的**性能承诺未达成**。当前 TensorRT 路径在 4K alpha 上比 CUDA + IOBinding 慢，必须先解决 hang 才能恢复 IOBinding 路径。

### 1.2 各模块要点

| 文件 | 关键实现 | 评价 |
|---|---|---|
| `utils/trt_manifest.py` | 指纹包含 GPU UUID/驱动/CUDA/TRT/ORT/model sha256/输入尺寸/downsample/FP16/CUDA graph；`cache_status()` 额外校验 shape_inferred ONNX 与 ≥1 MiB engine 真实存在 | 比计划更严格，OK |
| `ui/services/trt_warmup_process.py` | 额外做了 `_make_rvm_state_dims_unique`，把 `r1i..r4i` 与 `src` 共用的 `height/width` symbolic dim 重命名，产出 `*_shape_inferred.onnx`，避免 TRT profile 冲突；warmup 双重校验 active provider 含 `TensorrtExecutionProvider`，防 ORT 静默回退 | 关键修复，OK |
| `pipeline/matting.py:73-78` `_should_enable_rvm_iobinding` | TRT 在 active 列表时强制关 IOBinding | 当前的稳定 workaround，**也是性能瓶颈来源** |
| `pipeline/matting.py:889-893` | trt_fp16_enable 等改成 `"True"/"False"` | 与 ORT 文档约定 `"1"/"0"` 略不一致，但 ORT 接受两者，无功能影响 |
| `main.py:_validate_tensorrt_provider` | 启动时一次性 validate，stale/missing/failed 静默回退到 CUDA，并把 `config.MODEL_PATH` 切到 `*_shape_inferred.onnx` | 行为正确，回退路径清晰 |
| `utils/runtime_dll_paths.py` | dev/frozen 两态统一处理 TRT/CUDA/cuDNN DLL 路径 | OK |
| `config.py` | `ONNX_TRT_CUDA_GRAPH_ENABLE` 默认从 `1` 改成 `0` | 与计划文档示例不一致，但实际是规避 CUDA Graph + TRT 兼容问题，合理 |

### 1.3 可以删的"小尾巴"
- 无需立刻处理，但下一次清理时建议：
  - `trt_options` 字符串值统一回 `"1"/"0"`，与 ORT 官方示例对齐。
  - `trt_warmup_process._run_shape` 在 `batch>1` 但 `_supports_batch2=False` 时直接 return，没有发 `STAGE:2:skip`，UI 端预估时长会偏大，可后续优化。

---

## 2. Hang 问题根因假设（按可能性排序）

`pipeline/matting.py:1801-1849 _run_rvm_iobinding_from_dev` 在 `TensorrtExecutionProvider` 激活时调用 `sess.run_with_iobinding(binding)` 永不返回。下列是 4 个候选根因。

### 假设 A：输出 OrtValue 形状与 TRT 实际输出不匹配（最可能）

- `_rvm_output_shape_for` 用 `max(1, int(round(h * RVM_DOWNSAMPLE_RATIO / scale)))` 算 r1o..r4o 形状。
- RVM 内部 Resize 走的可能是 `floor` 或 `ceil`，结果与 `round` 可能差 1 像素。
- CUDA EP 会报错；TRT EP 在跨 TRT/CUDA partition 边界上，可能在 stream sync 上死等。

### 假设 B：recurrent OrtValue 内存所有权

- `self._rvm_rec_ort = rec_outs[:4]`：把 `binding.get_outputs()` 返回的 OrtValue 直接当作下一帧的输入。
- CUDA EP 下这些 OrtValue 持有独立内存，安全。
- TRT EP 下它们可能是 TRT engine 内部 workspace 视图，下一次 `run_with_iobinding` 会复用同一块内存，造成读写冲突 → 死锁或脏数据。

### 假设 C：TRT EP 与 CUDA EP 跨 stream 同步缺失

- `pipeline/matting.py:899-903` CUDA EP 用 `user_compute_stream=_CUDA_STREAM.ptr`。
- TRT EP 没有传 `trt_user_compute_stream`（ORT ≥ 1.18 才支持）。
- TRT subgraph 在自己 enqueue 的 stream 上 launch；CUDA EP 的 fallback 算子在用户 stream 上 launch。
- 跨 stream OrtValue 没 event 同步 → 死锁。

### 假设 D：第一帧 `(B,1,1,1)` recurrent state 触发 profile min 边界

- `_reset_rvm_rec_if_needed` 第一帧 state 是 `(batch,1,1,1)`，第二帧后才是真实尺寸。
- TRT 编译时按真实尺寸做 profile，min/opt/max 没覆盖 `(1,1,1,1)`，第一次 run 失败或 hang。

---

## 3. 诊断步骤（顺序执行）

> 目标：1 个工作日内定位 hang 根因。

### Step 1：原生 stack dump
- 复现 hang，在 server 进程上用 `py-spy dump --pid <pid> --native` 抓主线程 + native frame。
- 期望看到的关键帧：
  - `cuStreamSynchronize` / `cudaEventSynchronize` → 强烈指向假设 C 或 B。
  - `nvinfer1::IExecutionContext::enqueueV3` 内部 → 假设 A 或 D。
  - `OrtValue::Reshape` / shape mismatch → 假设 A。

### Step 2：让 ORT 自分配输出
- 临时改 `_run_rvm_iobinding_from_dev` 所有 `bind_ortvalue_output(...)` 为 `binding.bind_output(name, "cuda", 0)`，去掉预分配。
- 跑同一个 8K green，看是否还 hang。
  - 不 hang → 假设 A 成立，定位到 shape 不匹配。
  - 仍 hang → 排除假设 A。

### Step 3：强拷 recurrent state
- 在 `rec_outs = self._rvm_rec_outputs_from(...)` 之后：
  ```python
  rec_outs = [
      ort.OrtValue.ortvalue_from_numpy(
          cp.asnumpy(self._ortvalue_to_cupy(v)), "cuda", 0
      )
      for v in rec_outs[:4]
  ]
  ```
- 跑同一个 8K green，看是否还 hang。
  - 不 hang → 假设 B 成立。
  - 仍 hang → 排除假设 B。

### Step 4：关闭 CUDA 共享 stream
- 临时 `PT_CUDA_SHARED_STREAM=0` 启动。
- 看是否仍 hang。
  - 不 hang → 假设 C 成立。
  - 仍 hang → 升级 ORT 到 ≥ 1.18，在 `trt_options` 加 `"trt_user_compute_stream": str(int(_CUDA_STREAM.ptr))` 再试。

### Step 5：消除第一帧 (1,1,1,1) state
- 在 `Matter.__init__` warmup 阶段强制先跑一次真实尺寸：
  - `_reset_rvm_rec_ort_if_needed(batch=1, h=1024, w=1024)` → 立刻把 `_rvm_rec_ort` 各 OrtValue 重新分配到对应真实 H/W（按 RVM downsample 计算的 state H/W）并填 0。
- 如果只有第一次推理 hang、后续不 hang → 假设 D 成立。

---

## 4. 修复方案（按优先级与改动面排序）

### 方案 1（首选）：手工管理 device OrtValue，绕过 IOBinding 但保持 GPU 驻留
- **目标**：把 TRT + `sess.run_with_ort_values` 路径性能压到 ≤ 20 ms/帧，至少与 CUDA + IOBinding 持平。
- **改动**：
  - 用 `OrtValue.ortvalue_from_shape_and_type(shape, dtype, "cuda", 0)` 预分配 `src`、`r1i..r4i`、`downsample_ratio`，与输出 alpha 共 6 个 device 缓冲。
  - 用 `update_inplace(np_array)` 写入 host 端数据（虽然名字带 inplace，但内部会做 H2D 异步拷贝）。
  - state 复用：`self._rvm_rec_ort = list(sess.run_with_ort_values(self.output_names, feed))[2:6]`，配合每帧自分配解决假设 B 的内存重用问题。
  - 调用 `sess.run_with_ort_values(self.output_names, feed_ort_values)` 替代 `sess.run_with_iobinding(binding)`。CUDA EP 和 TRT EP 都支持。
- **预期收益**：消除 H2D/D2H 4-6 ms 损耗，回到 `ort_run ≈ 18-20 ms`。
- **风险**：低。这是 ORT 官方推荐的"device-resident inputs without IOBinding"模式。

### 方案 2：根据 Step 2-5 结果对症修
- Step 2 命中 → 把 `_rvm_output_shape_for` 改成读取 model 的 `output_metas[i].shape` 推导，遇到 `dim_param` 走 ORT 自分配。
- Step 3 命中 → 在 `_run_rvm_iobinding_from_dev` 内对 rec_outs 做 `cupy.ascontiguousarray + ortvalue_from_numpy` 拷贝。
- Step 4 命中 → 升 ORT 1.18+，传 `trt_user_compute_stream`；或纯 TRT EP 关 CUDA EP 共享 stream。
- Step 5 命中 → warmup 阶段强制初始化真实 state shape。

### 方案 3（中期）：直接用 TensorRT Python API，绕开 ORT TRT EP
- **场景**：方案 1+2 落地后，若 TRT 路径仍未达到 1.5-2× 增益（计划文档承诺），走此路径。
- **改动量**：中等，但目标明确。
- **思路**：
  1. warmup 阶段保留 ORT 路径生成 engine 缓存（沿用现有 `runtime_cache/trt_engines/*.engine`）。
  2. runtime 用 `tensorrt.Runtime.deserialize_cuda_engine` 直接加载 engine。
  3. 用 `IExecutionContext` + `execute_async_v3` 调度，binding 直接绑 CuPy 指针。
  4. 与 PyNv 共享 CUDA stream，零拷贝。
- **预期收益**：12-14 ms/帧（FP16，1024 输入，state 全 device 驻留），真正实现计划承诺。
- **风险**：需要维护 engine 兼容矩阵，但 manifest 已经做了指纹，复用即可。

### 方案 4（长期）：导出 static-shape RVM ONNX
- 为 `(1,3,1024,1024)` 与 `(2,3,1024,1024)` 各导出一份 state 尺寸全 freeze 为常量的 ONNX。
- TRT 编 static engine 没有 profile 烦恼，IOBinding 的死锁面大幅缩小。
- **改动位置**：模型导出端，不在 server 端。
- **风险**：需要离线脚本与一次性人工 validate。

---

## 5. 推荐时序

| 顺序 | 任务 | 预估工作量 | 触发条件 |
|---|---|---|---|
| ① | 第 3 节 Step 1 + Step 2 排查 | 半天 | 立刻 |
| ② | 方案 1（device-resident OrtValue + `run_with_ort_values`） | 1-2 天 | 不论 Step 1-2 结果都做，目标先打平 CUDA IOBinding |
| ③ | 根据 Step 3-5 结果落地方案 2 对症修复 | 0.5-1 天 | Step 命中 |
| ④ | 完成 ① ② ③ 后若 TRT 仍未达 1.5× | 方案 3 直接 TRT API | 3-5 天 | 必要时 |
| ⑤ | 方案 4 模型重新导出 | 1 天 + 验证 | 与 ④ 并行可选 |

---

## 6. 兜底建议（与原文档一致）
- **不要把 TensorRT 设为默认**。`ui/settings.py` 的 `inference_backend` 默认保持 `"cuda"`。
- UI 上保留"实验性"标签，直到方案 1 或方案 3 让 TRT 路径实测优于 CUDA + IOBinding 至少 30%。
- `_should_enable_rvm_iobinding` 在 TRT 活跃时关 IOBinding 的 workaround 保留，直到方案 1 落地后再改回"TRT + device OrtValue"。

---

## 7. 涉及文件清单
- `pipeline/matting.py`
  - `_should_enable_rvm_iobinding` (line 73-78)
  - `_provider_config` (line 886-913)
  - `_run_rvm_iobinding_from_dev` (line 1801-1849)
  - `_reset_rvm_rec_ort_if_needed` (line 1723-1740)
- `utils/trt_manifest.py`
- `utils/runtime_dll_paths.py`
- `ui/services/trt_warmup_process.py`
- `main.py:_validate_tensorrt_provider`
- `config.py:ONNX_TRT_*`

---

## 8. 验证清单（修复后必跑）
- [ ] 8K green，CUDA + IOBinding 基线：`ort_run` ≤ 20 ms。
- [ ] 4K alpha，TRT + 新 device OrtValue 路径：`ort_run` ≤ 20 ms，FPS ≥ 30。
- [ ] 4K alpha，TRT + 新路径：单 stream 连续播放 30 分钟无 hang、无 504。
- [ ] 切换 backend `cuda ↔ tensorrt` 重启服务，providers 日志正确。
- [ ] 手改 `manifest.json.fingerprint.driver_version` 模拟驱动升级 → 启动日志显示 `trt cache invalid (driver_version: ...)`，自动回退 CUDA。
- [ ] PyInstaller 干净机器 smoke：UI → 配置 TensorRT → 缓存完成 → 切到 TRT → 播 4K alpha 正常。
- [ ] 现有 31 个测试继续通过。
