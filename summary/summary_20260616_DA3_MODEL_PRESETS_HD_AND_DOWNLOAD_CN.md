# DA3 模型档位(普通/高清)+ 一键下载(2026-06-16)

## 目标(用户需求)

1. Small 也导出 1036 高清版,测速。
2. 删掉 Base 700;实时默认仍 Base518(留富余)。模型选择新增 **Base 高清(1036)** /
   **Small 高清(1036)**,给好显卡用户。默认 **Base 普通(518)**,Small 改名 **Small 普通(518)**。
   离线 UI 和实时配置 UI 都要改。
3. TRT 缓存按这几个档位分开。
4. 模型不存在时提供一键下载,来源 `https://huggingface.co/zerochocobo/DepthAnything3_ONNX`,
   兼容 hf-mirror。

## 导出与测速

`examples/da3_to_onnx.py` 已支持按尺寸命名(commit `d6929dc`)。新增:
- `da3_base_1036.onnx`、`da3_small_1036.onnx`(均 `--fold-preprocess`,torch↔ORT 误差 ~1e-4)。
- 删除 `da3_base_700.onnx`。
- 模型 gitignore(脚本/下载可得)。

测速(TRT 稳态,含 CPU letterbox 预处理,test_4k2d 1080p):
| 档位 | depth ms/帧 |
|---|---|
| Small 518 | 8.0 |
| Base 518 | 12.1 |
| Small 1036 | 25.9 |
| Base 1036 | 65.0 |

→ **Small 1036 ≈ 26ms,是 Base 1036 的 ~2.5× 快**,高清里性价比更高;Base 1036 最准但最慢。

## 模型档位(presets)

`offline/da3_depth.py` 引入统一的 preset 注册表(key = 用户选择值):
```
base     -> da3_base.onnx        518
small    -> da3_small.onnx       518
base_hd  -> da3_base_1036.onnx   1036
small_hd -> da3_small_1036.onnx  1036
```
- `Da3DepthEngine(variant=<preset>)` 自动解析 onnx 文件 + 输入尺寸(`engine.size`)+ 缓存目录。
- `config.TWO_DVR_MODEL` / CLI `--model` / 两个 UI 下拉都接受这 4 个 key。默认 `base`。
- **GPU letterbox 尺寸改用 `engine.size`**(原硬编 518):`offline/two_dvr_pynv.py`
  `convert_clip_pynv` 和 `pipeline/pynv_stream.py` `_worker_loop_two_dvr` 的 canvas/grid/
  kernel 入参都改成 `da3_size = engine.size`——否则高清模型仍只吃 518。

## TRT 缓存按档位分开(part 3)

`_trt_cache_dir(model)` / `trt_engine_cached(model)` 改为按 **preset key** 建子目录:
`runtime_cache/da3_trt/{base,small,base_hd,small_hd}`。不同输入尺寸的引擎不再共目录。
`build-trt --model` 增加 4 个档位 + `both`(=全部 4 个);`_ensure_trt_cache` 自动按所选档位
首次构建(base_hd 实测首建 ~78s,之后命中缓存)。

## 一键下载(part 4)

`offline/da3_depth.py`:
- `model_download_url(model)` = `${HF_ENDPOINT:-https://huggingface.co}/zerochocobo/
  DepthAnything3_ONNX/resolve/main/<file>`,**HF_ENDPOINT 即 hf-mirror 兼容**
  (设 `HF_ENDPOINT=https://hf-mirror.com`)。
- `download_model()`(流式下载 `.part` → 原子改名,带进度回调)、`model_available()`、
  `ensure_model_available()`(缺失则下载,失败返回 False 不抛)。
- **自动触发**:离线 CLI(single/batch/build-trt 前)与实时 worker 创建引擎前都先
  `ensure_model_available(model)`,缺失即下载并把进度打到日志(UI 日志可见)。
- CLI 新增 `python -m offline.two_dvr download --model {小档位|both}`。
- 实测:1 字节 range 请求 HF,`small`/`base_hd` 均 206,大小与本地一致(下载链路+重定向 OK)。

## UI(part 2)

- 离线 `ui/pages/two_dvr_page.py`:`_model_combo` 加 small_hd/base_hd,retranslate 设 4 项文案。
- 实时 `ui/pages/home_page.py` `TwoDvrLiveSettingsDialog`:model 下拉加 small_hd/base_hd,
  findData 放宽到 4 档。
- 文案(zh/en/ja):`model_small`→"Small 普通(518)"、`model_base`→"Base 普通(518)",
  新增 `model_small_hd`/`model_base_hd`="… 高清(1036)"。默认仍 Base 普通。

## 验证

- `py_compile` 全过;翻译 JSON 解析、新键存在;preset 解析/URL/ensure 正常;
  `tests/test_offline_outputs`+`test_two_dvr_hybrid_hole_fill` 11 passed。
- 离线 `--model base_hd` 端到端:TRT 引擎建到 `da3_trt/base_hd`(独立目录),`model=base_hd`、
  depth=TensorRT、~16fps(1036 离线)。

## 备注 / 后续

- 实时选 1036 大概率掉出 60fps,默认保持 base518(用户要求留富余)。
- `ui/widgets/da3_trt_dialog.py` 用的是自带的 `da3_trt_cache_dir`(仅 small/base 手动构建 UI),
  未改;高清档走运行时自动构建。可后续把高清也纳入该手动构建/预热 UI。
- 模型仍 gitignore;`models/DA3/get_DA3.txt` 资源链接由另一会话维护。
