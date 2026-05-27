# YOLO26m 替换 YOLO-World 开发计划（EfficientSAM 保留）

**日期**：2026-05-27
**作者**：架构调研 + 用户反馈（Claude 协同）
**目标读者**：执行开发的工程师
**关联**：
- 用户痛点：YOLOWorld+EfficientSAM 离线前置（recognition = `yoloworld_efficientsam`）出现**画面闪烁**、**单眼丢人**、**抠图边缘外扩**
- MatAnyone2 边缘问题已在 `summary_20260527_MATANYONE2_DRAG_*` 系列修复（segment=240、left/right buffer fix、Phase 1 控制），剩余的"边缘大/丢人/闪烁"基本可归到前置 mask 质量
- 新模型：YOLO26m（DETR 风格 head，COCO 80 类）已放置在 `models/yolo26m/yolo26m_model.onnx` 及 `yolo26m_model_fp16.onnx`
- EfficientSAM 已迁移到独立目录 `models/efficientsam/efficientsam_s.onnx`

---

## 0. TL;DR — 给开发者的 30 秒摘要

- **替换检测器**：YOLO-World v8 → **YOLO26m**（DETR head，输出 logits+boxes 双张量），同时**保留 EfficientSAM** 做精分割。
- **删掉**：`person_txt_feats.npy`、CLIP txt embeddings 整条链、`generate_yoloworld_person_txt_feats.py` 工具。
- **修复痛点**：
  1. **闪烁** → 因 YOLO-World 余弦相似度 score 校准差（top-score 全在 0.018–0.105，阈值 0.03 贴着分布中位数）。YOLO26m 是 COCO sigmoid score，threshold ~0.35 可用，margin 提升 ~10×。
  2. **单眼丢人** → 同上 + 放宽 plausibility filter（当前 `area∈[0.02,0.24]`、`aspect∈[0.12,1.05]` 太严，VR 近景/远景人物被拒）。
  3. **边缘外扩** → EfficientSAM 输出的 soft mask 在二值化前 halo 较大；新增 mask 后处理（二值化 + 1px erode）+ 保持 `box_expand=0.08` 不动（**不要听信"扩到 15%"那条建议**，会让 SAM 吃更多背景）。
- **不改 MatAnyone2 引擎**：本次只改前置；MatAnyone2 引擎的 bootstrap 处理已在 Phase 1 修复。
- **关键约束**：YOLO26m 输入**硬编码 640×640**，无法走 1280（需要重新 export 才能改），因此小目标/远景的 recall 是新增风险点，要靠 EfficientSAM 后续精修兜底。
- **模块名重命名**：`offline/yoloworld_efficientsam.py` → `offline/yolo26m_efficientsam.py`；CLI 选项前缀 `--ywes-*` → `--y26es-*`；UI label `YOLOWorld-EfficientSAM` → `YOLO26m-EfficientSAM`；保留旧 `--ywes-*` 一个版本作 backward alias。
- **不替换 SAM3 路径**：`recognition=sam3`（高显存模式）继续保留。

---

## 1. 新模型 (YOLO26m) 关键特征

通过 `onnx.load` 直接 inspect 得到（fp32 和 fp16 ONNX 一致）：

| 维度 | 值 |
|---|---|
| 输入名 | `pixel_values` （**不是** YOLO-World 的 `images`） |
| 输入形状 | `[batch_size, 3, 640, 640]` float32 |
| 输入尺寸 | **固定 640**（graph 烧死，无法动态改） |
| 输入归一化 | `pixel / 255.0`（letterbox + RGB；和 YOLO-World 一致，**不需要 ImageNet mean/std**）|
| 输出 #1 | `logits` `[B, 300, 80]` float32 — raw logits，需 sigmoid，80 = COCO classes |
| 输出 #2 | `pred_boxes` `[B, 300, 4]` float32 — **cxcywh, normalized to [0,1]** （DETR 约定）|
| 类别索引 | COCO `class_id=0` = person |
| 后处理 | DETR 风格：无 anchor、无 grid、**无需 NMS**（300 object queries 已稀疏；仍可加 NMS 兜底重复框） |

⚠️ 实操要点：
- `pred_boxes` 是 **归一化到 letterbox 输入空间**（不是绝对像素），需要 `× 640` 再去 letterbox 还原到原图。
- `logits` 是 sigmoid-style multi-label logits（每个 query 对 80 类各预测一个分数），不是 softmax；用 `sigmoid()` 不是 `softmax()`。
- fp16 模型 ONNX I/O 仍是 float32（内部 weights 是 fp16）→ ORT 自动处理 cast；建议默认用 **fp16 模型**，CUDA 推理更快。

---

## 2. 当前代码结构（待替换）

主文件：[`offline/yoloworld_efficientsam.py`](offline/yoloworld_efficientsam.py:1)（671 行）

主要类与函数：
- `Detection` dataclass (`box_xyxy`, `score`, `class_id`)
- `YoloWorldEfficientSamMasker.__init__`：加载 yolo + sam 两个 ONNX session，加载 txt_feats
- `.detect()` ([L122-157](offline/yoloworld_efficientsam.py:122))：letterbox → YOLO-World forward → output transpose → 解析 boxes_xywh + class_scores → NMS
- `._is_plausible_person_box()` ([L174-180](offline/yoloworld_efficientsam.py:174))：area/aspect/height 三个硬阈值过滤
- `.select_stereo_detections()` ([L199-249](offline/yoloworld_efficientsam.py:199))：左右眼 detection + cost-based pairing
- `._sam_mask_for_box()` ([L251-285](offline/yoloworld_efficientsam.py:251))：EfficientSAM box prompt → mask
- `.mask()` ([L287-311](offline/yoloworld_efficientsam.py:287))：union of detections → final mask + info dict
- `precompute_segment_masks()` ([L451-591](offline/yoloworld_efficientsam.py:451))：scan_points 采样、scene cut 检测、prepass 主循环、segment plan 规划、gap fill
- `precompute_segment_masks_subprocess()` ([L619-670](offline/yoloworld_efficientsam.py:619))：子进程包装（隔离 CUDA context）

依赖与下游：
- [`tools/offline_alpha_passthrough.py:915-955`](tools/offline_alpha_passthrough.py:915)：所有 `--ywes-*` argparse
- [`tools/offline_passthrough.py`](tools/offline_passthrough.py)：同上一份 argparse（dual-tool 重复）
- [`offline/convert.py:192`](offline/convert.py:192)：`--matanyone2-prepass yoloworld_efficientsam`
- [`ui/pages/offline_page.py:197, 558, 772`](ui/pages/offline_page.py:197)：UI 下拉项
- [`ui/translations/{zh_CN,ja_JP,en_US}.json`](ui/translations/)：标签 `recognition.yoloworld_efficientsam`
- `models/person_txt_feats.npy`：CLIP-encoded "person/human/man/woman/pedestrian" 5 类向量
- `tools/generate_yoloworld_person_txt_feats.py`：txt_feats 生成工具

EfficientSAM 段（[L251-285](offline/yoloworld_efficientsam.py:251)）**不改逻辑、只改路径**——把 `models/yoloworld_efficientsam/efficientsam_s.onnx` 改成 `models/efficientsam/efficientsam_s.onnx`。

---

## 3. 解决问题的机制（落到代码层面）

| 痛点 | 当前根因 | 新方案 |
|---|---|---|
| **画面闪烁** | YOLO-World 余弦 score top 全在 `0.02-0.10` 范围；阈值 0.03 紧贴中位数 → 抖动翻转 active/inactive；阶段性 score 起伏导致 stereo cost 求和符号不稳定 | YOLO26m sigmoid score 真概率；阈值 `0.35` 离群点 margin 提升 ~10×；同时**修复 stereo cost 量纲**（见 §5.3） |
| **单眼丢人** | (a) score 校准差 + threshold 0.03 切线性； (b) `_is_plausible_person_box` 三个硬阈值砍掉真人；(c) fallback `fallback_unfiltered` 无置信度护栏，反而放进去脏 detection | (a) ↑ 解决； (b) plausibility 三参数大幅放宽，新增 conf 护栏； (c) fallback 增加 min-score 护栏 |
| **抠图边缘外扩** | (a) `_sam_mask_for_box` 直接返回 sigmoid soft mask；(b) `union_area_ratio` 只用 ≥0.5 阈值但 mask 本身仍包含 halo；(c) 下游 MatAnyone2 bootstrap 虽然 erode=1 px，但 1px 在 1024 输入下相对 8K 原图非常小 | EfficientSAM mask 在写入 npz 前做：(i) 二值化 `mask >= 0.5`； (ii) 1-2 px morphological erosion； (iii) 可选保留 soft band（用于 MatAnyone2 first_frame_refine 软边） |

⚠️ 不动 `box_expand`（保持 `0.08`）。把 box 扩到 0.15 是反方向——EfficientSAM 的 box prompt label `[2, 3]`（top-left + bottom-right）相当于告诉 SAM "目标在这个区间内寻找"，box 越大 SAM 越容易把贴近人体的背景吃进来。

---

## 4. 总体方案

### 4.1 新建一个文件，不直接覆盖旧文件（灰度）

新建 [`offline/yolo26m_efficientsam.py`](offline/yolo26m_efficientsam.py)，保留 `offline/yoloworld_efficientsam.py` 一个发版周期。

理由：
1. 出问题可立即回滚 prepass 选项
2. A/B 用户测试容易
3. 子进程 CLI 双工具同时改风险大

### 4.2 一次性切换还是开关切换？

**开关切换**：UI 下拉新增 `YOLO26m-EfficientSAM` 选项，**默认选中**；旧 `YOLOWorld-EfficientSAM` 选项标记 `(Legacy)` 保留一个版本。

- argparse 选项 `--matanyone2-prepass` 增加新 choice：`yolo26m_efficientsam`
- 下个版本（验证通过后）：删除 `yoloworld_efficientsam` choice 和文件

### 4.3 模块命名

| 类型 | 旧 | 新 |
|---|---|---|
| Python 模块 | `offline.yoloworld_efficientsam` | `offline.yolo26m_efficientsam` |
| 主类 | `YoloWorldEfficientSamMasker` | `Yolo26mEfficientSamMasker` |
| argparse 前缀 | `--ywes-*` | `--y26es-*`（保留 `--ywes-*` alias） |
| prepass choice | `yoloworld_efficientsam` | `yolo26m_efficientsam` |
| UI i18n key | `recognition.yoloworld_efficientsam` | `recognition.yolo26m_efficientsam` |
| 模型目录 | `models/yoloworld_efficientsam/` | `models/yolo26m/` + `models/efficientsam/` |
| 子进程 npz 临时目录 | `_ywes_prepass/` | `_y26es_prepass/` |

---

## 5. 实施任务清单（按顺序）

### Phase A — 新模块 + 单元测试（开发者主要工作量）

#### A1. 新建 [`offline/yolo26m_efficientsam.py`](offline/yolo26m_efficientsam.py)

复制 `yoloworld_efficientsam.py` 为起点，做以下修改：

**A1.1 删除 txt_feats 相关全部代码**
- 删 `__init__` 的 `txt_feats_path` 入参、`self.txt_feats`、`self.num_classes`
- 删 `print` log 里的 `txt_classes=...`
- 后续 detect 不再读 txt_feats

**A1.2 改 `__init__` 默认参数**
```python
def __init__(
    self,
    model_dir: Path,            # models/yolo26m
    sam_model_dir: Path,        # models/efficientsam   (新参数，独立路径)
    provider: str = "cuda",
    yolo_model: str = "yolo26m_model_fp16.onnx",       # 默认 fp16
    sam_model: str = "efficientsam_s.onnx",
    yolo_size: int = 640,        # 改默认 1280 → 640（YOLO26m 硬编码）
    score_threshold: float = 0.35,  # 改 0.03 → 0.35
    nms_threshold: float = 0.6,
    box_expand: float = 0.08,    # 保持
    top_k: int = 1,
    person_class_id: int = 0,    # 新参数：COCO person
    binarize_mask: bool = True,  # 新参数：mask 二值化
    mask_erode_px: int = 1,      # 新参数：mask 腐蚀像素
) -> None:
    ...
    self.yolo = ort.InferenceSession(str(model_dir / yolo_model), ...)
    self.sam = ort.InferenceSession(str(sam_model_dir / sam_model), ...)
```

**A1.3 重写 `.detect()` — 这是改动最重的部分**

```python
def detect(self, image_rgb: np.ndarray, top_k: int | None = None) -> list[Detection]:
    h, w = image_rgb.shape[:2]
    # yolo_size 实际固定 640；letterbox 必须用 640
    inp, scale, pad_x, pad_y = _preprocess_yolo(image_rgb, self.yolo_size)
    # 输入名 pixel_values，不是 images
    outputs = self.yolo.run(None, {"pixel_values": inp})
    logits = outputs[0][0]      # [300, 80]
    pred_boxes = outputs[1][0]  # [300, 4] cxcywh normalized [0,1]
    
    # sigmoid 取 person 分数
    person_scores = _sigmoid(logits[:, self.person_class_id])  # [300]
    keep_mask = person_scores >= self.score_threshold
    if not np.any(keep_mask):
        return []
    
    boxes_norm = pred_boxes[keep_mask]       # [N, 4] cxcywh in [0,1]
    scores = person_scores[keep_mask]        # [N]
    
    # 从 normalized cxcywh 还原到原图 xyxy（先回到 letterbox 640 像素空间，再去 letterbox）
    cx = boxes_norm[:, 0] * self.yolo_size
    cy = boxes_norm[:, 1] * self.yolo_size
    bw = boxes_norm[:, 2] * self.yolo_size
    bh = boxes_norm[:, 3] * self.yolo_size
    x1 = (cx - bw / 2.0 - pad_x) / max(scale, 1e-6)
    y1 = (cy - bh / 2.0 - pad_y) / max(scale, 1e-6)
    x2 = (cx + bw / 2.0 - pad_x) / max(scale, 1e-6)
    y2 = (cy + bh / 2.0 - pad_y) / max(scale, 1e-6)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    valid = (boxes[:, 2] > boxes[:, 0] + 2) & (boxes[:, 3] > boxes[:, 1] + 2)
    boxes = boxes[valid]
    scores = scores[valid]
    # DETR 通常不需要 NMS，但兜底加一层（IoU 0.6）
    keep_indices = _nms(boxes, scores, self.nms_threshold)[: (top_k or self.top_k)]
    return [Detection(boxes[i], float(scores[i]), 0) for i in keep_indices]
```

⚠️ 测试要点：letterbox `pad_x, pad_y` 计算返回的是 **原图→640 的偏移**，不是 640→原图。复用现有 `_letterbox_rgb` 不动。

**A1.4 放宽 `_is_plausible_person_box`**

```python
def _is_plausible_person_box(self, det: Detection, image_shape: tuple[int, int, int]) -> bool:
    stats = self._box_stats(det, image_shape)
    return (
        0.005 <= stats["area"] <= 0.55     # 0.02-0.24 → 0.005-0.55（覆盖远景小目标到 VR 近景大目标）
        and 0.10 <= stats["aspect"] <= 2.5  # 0.12-1.05 → 0.10-2.5（覆盖躺姿/侧身）
        and stats["height"] >= 0.10        # 0.25 → 0.10（小目标）
        and det.score >= 0.45              # 新增：高置信度才考虑（双闸门）
    )
```

为什么 area 上限 0.55：VR 近景下，主体能占满一只眼半幅画面的过半。
为什么 aspect 上 2.5：人物躺下/弯腰时 w/h > 1 是常见的。
为什么 score 闸 0.45：和 score_threshold 0.35 形成"通过"vs"plausible"两个档位，避免 fallback 错位。

**A1.5 修 `select_stereo_detections` 的 cost 量纲 bug**

当前 cost：
```python
cost = (
    abs(ls["cy"] - rs["cy"]) * 4.0       # 范围 [0,4]
    + abs(ls["height"] - rs["height"]) * 2.0  # 范围 [0,2]
    + abs(ls["aspect"] - rs["aspect"])   # 范围 [0,~]
    + abs(ls["area"] - rs["area"]) * 2.0
    - (ldet.score + rdet.score)          # score 0-1 量级，最多减 2
)
```

问题：score 量级和 cost 项不平衡。如果两个候选都是低分（YOLO-World 时代 0.03 vs 0.05），减 0.08 vs 减 0.10 几乎无差别，cost 全由 cy/height/area 决定 → 高度依赖 plausibility 通过的候选；如果候选很贫瘠就退化到 `fallback_unfiltered`。

新 cost（按 score 加权）：
```python
geom_cost = (
    abs(ls["cy"] - rs["cy"]) * 4.0
    + abs(ls["height"] - rs["height"]) * 2.0
    + abs(ls["aspect"] - rs["aspect"]) * 1.0
    + abs(ls["area"] - rs["area"]) * 2.0
)
# score 0-1 量级，乘以 4 让它与 geom_cost 同量级
score_bonus = (ldet.score + rdet.score) * 4.0
cost = geom_cost - score_bonus
```

**A1.6 `fallback_unfiltered` 加分数护栏**

当前 `fallback_unfiltered` 直接取 detect 第一个，可能是 0.36 的低分误检。新增：
```python
fallback_min_score = max(self.score_threshold * 1.5, 0.45)
left_fallback = [d for d in left_all if d.score >= fallback_min_score]
right_fallback = [d for d in right_all if d.score >= fallback_min_score]
left_sel = left_fallback[:1] if left_fallback else []
right_sel = right_fallback[:1] if right_fallback else []
mode = "fallback_score_gate" if (left_sel or right_sel) else "no_detection"
```

**A1.7 `_sam_mask_for_box` 输出端 mask 后处理**

```python
def _sam_mask_for_box(self, image_rgb, box_xyxy, out_size):
    # ...原有 SAM forward 不变...
    mask = flat_masks[idx].astype(np.float32, copy=False)
    if mask.min() < 0.0 or mask.max() > 1.0:
        mask = _sigmoid(mask)
    mask = np.clip(mask, 0.0, 1.0)
    
    # === 新增后处理 ===
    if self.binarize_mask:
        bin_mask = (mask >= 0.5).astype(np.uint8)
        if self.mask_erode_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bin_mask = cv2.erode(bin_mask, kernel, iterations=int(self.mask_erode_px))
        mask = bin_mask.astype(np.float32)  # 0.0 or 1.0
    return mask
```

⚠️ 不要在这一步做 dilate；下游 MatAnyone2 `--matanyone2-bootstrap-dilate` 已经能控制。

**A1.8 EfficientSAM 路径变更**

只有目录改了：`model_dir / sam_model` → `sam_model_dir / sam_model`。

**A1.9 重写 `precompute_segment_masks` 中的 `YoloWorldEfficientSamMasker(...)` 构造**

替换为 `Yolo26mEfficientSamMasker(...)`，参数列表对应更新（删 `txt_feats_path`，加 `sam_model_dir`、`binarize_mask`、`mask_erode_px`）。

**A1.10 `precompute_segment_masks_subprocess` 命令构造**

子进程 cmd 中：
- 删 `--ywes-txt-feats`
- 改 `--ywes-yolo-model` 默认值
- 增加 `--y26es-sam-model-dir` 参数（独立 EfficientSAM 路径）
- 增加 `--y26es-binarize-mask` / `--y26es-mask-erode-px`

⚠️ 子进程 npz 临时目录路径同步换名：`debug_output/_ywes_prepass` → `debug_output/_y26es_prepass`。

#### A2. 单元测试 [`tests/test_yolo26m_efficientsam.py`](tests/test_yolo26m_efficientsam.py)

新文件，至少覆盖：
1. `_letterbox_rgb` 已有测试不动
2. `detect()` 用 mocked `ort.InferenceSession.run`，验证：
   - 输入字典只有 `pixel_values`（不含 `txt_feats`、`images`）
   - 输出解析 logits[300,80] / pred_boxes[300,4]
   - 归一化 cxcywh → 绝对 xyxy 计算正确（用 letterbox 640 反算原图坐标）
   - score 阈值 0.35 默认正确
3. `_is_plausible_person_box` 边界：area 0.005/0.55、aspect 0.10/2.5、height 0.10、score 0.45
4. `select_stereo_detections` cost 量纲：
   - 两边候选都高分时偏向高分对
   - 高分+几何差 vs 低分+几何接近 → 前者胜（验证 score_bonus * 4.0 系数）
5. `fallback_score_gate` 模式：
   - 候选全低于 0.45 → 返回空
   - 候选高于 0.45 但低于 plausibility area 范围 → 仍返回（fallback 不应用 plausibility）
6. `_sam_mask_for_box` 后处理：
   - `binarize_mask=True` 输出只有 0/1
   - `mask_erode_px=1` 后区域比原 mask 小
   - `binarize_mask=False` 保留 soft 输出

#### A3. 真实 ONNX session 烟雾测试（可选，CI 跳过）

在本地或 dev 机器上跑一个 1 帧的真实推理，验证：
- `pixel_values` 输入名匹配 ONNX
- 输出 shape 与 dtype 符合预期
- person score top 在合理范围（0.3-0.9 区间）

### Phase B — CLI 与子进程参数

#### B1. [`tools/offline_alpha_passthrough.py`](tools/offline_alpha_passthrough.py)

**B1.1 argparse 修改**

`--matanyone2-prepass` choices：
```python
choices=["sam3", "yoloworld_efficientsam", "yolo26m_efficientsam"]
```

新增参数族 `--y26es-*`（参考现有 `--ywes-*`，做以下变化）：

```python
parser.add_argument("--y26es-model-dir", default=str(config.ROOT / "models" / "yolo26m"))
parser.add_argument("--y26es-sam-model-dir", default=str(config.ROOT / "models" / "efficientsam"))
parser.add_argument("--y26es-provider", default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--y26es-yolo-model", default="yolo26m_model_fp16.onnx")
parser.add_argument("--y26es-sam-model", default="efficientsam_s.onnx")
parser.add_argument("--y26es-yolo-size", type=int, default=640,
                    help="YOLO26m letterbox input size; graph 固定 640，参数仅用于 letterbox 计算")
parser.add_argument("--y26es-score-threshold", type=float, default=0.35)
parser.add_argument("--y26es-nms-threshold", type=float, default=0.6)
parser.add_argument("--y26es-box-expand", type=float, default=0.08)
parser.add_argument("--y26es-top-k", type=int, default=1)
parser.add_argument("--y26es-binarize-mask", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--y26es-mask-erode-px", type=int, default=1)
parser.add_argument("--y26es-scan", default="hybrid", choices=["keyframe", "interval", "hybrid"])
parser.add_argument("--y26es-scan-interval-sec", type=float, default=1.0)
parser.add_argument("--y26es-active-min-area-ratio", type=float, default=0.001)
parser.add_argument("--y26es-gap-fill-frames", type=int, default=300)
parser.add_argument("--y26es-debug-dir", default="")
parser.add_argument("--y26es-cut-on-count-change", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--y26es-cut-every-active-sample", action="store_true")
parser.add_argument("--y26es-fail-on-empty", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--y26es-subprocess", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--y26es-prepass-out", default="", help=argparse.SUPPRESS)
```

旧 `--ywes-*` 参数保留（不删，作 backward alias）。

**B1.2 dispatch**

```python
if args.engine == "matanyone2_onnx" and not args.mask and args.matanyone2_prepass == "yolo26m_efficientsam":
    from offline.yolo26m_efficientsam import precompute_segment_masks as _precompute_y26es_segment_masks
    from offline.yolo26m_efficientsam import write_prepass_result as _write_y26es_prepass_result
    sam3_masks, segment_starts = _precompute_y26es_segment_masks(args, src, dec, source_fps, fps, target, cfr_source_index)
    args._y26es_child = bool(args.y26es_prepass_out)
elif args.engine == "matanyone2_onnx" and not args.mask and args.matanyone2_prepass == "yoloworld_efficientsam":
    # 旧路径保留
    ...
```

**B1.3 在 `args` 初始化处增加**

```python
args._y26es_child = bool(args.y26es_prepass_out)
```

#### B2. [`tools/offline_passthrough.py`](tools/offline_passthrough.py)

镜像 B1，所有同名修改重复一份。**重要**：本项目两个工具脚本是手动同步的，不要假装能一键 sed，每个 add_argument 都要确认。

### Phase C — Convert / UI / 翻译

#### C1. [`offline/convert.py`](offline/convert.py)

[L192](offline/convert.py:192)：
```python
if args.engine == "matanyone2_medium":
    cmd.extend(["--matanyone2-prepass", "yolo26m_efficientsam"])
```

#### C2. [`ui/pages/offline_page.py`](ui/pages/offline_page.py)

[L195-199](ui/pages/offline_page.py:195)：
```python
def _recognition_combo(self) -> QComboBox:
    combo = _fit_combo(QComboBox())
    combo.addItem("", "yolo26m_efficientsam")        # 新默认
    combo.addItem("", "yoloworld_efficientsam")      # 标 (Legacy)
    combo.addItem("", "sam3")
    return combo
```

[L558](ui/pages/offline_page.py:558)：
```python
return "matanyone2_medium" if recognition in ("yolo26m_efficientsam", "yoloworld_efficientsam") else "matanyone2"
```

[L772](ui/pages/offline_page.py:772) 及附近的 setItemText 循环增加新 key 的翻译。

#### C3. UI 翻译

新增三语：
```json
"recognition.yolo26m_efficientsam": "YOLO26m-EfficientSAM",
"recognition.yoloworld_efficientsam": "YOLOWorld-EfficientSAM (Legacy)",
```

`offline.matanyone_help_msg` 中文：
```
请查看 models 目录中的说明文件，下载 MatAnyone2 模型。
YOLO26m-EfficientSAM 模式还需要下载 YOLO26m 和 EfficientSAM 的 ONNX 模型。
SAM3 模式还需要下载 SAM3 模型，并需要至少 16GB 显存。
```

英文、日文同步。三个翻译 JSON 均为 **UTF-8 with BOM**，保留 BOM。

### Phase D — 清理与文档

#### D1. 文件清理（**等 D-1 用户验证通过后再做**）

下个版本删除：
- `tools/generate_yoloworld_person_txt_feats.py`
- `models/person_txt_feats.npy`
- `models/yoloworld_efficientsam/yolov8s-worldv2.onnx`
- `models/yoloworld_efficientsam/yolov8l-worldv2.onnx`
- `models/yoloworld_efficientsam/get_yolo-wolrd_model_readme.txt`
- `offline/yoloworld_efficientsam.py`
- 所有 `--ywes-*` argparse 项

⚠️ 本次开发周期**不删**，灰度一个版本。

#### D2. `models/yolo26m/新建文本文档.txt` 重命名

建议改为 `get_yolo26m_model_readme.txt`，与项目其他模型 README 命名风格统一。

#### D3. PROJECT.md 更新

在 §"Key Configuration Reference" 表中：
- 删除 `PT_YWES_*`（如果有 — 实际上目前的 YW+ES 配置在 argparse 而不是 config.py，无需改 PROJECT.md 表）
- 在 §"Notes for Future Work" 提一句 `yolo26m_efficientsam` 是新默认前置

#### D4. 新建 `summary/summary_20260527_YOLO26M_REPLACE_YOLOWORLD_RESULTS_CN.md`

验证完成后写：实测的 score 分布、闪烁次数对比、边缘宽度对比、单眼丢人率对比。

### Phase E — 验证（user-side QA）

#### E1. 烟雾测试（开发者）

```bat
.\.venv\Scripts\python.exe tools\offline_alpha_passthrough.py videos\<test.mp4> ^
    --engine matanyone2_onnx ^
    --matanyone2-prepass yolo26m_efficientsam ^
    --y26es-debug-dir debug_output\y26es_smoke ^
    --frames 60
```

检查：
- 子进程正常启动并退出
- `debug_output/y26es_smoke/seg_*_left_mask.png` 是干净的人形二值 mask（不应有大片背景）
- log 中 `score=` 在 0.4-0.9 区间（YOLO-World 时代是 0.02-0.10）
- log 中 `stereo=paired` 比例显著高于旧路径（pair 配对成功率提高）

#### E2. 回归（开发者）

```bat
.\.venv\Scripts\python.exe -m py_compile config.py ^
    offline\yolo26m_efficientsam.py ^
    offline\yoloworld_efficientsam.py ^
    offline\convert.py ^
    tools\offline_passthrough.py ^
    tools\offline_alpha_passthrough.py ^
    ui\pages\offline_page.py

.\.venv\Scripts\python.exe -m pytest ^
    tests\test_yolo26m_efficientsam.py ^
    tests\test_matanyone2_engine.py ^
    tests\test_offline_convert.py ^
    tests\test_matanyone2_trt_runtime_paths.py
```

期望：全部 pass。

#### E3. 用户验证（产品验收）

请用户用之前出问题的几个素材（包括 `72456_3840p.mp4`）重新跑 `matanyone2_medium`，对比：
- ✅ 闪烁消失或显著减少
- ✅ 单眼丢人现象消失
- ✅ 抠图边缘紧贴人体轮廓
- ✅ 处理速度与旧路径相当或更快（YOLO26m 640 输入 vs YOLO-World 1280 输入，理论上更快）

---

## 6. 风险与回滚

| 风险 | 影响 | 缓解 |
|---|---|---|
| YOLO26m 640 输入小，VR 远景人物 recall 下降 | 远处人物完全丢检 | (a) plausibility/score 已放宽；(b) 必要时切回 YOLOWorld-EfficientSAM Legacy；(c) 用户考虑 export YOLO26m 到 1024/1280（需 Ultralytics 工具） |
| DETR head 输出无 NMS，相同人物可能多框 | top_k=1 时第一个不是最优 | 已保留 `_nms()` 兜底，IoU 0.6 |
| COCO 训练数据无 fisheye 失真 | VR 鱼眼边缘人物识别率仍受限 | 与 YOLO-World 同问题，不变好不变坏；属于数据域差异 |
| mask 二值化 + erode 在头发/手指细节丢失 | 首帧 mask 比 soft mask 略保守 | MatAnyone2 first_frame_refine 在递归 3 次后会扩 mask；如果验证发现头发明显缩进，调 `--y26es-mask-erode-px 0` |
| EfficientSAM 路径搬迁后 dist/ 打包缺失 | 打包后 ONNX 找不到 | PyInstaller spec 检查 `models/efficientsam/` 是否在 datas 列表里 |
| 翻译文件 BOM 丢失 | UI 显示乱码 | 写入时强制 `utf-8-sig` |

### 回滚步骤

如果用户反馈新路径质量倒退：
1. UI 默认 combo 改回 `yoloworld_efficientsam`（一行改 [offline_page.py:197](ui/pages/offline_page.py:197) 顺序）
2. `offline/convert.py:192` 改回 `yoloworld_efficientsam`
3. 不删任何文件；下版本再决定方向

---

## 7. 估时

| Phase | 工作量 | 备注 |
|---|---|---|
| A1 新模块 | 1.5 天 | detect() 重写 + plausibility 调参 + cost 修复 + mask 后处理 |
| A2 单测 | 0.5 天 | 6 组 case |
| A3 烟雾 | 0.5 天 | 包含找 ONNX 输入/输出 dtype 调试时间 |
| B1+B2 CLI | 0.5 天 | argparse 重复两遍 |
| C convert/UI/翻译 | 0.5 天 | UI 下拉 + i18n |
| D 文档 | 0.5 天 | summary + PROJECT.md |
| E 用户验证 + 调参 | 1 天 | 大概率要根据用户反馈微调阈值 |
| **合计** | **~4.5 天** | |

---

## 8. 给开发者的注意事项

1. **不要假设 YOLO26m 行为和 YOLO-World 一样**：输出结构、输入名、归一化、score 含义、阈值范围、有无 NMS 全部不同。本计划 §1 是 ground truth，请以此为准。
2. **不要修改 EfficientSAM 调用逻辑**：除了模型路径迁移，`_sam_mask_for_box` 的输入预处理、box prompt label `[2.0, 3.0]`、`batched_point_coords` 形状全部保持。只在输出端加二值化和 erode。
3. **不要扩 `box_expand`**：用户的"边缘外扩"和"box 给 SAM 的范围"是两件事。box 扩 0.15 会让 SAM 吃更多背景，反向加剧问题。
4. **不要改 MatAnyone2 engine**：本次只改前置。MatAnyone2 内部的 bootstrap_threshold/erode/dilate/refine 已在 Phase 1 完成调优，不动。
5. **PyInstaller spec**：`pyproject.toml` 或 `tools/build_dist.py`（如存在）需检查 `models/efficientsam/` 加入打包 datas。
6. **opset 18**：YOLO26m 是 opset 18，ORT 版本需要支持（当前项目 ORT 1.20+ 应该没问题，预先核对）。
7. **CUDA context isolation 仍需保留**：YOLO26m + EfficientSAM 联合占用 VRAM 在 4GB+ 量级，MatAnyone2 1024 又要 8GB+，必须保留 `--y26es-subprocess` 默认开启，子进程结束自动释放 VRAM。

---

## 9. 验收标准 (Definition of Done)

- [ ] `offline/yolo26m_efficientsam.py` 新文件创建并 import 无错
- [ ] `tests/test_yolo26m_efficientsam.py` 6 组测试全部 pass
- [ ] 两个 CLI 工具的 `--matanyone2-prepass yolo26m_efficientsam` 烟雾测试 60 帧通过
- [ ] UI 下拉新选项可见、中英日三语正确
- [ ] 默认配置（不传 `--y26es-*`）跑 `matanyone2_medium` 完成 30 秒视频处理
- [ ] 用户在之前出问题的素材上确认：闪烁、单眼丢人、边缘外扩三个问题均明显改善
- [ ] 全套回归 `py_compile` + `pytest` 通过
- [ ] PROJECT.md 与 i18n 同步更新
- [ ] 新增 `summary/summary_20260527_YOLO26M_REPLACE_YOLOWORLD_RESULTS_CN.md` 记录实测对比数据

---

## 附录 A — 测试视频清单（建议）

| 文件 | 测试目的 |
|---|---|
| `videos/72456_3840p.mp4` | 已知 drag/floor inclusion 问题的素材 |
| `videos/test_8k_*.mp4` | 8K 远景/小目标 recall 测试 |
| 用户提供的"闪烁严重"素材 | 闪烁次数对比 |
| 用户提供的"单眼丢人"素材 | 双眼 stereo pairing 成功率对比 |

## 附录 B — 关键数据结构对照表

| 字段 | YOLO-World | YOLO26m |
|---|---|---|
| ONNX 输入键 | `images` + `txt_feats` | `pixel_values` |
| 输入大小 | 1280×1280（可调） | 640×640（固定） |
| 输出 tensor 数 | 1 | 2 |
| 输出 #1 shape | `[1, 4+C, N]` (C=类别数) | `logits [1, 300, 80]` |
| 输出 #2 shape | — | `pred_boxes [1, 300, 4]` |
| boxes 单位 | xywh 绝对像素 | cxcywh normalized [0,1] |
| scores 解析 | `class_scores[argmax]` | `sigmoid(logits[:, 0])` for person |
| score 范围 | ~0.02-0.10 | ~0.30-0.95 |
| 推荐阈值 | 0.03 | 0.35 |
| 需要 NMS | 是 | DETR head 通常不需要，兜底加 |
| 额外资源 | `person_txt_feats.npy`（CLIP-encoded） | 无 |
