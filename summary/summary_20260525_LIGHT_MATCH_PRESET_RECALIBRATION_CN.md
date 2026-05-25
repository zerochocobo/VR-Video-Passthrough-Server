# 光照匹配预设重新校准（方案 A）

日期：2026-05-25
目标：解决用户反馈的"默认 daylight 偏黄、night_cool 不冷"的问题，把三档预设校正到符合显示器白点 (D65) 的色温梯度。

## 1. 问题背景

### 1.1 用户主诉
默认的"自然日光"（daylight）画面偏黄，三档预设整体感觉都偏暖。

### 1.2 根因分析

`pipeline/light_match.py` 把 `NEUTRAL_TEMP_K = 6500K` 当成色温运算锚点（line 71），`_rgb_gains()` (line 159-172) 用 `target / neutral` 计算 chroma gain。该锚点选择是对的——6500K = D65，是显示器/电视/Quest3 LCD 的行业标准白点。

但 `ui/pages/home_page.py:55-59` 的三档预设值与该锚点不匹配：

| Preset | temp_k | chroma gain (R/G/B, 经 luma-preserve 归一) | R/B 比值 | 视觉 |
|---|---|---|---|---|
| home_warm  | 4000K | 1.191 / 0.965 / 0.791 | 1.51 | 重度偏黄/红（钨丝灯） |
| daylight   | **5500K** | 1.058 / 0.989 / 0.940 | **1.12** | **明显偏黄** ← 用户主诉 |
| night_cool | **6500K** | 1.000 / 1.000 / 1.000 | **1.00** | **几乎中性，并不"冷"** |

#### 两个关键错配
- **daylight 5500K = D55（中午太阳直射光）**，在 D65 锚点下被乘以偏暖系数，输出偏黄。命名"自然日光"暗示中性白，与实际表现冲突。
- **night_cool 6500K 正好等于 NEUTRAL_TEMP_K**，色彩矩阵全是 1.0，色温维度完全无效。剩下的 -0.1 EV / 0.95 saturation 微调肉眼几乎不可察。"冷白光"名不副实。

#### 历史成因
`CHANGELOG.md` 提到曾经 "softened the warm preset"——只调整了 home_warm，daylight/night_cool 一直没动，导致三档相对锚点分布失衡。

## 2. 方案 A：重新校准三档色温

### 2.1 新预设值

`ui/pages/home_page.py:55-59`：

```python
LIGHT_MATCH_PRESETS = {
    "home_warm":  {"temp_k": 4000, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
    "daylight":   {"temp_k": 6500, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
    "night_cool": {"temp_k": 8000, "tint": 0, "exposure_ev": 0.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0},
}
```

变更点：
- `home_warm` 保留 4000K，意图就是暖光氛围。
- `daylight` 4500K → **6500K (D65)**，真正中性白，匹配显示器白点。
- `night_cool` 6500K → **8000K**，明显偏冷蓝。
- 撤销 `night_cool` 的 `exposure_ev=-0.1` 和 `saturation=0.95`——这两个值视觉上几乎看不出，仅徒增混淆。冷白光应该用色温（chroma gain）表达，而不是搞 EV/sat 微调。

### 2.2 校准后的色偏

| Preset | temp_k | chroma gain (R/G/B) | R/B 比值 | 视觉 |
|---|---|---|---|---|
| home_warm  | 4000K | 1.191 / 0.965 / 0.791 | 1.51 | 暖黄（不变） |
| daylight   | 6500K | 1.000 / 1.000 / 1.000 | 1.00 | **中性白（D65）** |
| night_cool | 8000K | ~0.926 / 0.995 / 1.098 | **0.84** | **明显偏冷蓝** |

三档真正形成 **暖 → 中性 → 冷** 的色温梯度，与命名一致。

### 2.3 不动锚点 `NEUTRAL_TEMP_K = 6500`
- 6500K = D65 是显示器/电视/HMD LCD 行业白点，必须保留。
- 改锚点会导致同一份 fMP4 在不同播放器/设备上呈现不一致。

## 3. 兼容性 & migration

### 3.1 影响范围
- **现有用户 `light_match_enabled=False`**：无影响（不应用任何变换）。
- **现有用户 `light_match_enabled=True` + preset ∈ {home_warm, daylight, night_cool}**：开机后会看到画面颜色变化（主要是 daylight 用户：从偏黄变成中性；night_cool 用户：从几乎不变变成明显偏冷）。
- **现有用户 `preset=custom`**：完全不受影响（自定义 temp_k 不变）。

### 3.2 migration 策略

`ui/settings.py:load()` 已有 `20260524_light_match_daylight_default` 模式。新增一条：

```python
if not self._migration_done("20260525_light_match_temps_recalibrated", loaded):
    preset = str(loaded.get("light_match_preset", "custom") or "custom").strip().lower()
    if preset in {"daylight", "night_cool"}:
        # 用户之前选的是按旧色温值校准的 preset，新值色感会变。
        # 不强制改成 custom（那样会切断 preset 联动）；只记录一条 changelog/toast。
        pass
    self._mark_migration_done("20260525_light_match_temps_recalibrated")
```

**判断**：不做强制迁移，理由：
1. 旧 daylight 偏黄属于 bug 行为，校准成中性才符合用户预期，无需保留旧观感。
2. 旧 night_cool 几乎是 identity，校准成真正偏冷反而让该 preset 名副其实。
3. preset 选择是"档位"语义，不是"具体色温值"语义；用户选 daylight 期望的是"日光感"而不是"5500K 这个具体数值"。

唯一要做的是：第一次启动新版本时弹一次 toast / changelog 提示"光照匹配色温已校准"。

### 3.3 文档同步
- `CHANGELOG.md` 加 entry。
- `prompt/HANDOVER_20260525.md` 加备注（说明 daylight=D65、night_cool=8000K 的设计理由）。

## 4. 测试

### 4.1 单测更新

`tests/test_light_match.py`：
- 已有的 identity 判断（temp_k==NEUTRAL_TEMP_K 且其它为默认 → identity）现在会让 daylight preset 触发 identity 路径，省一次矩阵乘——是优化，不是回归，但要确认 normalize_light_match_params(daylight) 返回的对象 build_light_match_tables 后 identity=True。

- 新增 case：
  ```python
  def test_daylight_preset_is_identity(self):
      p = LightMatchParams(enabled=True, temp_k=6500, ...其余默认)
      t = build_light_match_tables(p)
      self.assertTrue(t.identity)

  def test_night_cool_preset_cools_blue(self):
      p = LightMatchParams(enabled=True, temp_k=8000, ...)
      t = build_light_match_tables(p)
      self.assertFalse(t.identity)
      # B 通道 chroma gain 应 > 1, R 通道应 < 1
  ```

`tests/test_settings.py:119` `test_light_match_default_preset_is_daylight`：保留，不变（默认仍是 daylight，只是 daylight 含义变成 D65）。

`tests/test_ui_smoke.py:106-117`：
- itemData(0)=="home_warm"、currentData()=="daylight"、findData("home_warm") 等都不变。
- 但如果有断言"切到 home_warm 后 temp_k slider 值=4000"——保持；如果断言 night_cool 后 temp_k=6500——**改为 8000**。

`tests/test_light_match_control.py:29`：preset 设为 home_warm，断言保持。

### 4.2 视觉验证
- 用一段含人物 + 室内/室外混合场景的 mp4，分别切三档，肉眼确认梯度合理。
- 在 Quest3 实机验证（用户主诉来源），重点看 daylight 是否仍偏黄。

## 5. 改动清单

| 文件 | 改动 |
|---|---|
| `ui/pages/home_page.py:55-59` | 三档预设值改写（核心） |
| `ui/settings.py:load()` | 新增 `20260525_light_match_temps_recalibrated` migration（仅打标记，不强改值） |
| `CHANGELOG.md` | 加一行说明色温重新校准 |
| `tests/test_light_match.py` | 新增 daylight=identity 和 night_cool=cool 两个 case |
| `tests/test_ui_smoke.py` | 如有 night_cool 对应 temp_k=6500 断言，改为 8000 |
| `prompt/HANDOVER_20260525.md` | 记录设计理由 |

## 6. 风险与回滚

### 6.1 风险
- 现有 daylight 用户的观感突变（偏黄 → 中性）。是否会有用户喜欢"偏黄"质感？低概率，但如有反馈可后续加一档 `warm_daylight=5500K`。
- night_cool 8000K 偏冷强度可能因显示器而异。Quest3 LCD 偏冷的概率较低；若用户反馈"过冷"，可降到 7500K。

### 6.2 回滚
- 配置驱动，直接还原 `LIGHT_MATCH_PRESETS` dict 即可，无 schema 变化。

## 7. 不在本次范围

以下是相关但本次不动的项：
- `tint` 极性约定（line 163-165）：positive=magenta、negative=green，与 Lightroom 一致，无问题。
- gamma/contrast/saturation 取值范围（GAMMA_MIN..MAX 等）：取值合理，不动。
- `_kelvin_to_rgb` 算法（Tanner Helland 近似）：与业界常用版本一致，不动。
- 是否新增第 4 档预设（如 D50 印刷/D75 极冷）：不必，三档够用，需要时用 custom。
