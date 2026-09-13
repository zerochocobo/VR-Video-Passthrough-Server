# [DLSS5] 前缀功能 可行性与实施策划（中文）

日期：2026-09-12
分支：`DLSS`
状态：策划，未写任何功能代码

前置调研见 [summary_20260911_DLSS5_NR_AND_RTX_VSR_REVIEW_CN.md](summary_20260911_DLSS5_NR_AND_RTX_VSR_REVIEW_CN.md)。本文针对「30 系以上是否都能用」这一新情况重新核实，并给出 `[DLSS5]` 功能的落地方案。

---

## 1. 「30 系以上都能用了」——事实核准

技术上成立，实用上不成立。区别在于**用的是谁的 DLL**。

| | 官方 `nvngx_dlssnr.dll` | 社区修改版（如 ShortFuse `310.8.SF`） |
|---|---|---|
| RTX 50（Blackwell） | 可用 | 可用 |
| RTX 40（Ada） | `0xBAD00001` 拒绝 | 可跑，约 −39% 帧率 |
| RTX 30（Ampere） | `0xBAD00001` 拒绝 | 可跑，**实测低至约 2 FPS** |
| RTX 20（Turing） | 拒绝 | 仅概念验证，无可靠数据 |

修改版做了两件事：**patch 掉硬件 ID 检查**，并把 **FP8 张量指令重映射**到旧架构的 tensor core 上。Blackwell 原生支持 FP8，Ada 靠重映射能跑但打折，Ampere 没有 FP8 硬件，等于软件模拟 —— 这就是 2 FPS 的由来。

所以「30 系能用」的准确说法是：**能初始化、能出画面，但帧率不具备任何实用价值**。对我们这种每帧都要过一遍的视频处理，2 FPS 意味着 1 分钟的片子要跑 15 分钟以上，且 30 系用户会以为功能坏了。

### 官方 SDK 仍然没有

`github.com/NVIDIA/DLSS` 最新发布是 310.9.1，只含 Super Resolution / Ray Reconstruction / Frame Generation，**没有 Neural Rendering、没有可再分发的 `nvngx_dlssnr.dll`**。DLSS 5 SDK 目前只对游戏开发者通过官方渠道提供。

---

## 2. 成本推算：放到我们的场景

游戏侧公开数据（4K）：RTX 50 开 NR 从 71 → 35 FPS，反推 NR 本身约 **14.5 ms/帧 @4K**。

按我们这次在 VSR 上实测到的规律（NGX 耗时由**输入**分辨率主导）外推：

| 场景 | 输入像素 | NR 推算 ms/帧 | 叠加现有 VSR 后 | 推算 FPS |
|---|---|---|---|---|
| 2D 1080p | 2.1 MP | ~4 | — | ~200（单 NR） |
| 2D 4K | 8.3 MP | ~14.5 | — | ~65（单 NR） |
| VR 4K SBS（每眼 1920²） | 3.7×2 MP | ~26（双眼） | +34（VSR） | **~17** |
| VR 8K SBS（每眼 4096²） | 16.8×2 MP | ~116（双眼） | — | **~8** |

**这些是推算，不是实测**，只用于判断量级。结论：

- **实时 VR 接 NR：不可行。** 我们 4K VR 源的 VSR 已经吃掉 34 ms/帧（29 FPS 封顶），再叠 NR 直接掉到 20 FPS 以下。
- **实时 2D ≤1080p：有可能**，但需要 PoC 实测。
- **离线：唯一稳妥的落点。** 2D 4K 单 NR 约 65 FPS，VR 8K 约 8 FPS（慢但能跑完）。

---

## 3. `[DLSS5]` 功能形态建议

### 定位：离线增强模式，与 `[SuperRes]` 并列

沿用项目已有的模式体系（`green` / `alpha` / `two_dvr` / `rm` / `superres` / `face_beauty`），新增 `dlss5`：

| 维度 | 建议 |
|---|---|
| 离线 | **主战场**。输出 `[DLSS5]<片名>.mp4`，与超分一样支持单文件/批量/时间段 |
| 实时 | 第一版**不做**。理由见上；即使做也只限 2D ≤1080p |
| DLNA | 离线成品作为普通文件出现即可，不新增实时条目形态 |
| 与超分的关系 | 可串联：VSR 放大 → NR 增强（NR 在目标分辨率上工作）。成本叠加，需在 PoC 阶段量出总时间 |
| GPU 门槛 | 仅 RTX 50。40 系及以下**不提供**，功能卡片直接隐藏并说明原因 |
| DLL | 用户自备，放到指定目录；缺失时功能不可见 |

### 为什么不做实时

我们这次在超分上已经把实时的天花板摸清楚了：4K VR 源每眼约 17 ms 的 NGX 成本，无论输出多大。NR 是**同量级的第二次神经网络评估**，还要额外做 RGBA16F 转换。实时 VR 上没有任何余量。

---

## 4. 实施路线

### 阶段 0 —— 合规确认（先做，不写代码）

1. 确认 `nvngx_dlssnr.dll` 的获取与许可状态。**官方没有再分发许可，我们不打包、不内置、不提供下载。**
2. 明确只支持 RTX 50。这意味着大部分用户拿不到这个功能 —— 这是产品决策，不是技术问题。
3. 决定是否值得为一个「只有新卡能用、且依赖未公开接口」的功能投入。

### 阶段 1 —— 离线 PoC（一次性，不进产品）

目标只有一个数字：**NR 在 5060 Ti 上，1080p / 4K / 4096×4096（单眼）各多少 ms/帧**。

- 形态：独立进程 worker，D3D12 + NGX feature 18，RGBA16F 进 / RGBA8 出。
- 同时观察：连续帧闪烁（NR 逐帧无时域一致性）、场景切换是否需要 reset、皮肤/纹理的实际增益。
- 关键踩坑（来自 dlss5-visual-enhancer 源码注释）：**退出时不要调 `NVSDK_NGX_D3D12_Shutdown`，feature 18 成功 evaluate 之后调用会卡死。**

### 阶段 2 —— 若 PoC 数字可接受，接离线链路

复用这次会话已经打好的基础：

- `pipeline/superres_stage.py` 的形态 → 新增 `pipeline/dlss5_stage.py`，同样是「NV12 进、NV12 出、全程 GPU」，供离线链路调用。
- `offline/convert.py` 的引擎体系 → 新增 `--engine dlss5`。
- `utils/vr_naming.py` → 新增 `[DLSS5]` 前缀与输出命名，批量发现跳过已有成品。
- 新增 CUDA kernel：`nv12 → rgba16f` 与 `rgba8 → nv12`。
- VR 必须**分眼**处理（和 VSR 一样），注意 equirect 极点畸变会被细节增强放大。

### 阶段 3 —— 实时（只有在阶段 1 数字允许时才考虑）

仅 2D，上限 1080p，超过直接拒绝。复用现有的预检/超时/隔离子进程机制。

---

## 5. 我不建议做的事

- **不要引导用户使用社区修改版 DLL**。那是被 patch 过的 NVIDIA 二进制（绕过硬件检查 + 重映射张量指令）。让产品去依赖它，等于把许可风险和「一次驱动更新就碎」的风险都揽到自己身上，而 30 系换来的还是 2 FPS。
- **不要实现调用方校验绕过 shim**。同上，而且这是绕过厂商的访问控制。要推进的正路是向 NVIDIA 申请 DLSS 5 SDK 访问。
- **不要为 40 系开口子**。−39% 听起来可接受，但同样依赖修改版 DLL。

---

## 6. 一句话结论

`[DLSS5]` 在技术上做得成，但今天它只是一个**「RTX 50 + 用户自备 DLL + 离线处理」**的功能。如果这三个前提你都接受，阶段 1 的 PoC 值得做 —— 它只需要一个数字就能决定后面要不要继续。如果不接受，更务实的方向是把已经跑通的超分链路做深（10-bit 源、实时吞吐定位、RM/超分串联）。

## 参考

- [DLSS5-NeuralScreen（社区项目，RTX 30/40/50）](https://github.com/perseval-BLR/DLSS5-NeuralScreen)
- [DLSS 5 Mods Reach RTX 20-40 GPUs（各代性能数字）](https://shattered.io/install-dlss-5-any-rtx-gpu-2026/)
- [DLSS 5 Now Works on RTX 20 & 30 GPUs](https://wccftech.com/dlss-5-now-works-on-rtx-20-30-gpus-older-graphics-apis-and-even-pcsx2/)
- [NVIDIA/DLSS 官方 SDK 发布页](https://github.com/NVIDIA/DLSS/releases)
- [NVIDIA: DLSS 5 3D-Guided Neural Rendering](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/)


---

## 7. PoC 结果（2026-09-12 实测，RTX 5060 Ti）

代码在 `models/dlss5/native/`（`dlssnr_probe_main.cpp` + `build_probe.ps1`），可重复运行：

```
dlssnr_probe.exe <runtime目录> <数据目录> [feature id] [api版本] [projectid]
```

### 结论：用公开 SDK 的合规路径，feature 18 拿不到

| 尝试 | Init | CreateFeature(18) |
|---|---|---|
| app id 0，API 0x13 / 0x14 / 0x15 | Success | `0xBAD0000C` FAIL_OutOfDate |
| **Init_with_ProjectID**，API 0x13 / 0x14 / 0x15 | Success | `0xBAD0000C` FAIL_OutOfDate |
| 任意 init，API 0x16 及以上 | `0xBAD0000C` | 到不了 |

对照实验（同一程序、同一参数，只换 feature id）：

| feature | 返回码 |
|---|---|
| 18 Neural Rendering | `0xBAD0000C` FAIL_OutOfDate |
| 16 VSR | `0xBAD0000B` FAIL_UnableToInitializeFeature |
| 1 DLSS SR | `0xBAD0000B` |
| 11 Frame Generation | `0xBAD0000B` |

其它 feature 给的是"参数不全"的正常错误，只有 18 给出不同的码，说明它被特殊对待。

### 排除掉的假设

- **不是 app id 被拒**：`Init` 在两种形式下都返回 Success；失败码也不是 `FAIL_Denied`(17)，那才是 "contact NVIDIA" 的拒绝码。
- **不是硬件不支持**：那会是 `FAIL_FeatureNotSupported`(1)，即 40 系用原版 DLL 拿到的 `0xBAD00001`。本机是 5060 Ti。
- **不是客户端库版本**：换用公开 DLSS SDK 310.9.1 的 `nvsdk_ngx_s.lib`（与 RTX Video SDK 那份 sha256 不同、大小差 1.2 MB）后结果完全一样。两份 SDK 的 `NVSDK_NGX_VERSION_API_MACRO` 都是 0x15。

### 参考实现为什么能跑

`neuroframe_engine.dll` 的字符串里写得很清楚：

```
NGX core initialization failed for API versions 0x13..0x20
Could not load NVIDIA NGX core _nvngx.dll. Tried runtime\_nvngx.dll, normal DLL
  search, and NVIDIA DriverStore packages matching nv*.inf_*
DLSSNR snippet Init_Ext via caller shim failed: 0x%08X; loaded shim=%s
CreateFeature(18) failed: 0x%08X. Check GPU support, driver, nvngx_dlssnr.dll, and caller shim.
Required caller shim exports are missing
```

它做了两件我们没做的事：

1. **动态加载驱动里的 `_nvngx.dll`**，因此能把 API 版本一路试到 `0x20`；我们链接静态库，上限就是 `0x15`（0x16 起 `Init` 直接失败）。
2. **通过 caller shim 调用**。`neuroframe_caller.dll` 只导出四个函数 —— `DLSSNR_CallInit/CallCreate/CallEvaluate/CallRelease` —— 且只依赖 `KERNEL32.dll`，不链接任何 NGX 库。它的作用就是充当 NGX 眼里的调用方模块。Zonnery 的 dlss5-nr-player 把这件事说得更直白："the NR DLL rejects direct calls from unknown callers"。

第 1 件事本身合法（`nvsdk_ngx_loader.h` 就是官方的动态加载方式）。第 2 件正撞 NVIDIA RTX SDKs LICENSE 第 4(d) 条。

### 那两个 DLL 的出身

二进制剥离得很干净：没有 PDB 路径、没有 URL、没有 github 字符串。从内容看是 Merserk 自己写的闭源产物 —— 内嵌自有 PTX kernel（`dlss5nr_nv12_to_rgb`、`dlss5nr_rgb_to_rgba16f`、`dlss5nr_tone_grid`、`dlss5nr_blur_skin_*` 等）和自有 ABI（`dlss5nr_process_*_v3..v6`），并非某个开源项目的编译产物。

但**同类的 caller shim 在多个公开项目里都有**，说明这是社区的通用做法而非某一家的发明：

- [Zonnery/dlss5-nr-player](https://github.com/Zonnery/dlss5-nr-player)（caller/nvngx.dll）
- [DaniilSokolyuk/video2dlssnr](https://github.com/DaniilSokolyuk/video2dlssnr)（nvngx.dll_dlssnr.dll caller-gate shim）
- [Dagherbou/OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR)
- [xenmods/DLSSNR-Cost-Scaler](https://github.com/xenmods/DLSSNR-Cost-Scaler)（DX12 proxy）

### 建议

合规路线只剩一条：**申请 DLSS 5 SDK 访问**（NVIDIA Developer Program）。它会同时解决这次遇到的两件事 —— 更新的 NGX 客户端（够得到 feature 18 的 API 版本）和一个被认可的 project id。

在拿到之前，`[DLSS5]` 功能无法在合规前提下推进。已经落地的资产（`models/dlss5/` 目录、运行时、打包条件化、PoC 程序）都会保留，拿到 SDK 后可直接继续。

---

## 8. 第 7 节的结论是错的：根因是显卡驱动（2026-09-12 复测）

第 7 节把失败归给「API 版本上限 + caller shim」。两条都不是根因。给 probe 装上
`NVSDK_NGX_LoggingInfo` 的日志回调重跑一次，NGX 直接给出了自己的说法：

```
[NVSDK_NGX_CreateFeature_Validate:661] error: required feature is not supported
by NGX runtime, please update display driver
```

更有说服力的是启动时的 snippet 扫描清单：

```
nvngx_dlvdenoiser / dlisp / dlresolve / dlssg / deepdvc / dlssd / truehdr / latewarp / vsr
```

**没有 dlssnr。** 本机驱动 581.57（`_nvngx.dll` 32.0.15.8157，2025-10-10）的 NGX 核心
里就没有 feature 18 这一项，所以 `CreateFeature_Validate` 在**尚未加载**
`nvngx_dlssnr.dll` 之前就拒绝了 —— 那 158 MB 的 snippet 从头到尾没被读过一次。
`FAIL_OutOfDate` 的字面意思就是「过期」，第 7 节却把它读成了「API 版本不匹配」。

### 随之失效的推论

| 第 7 节的说法 | 复测后 |
|---|---|
| app id 0 过不了校验 | 本来就不是问题（Init 一直 Success），这点第 7 节说对了 |
| 静态库 0x15 上限挡住了 feature 18 | 无关：核心在版本协商之前就拒了 |
| 必须要 caller shim | **未经检验**。核心连 feature 18 都不认，shim 有没有用测不出来 |
| 只能申请 DLSS 5 SDK | 不成立：缺的是驱动，不是 SDK |

参考实现（`neuroframe_engine.dll`）扫 API `0x13..0x20`，也印证了这一点 —— 更新的核心
才会暴露更高的 API 版本。它能跑，首先是因为跑它的机器驱动够新。

### 下一步

1. 把驱动升到 **616.64 或更新**（该版本起随驱动安装 Neural Rendering 运行时）。
2. 原样重跑 probe：
   ```
   models\dlss5\native\build\dlssnr_probe.exe models\dlss5\runtime %TEMP%\ngxprobe 18
   ```
   看 snippet 扫描清单里是否出现 `nvngx_dlssnr.dll`。出现了，就说明核心认这个 feature。
3. 只有在第 2 步之后，「app id 0 够不够」和「要不要 caller shim」这两个问题才第一次
   真正可测。在此之前谈它们都是猜。

升驱动对现有功能的风险：`nvngx_vsr.dll` / `nvngx_truehdr.dll` 是我们自己随包分发的
snippet，不依赖驱动版本，RTX VSR 与 TrueHDR 链路不受影响。

### probe 的改动

`dlssnr_probe_main.cpp` 现在通过 `featureInfo.LoggingInfo` 注册一个 VERBOSE 级回调，
把 NGX 的每一行日志原样打到 stdout。裸的 `NVSDK_NGX_Result` 只能靠猜，NGX 自己的日志
是直说的 —— 这次的教训就在这里。

---

## 9. 升驱动之后：真正的阻碍是那个 DLL 被改过（2026-09-12）

驱动从 581.57 升到 **616.92**，第 8 节的预测得到验证：错误码从
`0xBAD0000C`（FAIL_OutOfDate，核心不认识 feature 18）变成了
`0xBAD0000B`（FAIL_UnableToInitializeFeature）—— 核心现在认识这个 feature，
并且**真的去加载** `nvngx_dlssnr.dll` 了。

### 对照组先行

同一个 probe、同样 app id 0，只换成 feature 16 + `models/rtx_vsr/runtime`：

```
CreateFeature(feature 16)          0x00000001  Success
```

所以 probe、app id、init 路径、驱动都没问题。

### feature 18 的新错误

```
NGXLoadAndValidateSnippet:1586] nvLoadSignedLibraryW() failed on snippet
  '...nvngx_dlssnr.dll' missing or corrupted - ERROR_MOD_NOT_FOUND
NGXLoadMetaDataViaGetFileVersionInfo:1412] swscanf_s() failed on snippet ...
NGXSecureLoadFeature:1286] warning: ModuleName - nvngx_dlssnr.dll doesn't exist
  in any of the search paths!
```

`nvLoadSignedLibraryW` —— NGX 只加载**带 NVIDIA 签名**的 snippet。

### 证据：签名被从文件尾部截掉了

PE 证书表（data directory 第 4 项，存的是文件偏移而非 RVA）：

| 文件 | 证书表偏移 + 大小 | 文件大小 | |
|---|---|---|---|
| `nvngx_vsr.dll` | 19,129,856 + 10,288 | 19,140,144 | ✅ 刚好到尾 |
| `nvngx_truehdr.dll` | 3,945,984 + 9,768 | 3,955,752 | ✅ 刚好到尾 |
| `nvngx_dlssnr.dll` | 165,830,144 + 10,352 | **165,830,144** | ❌ 指向文件之外 |

`Get-AuthenticodeSignature`：VSR / TrueHDR = `Valid`（`CN=NVIDIA Corporation`），
dlssnr = `NotSigned`。

**从来没签过名的文件根本不会有这个表项。** 表项还在、字节没了，只能是
签完之后被删的。也就是说：`reference/DLSS.5.Visual.Enhancer.v8.0/` 里的那份
**不是 NVIDIA 的原版运行时**。

### 这推翻了我之前写的两句话

1. `get_dlss5.txt` 原本写着该发布包“bundles NVIDIA's unmodified runtime”
   —— 错的，已改。
2. 转分发分析建立在“这是 NVIDIA 未修改的 object code”上。条款 1(c)+2 允许转分发
   NVIDIA 的 object code，不包括被修改过的 NVIDIA 二进制。

参考实现那两件“多出来的事”到此也自洽了：自己加载 `_nvngx.dll` + caller shim，
合起来就是绕开 `nvLoadSignedLibraryW` 这道签名校验。

### 落地的防护

`build_exe.py` 的 `verify_dlss5_runtime()` 现在会检查签名，未签名的
`nvngx_dlssnr.dll` 直接 fail，不会进发布包。

### 剩下的路

申请 DLSS 5 SDK（NVIDIA Developer Program）拿到**带签名的** `nvngx_dlssnr.dll`。
其他一切已经就位：驱动够新、显卡对（Blackwell2 = RTX 50）、app id 0 能过、
probe 可复用、打包条件化已做、驱动版本检查已接入 UI。

### 顺带做的：驱动版本检查

每个 `nvngx_*.dll` 都在 Win32 版本资源的自定义键里声明了自己的要求：

| snippet | NGXMinimumDriverVersion | NGXGpuArchitecture |
|---|---|---|
| `nvngx_vsr.dll` | 550.00 | Turing（RTX 20+）|
| `nvngx_truehdr.dll` | 550.00 | Turing（RTX 20+）|
| `nvngx_dlssnr.dll` | 615.00 | Blackwell2（RTX 50）|

`utils/ngx_requirements.py` 直接读这些键，**一个数字都没写死**：换一份更新的
runtime 进去，检查自动跟着变。`ui/widgets/ngx_driver_notice.py` 在 SuperRes 实时
对话框和离线页面上显示警告，满足要求时完全不出现。

两个细节：

* 驱动版本比较要按数值，不能按字符串 —— `"581.57" > "615.00"` 字符串上成立。
* `nvngx_dlssnr.dll` 报的值长度越过了结尾符，天真地读会得到 `"615.00 "`，
  之后每一次比较都会失败。两点都有回归测试盯着。

nvidia-smi 拿不到驱动版本时一律不报警：宁可漏报，也不能把一个本来能用的
功能说成不能用。
