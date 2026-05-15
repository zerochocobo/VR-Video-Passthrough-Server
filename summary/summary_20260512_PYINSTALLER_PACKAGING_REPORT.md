# PyInstaller 打包问题专题（VR_Video_Toolbox + ptserver-server）

日期：2026-05-12
范围：`build_exe.bat` 的两个 onedir 构建合并产物 `dist\VR_Video_Toolbox\`

---

## 一、最终症状

运行 `dist\VR_Video_Toolbox\VR_Video_Toolbox.exe` 立即崩溃：

```
Traceback (most recent call last):
  File "app.py", line 53, in <module>
ImportError: DLL load failed while importing QtCore: 找不到指定的程序。
```

`找不到指定的程序` 对应 Win32 错误码 **`ERROR_PROC_NOT_FOUND`（127）**，是某个被加载的 DLL 引用了另一个 DLL 里**不存在的导出符号**。它和"找不到模块（126）"不是一回事，思路必须区分。

---

## 二、根本原因（最终定位）

| 项 | 内容 |
|---|---|
| **罪魁文件** | `dist\VR_Video_Toolbox\_internal\icuuc.dll`（1896960 B）、`icudt58.dll`（26216448 B） |
| **来源** | `C:\ProgramData\Anaconda3\Library\bin\` —— 系统 PATH 上的 Anaconda 安装 |
| **谁带进来的** | PyInstaller 的二进制依赖扫描器（Analysis 阶段）顺着 PATH 找到 icuuc.dll，把它当作某个 C 扩展的传递依赖打包了 |
| **为什么 Qt6Core 加载会失败** | PySide6 6.11 的 Qt6Core.dll 用 `U_DISABLE_RENAMING` 构建，导入表里全是**无后缀**的 ICU 函数名（`ucnv_open` / `ucnv_close` / `ucnv_reset` …）。Anaconda 这份 ICU 58 的导出**只有 `_58` 后缀版本**（`ucnv_open_58`），完全没有无后缀名 → procedure not found |
| **为什么开发环境（venv）能跑** | venv 的 PySide6 本身**不带 icuuc.dll**，Qt6Core 在 venv 中加载时回落到 `C:\Windows\System32\icuuc.dll`（Win10 1903+ 内置 ICU 的 stub forwarder，转发到 `icu.dll`，提供无后缀符号） |
| **为什么打包后会触发** | PyInstaller 把 Anaconda 的 ICU 58 复制进 `_internal\`，然后 `app.py:_prepare_qt_dll_paths` 把 `_internal\` 加进 `AddDllDirectory`。Windows DLL 解析时这个用户目录优先级**高于 System32**，于是用错了 ICU |

### 关键证据复现命令

确认 Qt6Core 的 ICU 导入名：

```bash
uv run python -c "
import pefile
pe = pefile.PE('dist/VR_Video_Toolbox/_internal/PySide6/Qt6Core.dll', fast_load=False)
for e in pe.DIRECTORY_ENTRY_IMPORT:
    if e.dll.decode().startswith('icu'):
        for imp in e.imports:
            print(imp.name.decode())
"
# 输出 ucnv_open / ucnv_close / ucnv_reset ... 等无后缀名
```

确认 Anaconda ICU 只有 `_58` 后缀：

```bash
uv run python -c "
import pefile
pe = pefile.PE('dist/VR_Video_Toolbox/_internal/icuuc.dll', fast_load=False)
names = [e.name.decode() for e in pe.DIRECTORY_ENTRY_EXPORT.symbols if e.name]
print('ucnv_open unsuffixed:', any(n=='ucnv_open' for n in names))
print('_58 suffix exists:', any('_58' in n for n in names))
"
# 输出：unsuffixed False / _58 True
```

PyInstaller 自己的 toc 也供认不讳：

```
D:\p\PTServer\build\VR_Video_Toolbox\Analysis-00.toc:
  ('icuuc.dll', 'C:\\ProgramData\\Anaconda3\\Library\\bin\\icuuc.dll', ...)
  ('icudt58.dll', 'C:\\ProgramData\\Anaconda3\\Library\\bin\\icudt58.dll', ...)
```

---

## 三、修复方案（已落地）

`build_exe.bat` 在 robocopy 合并完两份 `_internal` 之后，显式删除 Anaconda 风格的 ICU 残留：

```bat
for %%I in (icuuc.dll icudt58.dll icuin.dll icuin58.dll icuuc58.dll icudata.dll) do (
    if exist "%DIST_DIR%\_internal\%%I" del /q "%DIST_DIR%\_internal\%%I"
)
```

删除后 Qt6Core 加载时找不到 `_internal\icuuc.dll`，沿 DLL 搜索顺序回落到 `C:\Windows\System32\icuuc.dll`，导出符号匹配，加载成功。

**验证**：把这两个 dll 重命名为 .bak 后用 ctypes 加载 `dist\..._internal\PySide6\Qt6Core.dll` 返回 OK；直接启动 `VR_Video_Toolbox.exe`，进程常驻 ~114MB，不再立即退出。

---

## 四、过程中走过的弯路（避免下次复发）

调查过程中先后被三个**看起来很像、实际无关**的"伪线索"误导，记下来：

### 弯路 1：以为是两份 `_internal` 合并失败

最初观察到 toolbox 的 `MSVCP140.dll` 是 549176 B（PySide6 自带），server 的是 590632 B（onnxruntime/cupy 自带的更新版），且 toolbox 那份**没有**被 server 的覆盖。

排查得知：原 `xcopy /Y` **不会覆盖只读文件**，PyInstaller 经常把打包出的 DLL 设成只读 → 合并不完整。

修复：换 `robocopy /E /IS /IT`（或 `xcopy /R /Y`）。

→ 这是真问题，但**不是 QtCore ImportError 的原因**。VC 运行时高版本向后兼容，包内 549K 和 590K 那两份其实都能让 Qt 跑。

### 弯路 2：以为根级 Qt DLL 重复

原脚本第 91-94 行把 `_internal\PySide6\Qt6*.dll` / `Qt*.pyd` / `pyside6*.dll` 复制到 `_internal\` 根目录，目的是规避"PySide6 子目录里 .pyd 找不到同目录依赖"的传闻问题。结果在 `_internal\` 同时存在 `Qt6Core.dll` 两份。

实际情况：

- `app.py:_prepare_qt_dll_paths` 已经 `add_dll_directory(_internal/PySide6)`，PySide6 子目录在 DLL 搜索路径里，无需复制
- 复制出来的两份内容完全相同，本身并不会导致 procedure not found
- 但留着会造成歧义和体积浪费

修复：删掉这 4 行复制 + 加上"关键 DLL 唯一性"校验（`Qt6Core.dll`、`pyside6.abi3.dll`、`shiboken6.abi3.dll`、`python312.dll` 等出现 >1 份就 fail build）。VC runtime 不在校验列表，因为 PySide6/、shiboken6/、根目录各放一份是 PyInstaller 的正常布局。

→ 也是真问题（卫生层面），但**不是 QtCore ImportError 的原因**。

### 弯路 3：用 `uv run python` 在仓库根目录用 ctypes 加载 dist 里的 Qt6Core 做测试

测试逻辑混乱：

| 测试 | 结果 |
|---|---|
| `cwd=仓库根`，`add_dll_directory(_internal/PySide6)` 后 `WinDLL(...Qt6Core.dll)` | **OK** |
| `cwd=仓库根`，`add_dll_directory(_internal)` + `add_dll_directory(_internal/PySide6)` | FAIL 127 |
| 先 `WinDLL(_internal/MSVCP140.dll)` 再 `WinDLL(Qt6Core.dll)` | OK |
| `cwd=dist/.../_internal/PySide6`，`WinDLL('./Qt6Core.dll')`（裸 python） | OK |

差异看似随机，让人以为是 `_internal\MSVCP140.dll`（590K）和 PySide6\msvcp140.dll（549K）的版本冲突。其实是：

- `add_dll_directory(_internal)` 让 ICU 58 进了搜索路径
- 不加 `_internal`，Windows 直接回落到 System32，ICU 正常
- 先 `WinDLL(MSVCP140.dll)` "正好"让 Windows 解析过一次某中间链路，但和 ICU 无关；后续 Qt6Core 的 icuuc 仍走 System32

教训：**在 venv 环境里用 ctypes 测试打包产物时，venv 自己也会污染搜索路径**（PySide6 的 `__init__.py` 改 PATH、可能 preload shiboken6 等），结果**不能直接迁移**到真实 frozen exe 的运行场景。

定位 procedure not found 的正确顺序：

1. 用 `pefile` 看出问题 DLL 的导入表，列出它依赖谁
2. 对每个依赖项，定位 frozen 包内实际拿到哪一份（路径 + 大小 + 编译时间 + 关键 export 名称）
3. 与开发环境实际加载到的同名 DLL 对比（用 `procmon` 或 `Process Explorer`，或 `import ctypes; print(ctypes.WinDLL(name)._handle)` 后再去查模块路径）
4. 找出"开发环境里能跑，frozen 里不能跑"的真正分歧点 —— 通常是某个被 PyInstaller **从 PATH 上意外吸入**的旧版 DLL

---

## 五、当前 `build_exe.bat` 防御层（已落地）

按构建顺序：

1. **第 46-62 行**：先构建 `VR_Video_Toolbox`（UI），`--collect-binaries/data PySide6`、`--collect-binaries shiboken6`
2. **第 65-81 行**：再构建 `ptserver-server`（推理后端），`--collect-all onnxruntime/cupy/cupy_backends/pynvvideocodec`
3. **第 93-96 行**：用 `robocopy /E /IS /IT` 把 server `_internal` 合并进 toolbox `_internal`（强制覆盖只读文件，server 的较新 VC runtime 赢；后续 `cmd /c "exit /b 0"` 重置 ERRORLEVEL）
4. **第 106-108 行**：删 `_internal\` 根目录下任何 Anaconda 风格 ICU（`icuuc.dll` / `icudt58.dll` / `icuin*.dll` / `icudata.dll`）—— **本次修复的核心**
5. **第 115-122 行**：唯一性校验 `Qt6Core.dll` / `Qt6Gui.dll` / `Qt6Widgets.dll` / `pyside6.abi3.dll` / `shiboken6.abi3.dll` / `python312.dll` / `python3.dll`，>1 份即 fail
6. **第 126-131 行**：清掉外部 `resources\` 目录（PyInstaller 已把 `--add-data` 进 `_internal\resources\`，外面那份是冗余）

---

## 六、永久收尾建议（暂未实施）

下面这些是更彻底的方案，当前问题已修复但可以下次需要重构打包逻辑时参考：

1. **改用 PyInstaller 单 spec MULTIPACKAGE / `MERGE()`**
   官方推荐的"两个 exe 共享 `_internal`"做法。一个 `.spec` 里两个 `Analysis` + `MERGE(...)` + 两个 `EXE` + 一个 `COLLECT`，PyInstaller 自己保证共享依赖只出现一份，且解决 Analysis 时的 PATH 污染问题（可以在 spec 里精确控制 `binaries` 列表，把 Anaconda 路径排除掉）
2. **把 server 放到 `server/` 子目录而不是合并**
   `dist/VR_Video_Toolbox/server/ptserver-server.exe` + `dist/VR_Video_Toolbox/server/_internal/`。
   `ui/services/process_helpers.py:server_command()` 改成 `Path(sys.executable).parent / "server" / "ptserver-server.exe"`。彻底消除合并风险，代价是磁盘体积变大（python312.dll 等共享文件要存两份）
3. **构建时清理环境 PATH，避免 Anaconda 干扰**
   在 `build_exe.bat` 顶部加 `set PATH=%SystemRoot%\System32;%SystemRoot%;%LOCALAPPDATA%\uv\python\...` 之类的最小 PATH，让 PyInstaller 的二进制扫描器扫不到 Anaconda 的 DLL
4. **`--exclude-binaries`（如果未来有同类杂质）**
   PyInstaller 命令行没有直接排除某 .dll 的开关，但 spec 文件里可以在 Analysis 后过滤 `a.binaries`：
   ```python
   a.binaries = [b for b in a.binaries if not b[0].lower().startswith(('icuuc', 'icudt58'))]
   ```

---

## 七、快速诊断 cheat-sheet

下次遇到 `DLL load failed while importing X: 找不到指定的程序`：

```bash
# 1. 找 .pyd 实际位置
find "dist/.../" -iname "X.pyd"

# 2. 看 .pyd / 配套主 DLL 依赖什么
uv run python -c "import pefile; pe=pefile.PE('PATH'); [print(e.dll.decode()) for e in pe.DIRECTORY_ENTRY_IMPORT]"

# 3. 对每个怀疑对象，检查 frozen 内的同名 DLL：
#    - 路径
#    - 大小 / mtime
#    - 编译时间 (TimeDateStamp)
#    - 关键导出名（U_DISABLE_RENAMING / 版本后缀 / 是否带 _58 _73 _75 之类）

# 4. 对照 dev 环境同名 DLL 的来源（venv? System32? Anaconda?）

# 5. 看 PyInstaller 的 build\<name>\Analysis-00.toc，搜可疑 DLL，可知 PyInstaller 从哪个路径捞的
```

错误码速查：

| HRESULT | 名字 | 含义 |
|---|---|---|
| 126 | `ERROR_MOD_NOT_FOUND` | 找不到 DLL 本身（路径错） |
| 127 | `ERROR_PROC_NOT_FOUND` | DLL 找到了，但里面缺导出符号（版本错） |
| 193 | `ERROR_BAD_EXE_FORMAT` | 32/64 位不匹配 |
| 1114 | `ERROR_DLL_INIT_FAILED` | DLL 自己的 `DllMain` 返回 FALSE |

中文 Windows 显示：
- "找不到指定的模块" = 126
- "找不到指定的程序" = 127 ← 本次就是这个
