@echo off
setlocal
cd /d "%~dp0"

set "APP_NAME=VR_Video_Passthrough_Server"
set "SERVER_NAME=pt_core"
set "DIST_DIR=dist\%APP_NAME%"
set "ICON=resources\app.ico"

rem Capture uv.exe directory BEFORE we shrink PATH for the build. PyInstaller's
rem binary dependency walker scans PATH and sweeps in unrelated system DLLs
rem (Anaconda ICU, PostgreSQL LIBPQ, Kerberos krb5_64, DirectX SDK ...).
rem A minimal PATH avoids that bloat and previous QtCore ImportError failures.
for /f "delims=" %%U in ('where uv 2^>nul') do if not defined UV_DIR set "UV_DIR=%%~dpU"
for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY_DIR set "PY_DIR=%%~dpP"

if not defined UV_DIR if not defined PY_DIR (
    echo Neither uv.exe nor python.exe found in PATH.
    echo Open a shell where uv or python is on PATH and rerun this script.
    pause
    exit /b 1
)

rem Sanitized PATH: only Windows system dirs + project venv + the tool dirs we
rem captured above. Anything else (Anaconda, Strawberry Perl, PostgreSQL, etc.)
rem is intentionally invisible to PyInstaller's dependency scanner.
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%CD%\.venv\Scripts"
if defined UV_DIR set "PATH=%PATH%;%UV_DIR%"
if defined PY_DIR set "PATH=%PATH%;%PY_DIR%"
if exist "%CD%\.venv\Lib\site-packages\PySide6" set "PATH=%PATH%;%CD%\.venv\Lib\site-packages\PySide6"
if exist "%CD%\.venv\Lib\site-packages\shiboken6" set "PATH=%PATH%;%CD%\.venv\Lib\site-packages\shiboken6"
set "PYTHONPATH=%CD%\packaging;%CD%\packaging\hooks;%PYTHONPATH%"

rem Probe the active python first (covers conda envs, plain venvs, system python).
rem Only fall back to `uv run` when the active python lacks PyInstaller AND uv
rem is available with a project venv. This avoids the case where `where uv`
rem succeeds globally but `uv run` targets an empty/missing project venv while
rem PyInstaller is actually installed in the currently activated conda env.
set "PYI="
python -c "import PyInstaller" >nul 2>nul
if not errorlevel 1 (
    set "PYI=python -m PyInstaller"
) else (
    where uv >nul 2>nul
    if not errorlevel 1 (
        uv run python -c "import PyInstaller" >nul 2>nul
        if not errorlevel 1 set "PYI=uv run pyinstaller"
    )
)
if not defined PYI (
    echo PyInstaller is not installed in the active environment.
    echo Install it first, for example:
    echo   pip install pyinstaller
    echo or:
    echo   uv add --dev pyinstaller
    echo or:
    echo   uv pip install pyinstaller
    pause
    exit /b 1
)
echo Using: %PYI%

if exist "build" rmdir /s /q "build"
if exist "build" (
    echo Failed to remove build directory. Stop running packaged processes and try again.
    goto :fail
)
if exist "dist\%APP_NAME%" rmdir /s /q "dist\%APP_NAME%"
if exist "dist\%APP_NAME%" (
    echo Failed to remove dist\%APP_NAME%. Stop running packaged processes and try again.
    goto :fail
)
if exist "dist\%SERVER_NAME%" rmdir /s /q "dist\%SERVER_NAME%"
if exist "dist\%SERVER_NAME%" (
    echo Failed to remove dist\%SERVER_NAME%. Stop running packaged processes and try again.
    goto :fail
)
if exist "%APP_NAME%.spec" del "%APP_NAME%.spec"
if exist "%SERVER_NAME%.spec" del "%SERVER_NAME%.spec"

%PYI% ^
  --name "%APP_NAME%" ^
  --noconsole ^
  --onedir ^
  --icon "%ICON%" ^
  --additional-hooks-dir "packaging\hooks" ^
  --add-data "resources;resources" ^
  --add-data "ui\app_metadata.json;ui" ^
  --add-data "ui\translations;ui\translations" ^
  --add-data "ui\styles;ui\styles" ^
  --collect-binaries PySide6 ^
  --collect-binaries shiboken6 ^
  --collect-data PySide6 ^
  --hidden-import PySide6.QtCore ^
  --hidden-import PySide6.QtGui ^
  --hidden-import PySide6.QtWidgets ^
  --hidden-import shiboken6 ^
  ui\app.py
if errorlevel 1 goto :fail

%PYI% ^
  --name "%SERVER_NAME%" ^
  --console ^
  --onedir ^
  --icon "%ICON%" ^
  --add-data "resources;resources" ^
  --add-data "tools;tools" ^
  --add-data "offline;offline" ^
  --hidden-import offline.convert ^
  --collect-submodules pipeline ^
  --collect-submodules http_app ^
  --collect-submodules dlna ^
  --collect-submodules utils ^
  --hidden-import cupy_backends.cuda._softlink ^
  --collect-all onnxruntime ^
  --collect-all cupy ^
  --collect-all cupy_backends ^
  --collect-submodules cupy ^
  --collect-submodules cupy_backends ^
  --collect-all pynvvideocodec ^
  main.py
if errorlevel 1 goto :fail

if not exist "%DIST_DIR%" mkdir "%DIST_DIR%"
copy /y "dist\%SERVER_NAME%\%SERVER_NAME%.exe" "%DIST_DIR%\" >nul

rem Merge server _internal into toolbox _internal. robocopy is used instead of
rem xcopy because xcopy /Y does NOT overwrite read-only files (PyInstaller sets
rem some DLLs read-only), which silently leaves stale Qt-incompatible VC runtime
rem versions. Server ships newer MSVCP140/VCRUNTIME140 from onnxruntime/cupy and
rem must win the merge; newer VC runtime is backward-compatible for PySide6.
rem robocopy exit codes: 0-3 success, >=4 failure.
robocopy "dist\%SERVER_NAME%\_internal" "%DIST_DIR%\_internal" /E /IS /IT /NFL /NDL /NJH /NJS /NP
if %ERRORLEVEL% GEQ 4 goto :fail
rem Reset ERRORLEVEL so subsequent `if errorlevel 1` checks behave correctly.
cmd /c "exit /b 0"

rem PyNvVideoCodec dynamically loads the driver-matched extension by filename
rem (for example PyNvVideoCodec_130.cp312-win_amd64.pyd) via
rem importlib.util.spec_from_file_location(). PyInstaller sees VersionCheck but
rem misses those dynamic pyd files. Keep the package layout and copy the Python
rem wrappers too, because __init__.py imports decoders\ and transcoder\ after
rem loading the driver-matched extension.
if not exist ".venv\Lib\site-packages\PyNvVideoCodec" (
    echo PyNvVideoCodec package not found under .venv\Lib\site-packages.
    goto :fail
)
robocopy ".venv\Lib\site-packages\PyNvVideoCodec" "%DIST_DIR%\_internal\PyNvVideoCodec" /E /XF *.pyc /XD __pycache__ /NFL /NDL /NJH /NJS /NP
if %ERRORLEVEL% GEQ 4 goto :fail
cmd /c "exit /b 0"

rem PyInstaller's binary dependency walker may pick up stale ICU DLLs from a
rem system-wide Anaconda installation on PATH (icuuc.dll / icudt58.dll). These
rem are ICU 58 with versioned exports only (e.g. ucnv_open_58), while
rem PySide6 6.11's Qt6Core.dll imports unsuffixed names (ucnv_open). The stale
rem ICU shadows the Windows built-in ICU stub (C:\Windows\System32\icuuc.dll)
rem and causes "DLL load failed while importing QtCore" (WinError 127). Remove
rem any Anaconda-origin ICU from the merged distribution so the loader falls
rem through to the system ICU.
for %%I in (icuuc.dll icudt58.dll icuin.dll icuin58.dll icuuc58.dll icudata.dll) do (
    if exist "%DIST_DIR%\_internal\%%I" del /q "%DIST_DIR%\_internal\%%I"
)

rem Verify no duplicated critical DLLs are present. Duplicates of these specific
rem DLLs trigger Windows DLL resolution ambiguity and have caused QtCore
rem ImportError in past builds. VC runtime DLLs (MSVCP140/VCRUNTIME140) are
rem intentionally excluded: PyInstaller legitimately bundles multiple copies in
rem PySide6\, shiboken6\, and the root from independent C-extension wheels.
for %%N in (Qt6Core.dll Qt6Gui.dll Qt6Widgets.dll pyside6.abi3.dll shiboken6.abi3.dll python312.dll python3.dll) do (
    for /f %%C in ('dir /s /b "%DIST_DIR%\_internal\%%N" 2^>nul ^| find /v /c ""') do (
        if %%C GTR 1 (
            echo Duplicated DLL detected: %%N has %%C copies under _internal
            goto :fail
        )
    )
)

rem Resources are bundled as PyInstaller data under _internal\resources. Do not
rem create an external resources directory in the final distribution.
if exist "%DIST_DIR%\resources" rmdir /s /q "%DIST_DIR%\resources"

rem Models are intentionally not bundled. Keep an empty external models
rem directory so release users can add the models they need after unpacking.
if exist "%DIST_DIR%\models" rmdir /s /q "%DIST_DIR%\models"
mkdir "%DIST_DIR%\models" >nul 2>nul

rem Some CuPy wheels probe cuTENSOR via softlink. If the DLLs are installed in
rem CUDA_PATH, copy them beside the other bundled CUDA DLLs. They are optional
rem on machines where the installed CuPy wheel does not ship or need them.
if defined CUDA_PATH (
    for /r "%CUDA_PATH%" %%F in (cuTENSOR.dll cuTENSORMg.dll) do (
        if exist "%%F" copy /y "%%F" "%DIST_DIR%\_internal\" >nul
    )
)

echo.
echo Build complete:
echo   %DIST_DIR%\%APP_NAME%.exe
echo   %DIST_DIR%\%SERVER_NAME%.exe
echo.
echo This is an onedir build. Distribute the whole "%DIST_DIR%" folder, not only the exe.
pause
exit /b 0

:fail
echo.
echo Build failed.
pause
exit /b 1
