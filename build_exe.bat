@echo off
setlocal
cd /d "%~dp0"

rem ONNX Runtime CUDAExecutionProvider needs cuDNN 9. It is now bundled from the
rem pip nvidia-cudnn-cu12 wheel (see pyproject.toml), so a system cuDNN is
rem OPTIONAL. If PT_CUDNN_BIN points at a valid system cuDNN we prefer it;
rem otherwise the build falls back to the pip cuDNN under .venv\Lib\site-packages\nvidia.
if not defined PT_CUDNN_BIN (
    set "PT_CUDNN_BIN=C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64"
)

if not exist "%PT_CUDNN_BIN%\cudnn64_9.dll" (
    echo PT_CUDNN_BIN has no cudnn64_9.dll - using the bundled pip cuDNN instead.
    set "PT_CUDNN_BIN="
)

echo Using PT_CUDNN_BIN=%PT_CUDNN_BIN%

python build_exe.py %*
if errorlevel 1 goto :fail

echo.
echo Build script completed.
python make_update_package.py
pause
exit /b 0

:fail
echo.
echo Build failed.
pause
exit /b 1
