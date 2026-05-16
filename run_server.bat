@echo off
REM VR Passthrough Server startup.
REM Runtime defaults live in config.py. Override PT_* here only for temporary diagnostics.

REM Development runtime dependency: ONNX Runtime CUDAExecutionProvider needs cuDNN 9.
REM Packaged builds use bundled DLLs under _internal; this is only for direct Python runs.
if not defined PT_CUDNN_BIN (
    set "PT_CUDNN_BIN=C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64"
)

if exist "%PT_CUDNN_BIN%\cudnn64_9.dll" (
    set "PATH=%PT_CUDNN_BIN%;%PATH%"
)

uv run python main.py
exit /b %ERRORLEVEL%
