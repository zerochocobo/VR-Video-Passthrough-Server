@echo off
setlocal
cd /d "%~dp0"

rem Local build dependency: ONNX Runtime CUDAExecutionProvider needs cuDNN 9.
rem Keep this path here so running build_exe.bat is enough on this build PC.
rem Override PT_CUDNN_BIN before running if your cuDNN is installed elsewhere.
if not defined PT_CUDNN_BIN (
    set "PT_CUDNN_BIN=C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64"
)

if not exist "%PT_CUDNN_BIN%\cudnn64_9.dll" (
    echo Missing cudnn64_9.dll under PT_CUDNN_BIN:
    echo   %PT_CUDNN_BIN%
    echo Set PT_CUDNN_BIN to your cuDNN v9 bin directory and retry.
    goto :fail
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
