@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if %ERRORLEVEL%==0 (
    uv run python -m PyInstaller --noconfirm --workpath build --distpath dist PT_DLNA_Client_Simulator.spec
) else (
    python -m PyInstaller --noconfirm --workpath build --distpath dist PT_DLNA_Client_Simulator.spec
)
exit /b %ERRORLEVEL%
