@echo off
REM VR Passthrough Server startup.
REM Runtime defaults live in config.py. Override PT_* here only for temporary diagnostics.

uv run python main.py
exit /b %ERRORLEVEL%
