@echo off
setlocal enableextensions
pushd "%~dp0"

if not exist "logs" mkdir "logs"

if not exist ".venv" (
  where py >nul 2>&1 && (py -3.11 -m venv .venv || py -3 -m venv .venv) || (python -m venv .venv)
)

call ".venv\Scripts\activate.bat"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

start "" /min ".venv\Scripts\pythonw.exe" main.py 1>>"logs\run.out.log" 2>>"logs\run.err.log"

start "" powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8000'"

popd
endlocal
