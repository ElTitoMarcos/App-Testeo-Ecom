@echo off
setlocal enableextensions
pushd "%~dp0"

if not exist "logs" mkdir "logs"

if not exist ".venv" (
  where py >nul 2>&1 && (py -3.11 -m venv .venv || py -3 -m venv .venv) || (python -m venv .venv)
)

REM Activar venv
call ".venv\Scripts\activate.bat"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Instalar dependencias si fuese necesario
if exist "requirements.txt" (
  python -m pip install --quiet -r requirements.txt 1>>"logs\setup.out.log" 2>>"logs\setup.err.log"
)

for %%I in (".venv\Scripts\pythonw.exe") do set "PYW_PATH=%%~fI"
if not exist "%PYW_PATH%" (
  echo No se encontró pythonw.exe en la venv. >"logs\run.err.log"
  popd
  endlocal
  exit /b 1
)

REM Lanzar sin consola con pythonw.exe y redirigir salidas a logs
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process -FilePath '%PYW_PATH%' -ArgumentList '-m','product_research_app.web_app' -WindowStyle Hidden -RedirectStandardOutput 'logs\\run.out.log' -RedirectStandardError 'logs\\run.err.log'"

REM Abrir navegador tras 3s (ajusta URL si procede)
powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8000'" >nul 2>&1

popd
endlocal
