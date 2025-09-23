@echo off
setlocal enableextensions
pushd "%~dp0"

REM ======== CONFIG ========
set "ENTRYPOINT=main.py"   REM <<< CAMBIAR AQUÍ si tu entrypoint no es main.py
set "URL=http://127.0.0.1:8000"
REM ========================

if not exist "logs" mkdir "logs"

REM Crear venv si no existe (necesitamos pythonw en el venv)
if not exist ".venv\Scripts\pythonw.exe" (
  where py >nul 2>&1 && (py -3 -m venv .venv) || (python -m venv .venv) || goto :no_python
)

REM Instalar deps si hay requirements (primera vez, en background rápido)
if exist requirements.txt (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\pip.exe" install -r requirements.txt >nul
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Ejecutar servidor con pythonw y redirigir salidas a logs (sin consola)
start "" /min powershell -NoProfile -WindowStyle Hidden -Command ^
  "$p=(Resolve-Path '.\.venv\Scripts\pythonw.exe'); $wd=(Get-Location).Path; " ^
  "Start-Process -FilePath $p -ArgumentList '%ENTRYPOINT%' -WorkingDirectory $wd -RedirectStandardOutput 'logs\run.out.log' -RedirectStandardError 'logs\run.err.log'; " ^
  "$u='%URL%'; for($i=0;$i -lt 60;$i++){ try{ $r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2; if($r.StatusCode -ge 200 -and $r.StatusCode -lt 500){ Start-Process $u; break } }catch{}; Start-Sleep -Seconds 1 }"

popd
endlocal
exit /b 0

:no_python
echo [ERROR] No se encontro Python 3.10+ para crear el entorno.
exit /b 1
