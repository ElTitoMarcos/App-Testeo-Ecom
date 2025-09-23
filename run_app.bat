@echo off
setlocal enableextensions
pushd "%~dp0"

REM ======== CONFIG ========
set "ENTRYPOINT=main.py"   REM <<< CAMBIAR AQUÍ si tu entrypoint no es main.py
set "URL=http://127.0.0.1:8000"
REM ========================

if not exist "logs" mkdir "logs"

REM Crear venv si no existe
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>&1 && (py -3 -m venv .venv) || (python -m venv .venv) || (
    echo [ERROR] No se encontro Python 3.10+.
    pause & exit /b 1
  )
)

call ".venv\Scripts\activate.bat"

REM Instalar deps si hay requirements (primera vez)
if exist requirements.txt (
  python -m pip install --upgrade pip >nul
  pip install -r requirements.txt
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Lanzar comprobador de readiness que abre el navegador cuando responda
start "" powershell -NoProfile -WindowStyle Hidden -Command ^
  "$u='%URL%'; for($i=0;$i -lt 60;$i++){ try{ $r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2; if($r.StatusCode -ge 200 -and $r.StatusCode -lt 500){ Start-Process $u; break } }catch{}; Start-Sleep -Seconds 1 }"

echo Iniciando servidor...
python "%ENTRYPOINT%" 1>>"logs\run.out.log" 2>>"logs\run.err.log"
set ERR=%ERRORLEVEL%

echo(
if not "%ERR%"=="0" (
  echo [ERROR] Codigo %ERR%. Revisa logs\run.err.log
) else (
  echo Servidor finalizado.
)
echo(
pause

popd
endlocal
