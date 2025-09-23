@echo off
setlocal enableextensions
REM Ejecutar desde la carpeta del script
cd /d "%~dp0"

REM ======== CONFIG ========
set "ENTRYPOINT=main.py"   REM <<< CAMBIAR si tu entrada no es main.py
set "URL=http://127.0.0.1:8000"
set "TIMEOUT_SECS=60"
REM ========================

echo [INFO] Directorio: %CD%
if not exist "logs" mkdir "logs"

REM Crear venv si no existe
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creando entorno virtual .venv ...
  where py >nul 2>&1 && (py -3 -m venv .venv) || (python -m venv .venv)
  if errorlevel 1 (
    echo [ERROR] No se pudo crear el venv. Asegura Python 3.10+ instalado.
    pause & exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] No se pudo activar el venv.
  pause & exit /b 1
)

echo [INFO] Python:
python -V
if errorlevel 1 (
  echo [ERROR] Python no accesible.
  pause & exit /b 1
)

REM Instalar deps si hay requirements.txt
if exist requirements.txt (
  echo [INFO] Instalando dependencias (si faltan)...
  python -m pip install --upgrade pip >nul
  pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Fallo instalando requirements.txt
    pause & exit /b 1
  )
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo [INFO] Lanzando %ENTRYPOINT% ...
start "" cmd /c ^
  "python "%ENTRYPOINT%" 1>>"logs\run.out.log" 2>>"logs\run.err.log""

REM Espera activa a que el servidor responda en URL
echo [INFO] Esperando a que %URL% responda (timeout %TIMEOUT_SECS%s)...
for /l %%I in (1,1,%TIMEOUT_SECS%) do (
  powershell -NoProfile -Command "$u='%URL%';try{$r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2; if($r.StatusCode -ge 100){exit 0}}catch{exit 1}" >nul
  if "%ERRORLEVEL%"=="0" (
    echo [OK] El servidor responde en %URL%
    goto :end
  )
  >nul ping -n 2 127.0.0.1
)

echo [FAIL] No hubo respuesta a tiempo. Revisa logs\run.err.log

:end
echo.
echo [FIN] Pulsa una tecla para cerrar...
pause >nul
endlocal
