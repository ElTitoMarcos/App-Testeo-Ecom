@echo off
setlocal enableextensions
REM Ejecutar desde la carpeta del script
cd /d "%~dp0"

REM ======== CONFIG ========
set "MODULE=product_research_app.web_app"  REM <<< Cambia a product_research_app.main si prefieres la CLI
set "HOST=127.0.0.1"
set "PORT=8000"
REM ========================

echo [INFO] Directorio: %CD%
if not exist "logs" mkdir "logs"

REM Crear venv si no existe (requiere Python 3.x)
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creando entorno virtual .venv ...
  where py >nul 2>&1 && (py -3 -m venv .venv) || (python -m venv .venv)
  if errorlevel 1 (
    echo [ERROR] No se pudo crear el venv. Asegura Python 3.10+ instalado.
    pause & exit /b 1
  )
)

REM Activar venv
call ".venv\Scripts\activate.bat" || (
  echo [ERROR] No se pudo activar el venv.
  pause & exit /b 1
)

echo [INFO] Python:
python -V || (echo [ERROR] Python no accesible en el venv & pause & exit /b 1)

REM Instalar deps si existe requirements.txt
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

echo.
echo ============================
echo Iniciando %MODULE% en http://%HOST%:%PORT%
echo (esta ventana quedara abierta)
echo ============================
echo.

REM Ejecutar modulo como paquete (-m) para respetar imports relativos
python -m %MODULE% --host %HOST% --port %PORT%
set ERR=%ERRORLEVEL%

echo.
if not "%ERR%"=="0" (
  echo [ERROR] El proceso termino con codigo %ERR%.
) else (
  echo [OK] Proceso finalizado.
)
echo.
pause
endlocal
