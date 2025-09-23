@echo off
setlocal enableextensions
REM Ejecutar desde la carpeta del script
cd /d "%~dp0"

echo [INFO] Directorio: %CD%
if not exist "logs" mkdir "logs"

REM ===== Resolver Python del sistema =====
set "PY_CMD="
where py >nul 2>&1  && set "PY_CMD=py -3"
if not defined PY_CMD where python >nul 2>&1 && set "PY_CMD=python"
if not defined PY_CMD (
  echo [ERROR] No se encontro Python 3.x en PATH. Instala Python 3.10+.
  pause & exit /b 1
)

REM ===== Crear venv si no existe =====
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creando entorno virtual .venv ...
  %PY_CMD% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] No se pudo crear .venv
    pause & exit /b 1
  )
)

REM ===== Activar venv =====
call ".venv\Scripts\activate.bat" || (
  echo [ERROR] No se pudo activar .venv
  pause & exit /b 1
)

echo [INFO] Python en venv:
python -V || (echo [ERROR] Python no accesible en .venv & pause & exit /b 1)

REM ===== Instalar deps si procede =====
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

REM ===== Ejecutar el servidor en la MISMA ventana (sin START) =====
echo.
echo ============================
echo Ejecutando: python -m product_research_app.web_app
echo (esta ventana quedara abierta; errores se veran aqui)
echo ============================
echo.

python -m product_research_app.web_app
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
