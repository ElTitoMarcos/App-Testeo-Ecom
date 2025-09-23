@echo off
setlocal enableextensions
REM Ejecutar siempre desde la carpeta del script
cd /d "%~dp0"

REM ======== CONFIG ========
set "ENTRYPOINT=main.py"   REM <<< CAMBIAR SI TU ENTRADA NO ES main.py
REM ========================

REM Crear venv si no existe (requiere Python 3.x instalado)
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>&1 && (py -3 -m venv .venv) || (python -m venv .venv) || (
    echo [ERROR] No se encontro Python 3.x para crear el entorno virtual.
    echo Instala Python 3.10+ y vuelve a ejecutar este .bat
    pause & exit /b 1
  )
)

REM Activar venv
call ".venv\Scripts\activate.bat"

REM Instalar deps si hay requirements.txt
if exist requirements.txt (
  python -m pip install --upgrade pip >nul
  pip install -r requirements.txt
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================
echo Iniciando %ENTRYPOINT% ...
echo (cierra esta ventana para detener el servidor)
echo ============================
echo.

python "%ENTRYPOINT%"
set ERR=%ERRORLEVEL%

echo.
if not "%ERR%"=="0" (
  echo [ERROR] El proceso retorno codigo %ERR%.
) else (
  echo [OK] Proceso finalizado.
)
echo.
pause
endlocal
