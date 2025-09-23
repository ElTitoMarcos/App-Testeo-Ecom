@echo off

setlocal enableextensions

cd /d "%~dp0"



REM ======== CONFIG ========

set "ENTRYPOINT=main.py"   REM <<< cambia si tu entrada no es main.py

set "TEST_URL=http://127.0.0.1:8000"

set "TEST_TIMEOUT=60"

REM ========================



echo [TEST] Preparando entorno Windows...

if not exist ".venv\Scripts\python.exe" (

  where py >nul 2>&1 && (py -3 -m venv .venv) || (python -m venv .venv) || (

    echo [ERROR] No se encontro Python 3.x para crear el venv.

    pause & exit /b 1

  )

)



call ".venv\Scripts\activate.bat"



if exist requirements.txt (

  python -m pip install --upgrade pip >nul

  pip install -r requirements.txt

)



set "PYTHONUTF8=1"

set "PYTHONIOENCODING=utf-8"

echo [TEST] Ejecutando smoke test...

python -X utf8 -u smoketest.py --entrypoint "%ENTRYPOINT%" --url "%TEST_URL%" --timeout %TEST_TIMEOUT% --python "%CD%\.venv\Scripts\python.exe"

set ERR=%ERRORLEVEL%



echo.

if "%ERR%"=="0" (

  echo [OK] Smoke test Windows PASADO.

) else (

  echo [FAIL] Smoke test Windows FALLO (codigo %ERR%). Revisa logs\test_server.err.log

)

echo.

pause

endlocal

