@echo off
:: build_exe.bat — alternativa en CMD
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

where py.exe >nul 2>nul || (echo Necesitas Python 3.11+ para COMPILAR. & pause & exit /b 1)

if not exist ".venv_build\Scripts\python.exe" (
  py -3 -m venv .venv_build || (echo Error creando .venv_build & pause & exit /b 1)
)
set "PY=%CD%\.venv_build\Scripts\python.exe"
set "PIP=%CD%\.venv_build\Scripts\pip.exe"

"%PY%" -m pip install --upgrade pip wheel setuptools
"%PIP%" install pyinstaller==6.*
if exist "requirements.txt" "%PIP%" install -r requirements.txt

if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

set ICON=
if exist "app.ico" set ICON=--icon app.ico

set DATA_ARGS=
:: Si tienes recursos FUERA del paquete, descomenta:
:: set DATA_ARGS=--add-data "static;static" --add-data "templates;templates"

"%PY%" -m PyInstaller --noconfirm --onefile --name "EcomTestingApp" %ICON% ^
  --collect-all product_research_app ^
  --collect-all uvicorn --collect-all starlette ^
  --collect-all jinja2 --collect-all markupsafe ^
  %DATA_ARGS% ^
  entrypoint_exe.py

if errorlevel 1 (echo Fallo de compilacion & pause & exit /b 1)
echo OK. EXE en .\dist\EcomTestingApp.exe
pause
