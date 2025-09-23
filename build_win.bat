@echo off
setlocal enableextensions
pushd "%~dp0"

set "ENTRYPOINT=product_research_app\web_app.py"
set "NAME=EcomTestingApp"
set "DATA_STATIC=product_research_app\static;product_research_app\static"
set "DATA_TEMPLATES=product_research_app\templates;product_research_app\templates"

if not exist "product_research_app\static" (
  echo No se encontro el directorio requerido product_research_app\static.
  popd
  endlocal
  exit /b 1
)

set "EXTRA_DATAS=--add-data \"%DATA_STATIC%\""
if exist "product_research_app\templates" (
  set "EXTRA_DATAS=%EXTRA_DATAS% --add-data \"%DATA_TEMPLATES%\""
)

set "PYTHON=py"
where py >nul 2>&1
if errorlevel 1 set "PYTHON=python"

%PYTHON% -m pip install --upgrade pyinstaller

%PYTHON% -m PyInstaller --noconfirm --clean --name "%NAME%" --windowed %EXTRA_DATAS% "%ENTRYPOINT%"

popd
endlocal
