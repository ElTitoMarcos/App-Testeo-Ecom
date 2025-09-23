# build_exe.ps1 — Compila EXE portable con PyInstaller
# Uso:
#   1) PowerShell como usuario normal
#   2) Set-ExecutionPolicy -Scope Process Bypass
#   3) .\build_exe.ps1
# Resultado: .\dist\EcomTestingApp.exe

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host "== Preparando entorno virtual de build ==" -ForegroundColor Cyan
$venv = Join-Path $ROOT ".venv_build"
if (-not (Test-Path $venv)) {
  py -3 -m venv $venv
}
$py = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

& $py -m pip install --upgrade pip wheel setuptools
& $pip install pyinstaller==6.*

# Instalar deps de la app para discovery de imports
if (Test-Path (Join-Path $ROOT "requirements.txt")) {
  & $pip install -r requirements.txt
}

# Icono opcional
$iconArg = ""
if (Test-Path (Join-Path $ROOT "app.ico")) {
  $iconArg = "--icon app.ico"
}

# Limpieza segura
Remove-Item -Recurse -Force build,dist -ErrorAction SilentlyContinue

# Si tienes recursos FUERA del paquete (p. ej. "static" y "templates"), descomenta y ajusta:
# $dataArgs = '--add-data "static;static" --add-data "templates;templates"'
$dataArgs = ""

# Algunas libs web comunes (si existen, PyInstaller las incluirá; si no existen, no pasa nada)
$collect = @(
  "--collect-all product_research_app",
  "--collect-all uvicorn",
  "--collect-all starlette",
  "--collect-all jinja2",
  "--collect-all markupsafe"
) -join " "

$cmd = @"
"$py" -m PyInstaller --noconfirm --onefile `
  --name "EcomTestingApp" `
  $iconArg `
  $collect `
  $dataArgs `
  entrypoint_exe.py
"@

Write-Host "== Compilando EXE con PyInstaller ==" -ForegroundColor Cyan
Invoke-Expression $cmd

Write-Host "== Listo ==" -ForegroundColor Green
Write-Host "Binario: $(Join-Path $ROOT 'dist\EcomTestingApp.exe')"
