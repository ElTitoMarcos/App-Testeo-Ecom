# build_exe_auto.ps1 — Compila EXE y auto-instala Python 3.12 si falta
# Uso:
#   1) PowerShell (usuario normal)
#   2) Set-ExecutionPolicy -Scope Process Bypass
#   3) .\build_exe_auto.ps1
# Resultado: .\dist\EcomTestingApp.exe

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

function Get-RealPython {
  # Devuelve ruta a python.exe (no alias de WindowsApps) o $null
  try {
    # 1) py -3
    $pyPath = & py -3 -c "import sys,os;print(sys.executable)" 2>$null
    if ($pyPath -and ($pyPath -notmatch 'WindowsApps')) { return $pyPath }
  } catch {}
  try {
    # 2) where python
    $cands = & where python 2>$null
    foreach ($c in $cands) {
      if ($c -and ($c -notmatch 'WindowsApps')) { return $c }
    }
  } catch {}
  try {
    # 3) Registro InstallPath
    $roots = @(
      'HKCU:\Software\Python\PythonCore',
      'HKLM:\Software\Python\PythonCore',
      'HKLM:\Software\WOW6432Node\Python\PythonCore'
    )
    foreach ($r in $roots) {
      if (Test-Path $r) {
        Get-ChildItem $r -Recurse -EA SilentlyContinue | ForEach-Object {
          try {
            $ip = (Get-ItemProperty $_.PSPath -EA SilentlyContinue).InstallPath
            if ($ip) {
              $exe = Join-Path $ip 'python.exe'
              if (Test-Path $exe) { return $exe }
            }
          } catch {}
        }
      }
    }
  } catch {}
  return $null
}

function Ensure-Python {
  $py = Get-RealPython
  if ($py) { return $py }

  Write-Host ">> Instalando Python 3.12 (winget)..." -ForegroundColor Yellow
  try {
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements --silent
  } catch {}

  $py = Get-RealPython
  if ($py) { return $py }

  Write-Host ">> winget no disponible o falló. Descargando instalador oficial..." -ForegroundColor Yellow
  $arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "arm64" }  # para mayoría será amd64
  if ($arch -ne "amd64") { $arch = "amd64" }  # fuerza amd64 en la mayoría de casos
  $ver = "3.12.6"
  $url = "https://www.python.org/ftp/python/$ver/python-$ver-embed-$arch.zip" # (comprobación de red)
  # Preferimos el instalador clásico, no el embebido, para tener pip/venv listos:
  $url = "https://www.python.org/ftp/python/$ver/python-$ver-$arch.exe"

  $dst = Join-Path $env:TEMP "python-$ver-$arch-installer.exe"
  Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $dst

  Write-Host ">> Instalando Python en modo silencioso..." -ForegroundColor Yellow
  & $dst /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 SimpleInstall=1
  Start-Sleep -Seconds 5

  $py = Get-RealPython
  if (-not $py) {
    throw "No se pudo instalar/detectar Python. Desactiva 'Alias de ejecución de aplicaciones' para python/python3 en Configuración o instala manualmente Python 3.12."
  }
  return $py
}

# 0) Asegurar Python real
$PY = Ensure-Python
Write-Host ">> Python: $PY" -ForegroundColor Cyan
$pyVer = & $PY -c "import sys;print(sys.version.split()[0])"
if ([version]$pyVer -lt [version]"3.11") {
  throw "Se requiere Python >= 3.11 para compilar. Detectado: $pyVer"
}

# 1) Crear venv de build
$venv = Join-Path $ROOT ".venv_build"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
  & $PY -m venv $venv
}
$py = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

# 2) Herramientas + deps de la app
& $py -m pip install --upgrade pip wheel setuptools
if (Test-Path (Join-Path $ROOT "requirements.txt")) {
  & $pip install -r requirements.txt
}
& $pip install pyinstaller==6.*

# 3) Limpiar build previos
Remove-Item -Recurse -Force build,dist -ErrorAction SilentlyContinue

# 4) Icono opcional
$iconArg = ""
if (Test-Path (Join-Path $ROOT "app.ico")) { $iconArg = "--icon app.ico" }

# 5) Datos extra (descomenta si tienes carpetas FUERA del paquete)
$dataArgs = ""
# $dataArgs = '--add-data "static;static" --add-data "templates;templates"'

# 6) Compilar (recoge recursos del paquete)
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

Write-Host ">> Compilando EXE..." -ForegroundColor Cyan
Invoke-Expression $cmd

Write-Host "== Listo ==" -ForegroundColor Green
Write-Host ("Binario: " + (Join-Path $ROOT 'dist\EcomTestingApp.exe'))

Ejecución

Abre PowerShell en la carpeta del proyecto.

Ejecuta:

Set-ExecutionPolicy -Scope Process Bypass
.\build_exe_auto.ps1
