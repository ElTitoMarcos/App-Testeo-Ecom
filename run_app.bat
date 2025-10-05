@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

:: =========================
:: Ecom Testing App — run_app.bat (todo en uno, sin msstore y sin reinicios)
:: =========================

set "ROOT=%~dp0"
cd /d "%ROOT%"
if not exist "logs" mkdir "logs"
echo ==================================>> "logs\session.log"
echo %DATE% %TIME% — Bootstrap run_app.bat >> "logs\session.log"

:: ---------- Config ----------
set "PY_EXPECTED_MIN=3.11"
set "PY_SETUP_VER=3.12.6"
if not defined PRAPP_HOST set "PRAPP_HOST=127.0.0.1"
if not defined PRAPP_PORT set "PRAPP_PORT=8000"
if not defined PRAPP_AUTO_OPEN set "PRAPP_AUTO_OPEN=1"
if not defined PRAPP_BROWSER_URL set "PRAPP_BROWSER_URL=http://%PRAPP_HOST%:%PRAPP_PORT%/"

:: Detectar arquitectura
set "ARCH=amd64"
if /I "%PROCESSOR_ARCHITECTURE%"=="x86"  set "ARCH=x86"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=amd64"

:: ---------- Función: encontrar Python real (no WindowsApps) ----------
:set_find_python
set "PYEXE="

:: 1) .venv si ya existe
if exist "%ROOT%.venv\Scripts\python.exe" (
  set "PYEXE=%ROOT%.venv\Scripts\python.exe"
  goto :found_python
)

:: 2) py -3
for /f "usebackq delims=" %%I in (`py -3 -c "import sys;print(sys.executable)" 2^>NUL`) do (
  echo %%I | find /I "WindowsApps" >nul || set "PYEXE=%%I"
)
if defined PYEXE goto :found_python

:: 3) Registro — ExecutablePath / InstallPath
for /f "tokens=2,*" %%a in ('reg query "HKCU\Software\Python\PythonCore" /s /v ExecutablePath 2^>nul ^| findstr /I "ExecutablePath"') do (
  echo %%b | find /I "WindowsApps" >nul || if exist "%%b" set "PYEXE=%%b"
)
if defined PYEXE goto :found_python
for /f "tokens=2,*" %%a in ('reg query "HKLM\Software\Python\PythonCore" /s /v ExecutablePath 2^>nul ^| findstr /I "ExecutablePath"') do (
  echo %%b | find /I "WindowsApps" >nul || if exist "%%b" set "PYEXE=%%b"
)
if defined PYEXE goto :found_python

for /f "tokens=2,*" %%a in ('reg query "HKCU\Software\Python\PythonCore" /s /v InstallPath 2^>nul ^| findstr /I "InstallPath"') do (
  if exist "%%b\python.exe" set "PYEXE=%%b\python.exe"
)
if defined PYEXE goto :found_python
for /f "tokens=2,*" %%a in ('reg query "HKLM\Software\Python\PythonCore" /s /v InstallPath 2^>nul ^| findstr /I "InstallPath"') do (
  if exist "%%b\python.exe" set "PYEXE=%%b\python.exe"
)
if defined PYEXE goto :found_python

:: 4) PATH (filtrando WindowsApps)
for /f "delims=" %%P in ('where python.exe 2^>nul') do (
  echo %%P | find /I "WindowsApps" >nul
  if errorlevel 1 set "PYEXE=%%P"
)
if defined PYEXE goto :found_python

:found_python
if defined PYEXE (
  echo [INFO] Python detectado: "%PYEXE%"
  echo [INFO] Python detectado: "%PYEXE%" >> "logs\session.log"
) else (
  echo [WARN] Python no encontrado. Intentando instalar...>> "logs\session.log"
  goto :install_python
)

:: ---------- Verificar versión mínima (robusto con espacios) ----------
set "TMPVERTXT=%TEMP%\pyver_%RANDOM%%RANDOM%.txt"
"%PYEXE%" -V 2>&1 > "%TMPVERTXT%"
set /p PYVERLINE=<"%TMPVERTXT%"
del /f /q "%TMPVERTXT%" >nul 2>&1

set "PYVER=0.0"
for /f "tokens=2 delims= " %%v in ("%PYVERLINE%") do set "PYVER=%%v"
echo [INFO] Version Python: %PYVER% >> "logs\session.log"

for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do set _MAJ=%%a& set _MIN=%%b
for /f "tokens=1,2 delims=." %%a in ("%PY_EXPECTED_MIN%") do set _EMAJ=%%a& set _EMIN=%%b
if "%_MAJ%"=="%_EMAJ%" if %_MIN% LSS %_EMIN% set "PYEXE="

if not defined PYEXE (
  echo [WARN] Python %PYVER% < %PY_EXPECTED_MIN%. Intentando instalar...>> "logs\session.log"
  goto :install_python
)

goto :python_ready

:: ---------- Instalar Python (PRIMERO python.org silencioso; luego winget forzado a fuente 'winget') ----------
:install_python
echo [INFO] Instalando Python %PY_SETUP_VER% desde python.org (silencioso)...
set "PY_URL=https://www.python.org/ftp/python/%PY_SETUP_VER%/python-%PY_SETUP_VER%-%ARCH%.exe"
set "PY_EXE_DL=%TEMP%\python-%PY_SETUP_VER%-%ARCH%.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try{Invoke-WebRequest -UseBasicParsing -Uri '%PY_URL%' -OutFile '%PY_EXE_DL%';exit 0}catch{exit 1}"
if not errorlevel 1 (
  "%PY_EXE_DL%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 SimpleInstall=1
  ping 127.0.0.1 -n 3 >nul
  call :set_find_python
)

if not defined PYEXE (
  echo [WARN] Descarga/instalación directa falló. Probando winget (fuente 'winget', sin msstore)...
  where winget >nul 2>nul
  if not errorlevel 1 (
    winget install -e --id Python.Python.3.12 -s winget --silent --accept-package-agreements --accept-source-agreements
    call :set_find_python
  )
)

if not defined PYEXE (
  echo [WARN] No fue posible instalar/detectar Python del sistema. Usaremos Python embebido.>> "logs\session.log"
  goto :fallback_embed
)

goto :python_ready

:: ---------- Fallback: Python embebido portátil ----------
:fallback_embed
set "EMBED_DIR=%ROOT%python_embed"
set "EMBED_EXE=%EMBED_DIR%\python.exe"
if not exist "%EMBED_EXE%" (
  echo [INFO] Descargando Python embebido...
  set "TMPZIP=%TEMP%\python-embed.zip"
  del /f /q "%TMPZIP%" >nul 2>&1
  set "URL=https://www.python.org/ftp/python/%PY_SETUP_VER%/python-%PY_SETUP_VER%-embed-%ARCH%.zip"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try{Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -OutFile '%TMPZIP%';exit 0}catch{exit 1}"
  if errorlevel 1 (
    echo [ERROR] No pude descargar Python embebido. Revisa tu conexión. >> "logs\session.log"
    echo [ERROR] No se pudo obtener Python. Cierra y vuelve a intentar.
    pause & exit /b 1
  )
  mkdir "%EMBED_DIR%" >nul 2>&1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '%TMPZIP%' '%EMBED_DIR%'"
  del /f /q "%TMPZIP%" >nul 2>&1

  :: Habilitar import site en python3x._pth
  for %%F in ("%EMBED_DIR%\python3*._pth") do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='%%~fF';$t=Get-Content -Raw $p; $t=$t -replace '^\s*#\s*import\s+site','import site'; if($t -notmatch 'import\s+site'){ $t=$t + \"`r`nimport site`r`n\"}; Set-Content -NoNewline -Path $p -Value $t -Encoding ASCII"
  )
  :: Instalar pip en el embebido
  set "GETPIP=%EMBED_DIR%\get-pip.py"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GETPIP%'"
  "%EMBED_EXE%" "%GETPIP%" --no-warn-script-location
)

set "PYEXE=%EMBED_EXE%"
echo [INFO] Usando Python embebido: %PYEXE% >> "logs\session.log"
goto :python_ready

:: ---------- Preparar entorno (venv si hay Python de sistema) ----------
:python_ready
set "IS_EMBED=0"
echo %PYEXE% | find /I "%ROOT%python_embed\" >nul && set "IS_EMBED=1"

if "%IS_EMBED%"=="0" (
  if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo [INFO] Creando entorno virtual .venv ...
    "%PYEXE%" -m venv "%ROOT%.venv" || (echo [ERROR] No se pudo crear .venv & pause & exit /b 1)
  )
  set "PYEXE=%ROOT%.venv\Scripts\python.exe"
)

echo [INFO] Actualizando instaladores... >> "logs\session.log"
"%PYEXE%" -m pip install --upgrade pip wheel setuptools >> "logs\session.log" 2>&1

if exist "%ROOT%requirements.txt" (
  echo [INFO] Instalando dependencias desde requirements.txt ...
  "%PYEXE%" -m pip install -r "%ROOT%requirements.txt" >> "logs\session.log" 2>&1
  if errorlevel 1 (
    echo [ERROR] Fallo instalando dependencias. Revisa logs\session.log
    type "logs\session.log"
    pause & exit /b 1
  )
) else (
  echo [WARN] No se encontró requirements.txt >> "logs\session.log"
)

:: ---------- Detectar paquete ----------
set "PKG=product_research_app"
if not exist "%ROOT%%PKG%\__init__.py" (
  for /d %%D in (*) do (
    if exist "%%D\__init__.py" (
      set "PKG=%%D"
      goto :pkg_ok
    )
  )
)
:pkg_ok

:: ---------- Lanzar servidor DIRECTO desde CMD (sin tuberías, sin relanzar) ----------
echo [INFO] Lanzando: -m %PKG%  >> "logs\session.log"
echo Iniciando servidor de Ecom Testing App...
"%PYEXE%" -m %PKG%
set "RC=%ERRORLEVEL%"

echo [INFO] Proceso terminado con código %RC% >> "logs\session.log"
if not "%RC%"=="0" (
  echo.
  echo [ERROR] El servidor no se inició correctamente. Revisa logs\session.log
  pause
)
exit /b %RC%
