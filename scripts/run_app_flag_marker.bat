@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

:: ============================================================
:: Ecom Testing App — run_app_flag_marker.bat
:: One-time dependency bootstrap using a marker file
:: ============================================================

:: ---------- Paths and constants ----------
set "ROOT=%~dp0"
set "LOG_DIR=%ROOT%logs"
set "LOG_FILE=%LOG_DIR%\setup_flag.log"
set "FLAG_FILE=%ROOT%config\setup_complete.flag"
set "PY_EXPECTED_MIN=3.11"
set "APP_DEFAULT_PORT=8000"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%ROOT%config" mkdir "%ROOT%config"

echo ============================================================>>"%LOG_FILE%"
echo %DATE% %TIME% — run_app_flag_marker bootstrap >>"%LOG_FILE%"

:: ---------- Fast-path: skip dependency setup when marker exists ----------
if exist "%FLAG_FILE%" (
  echo [INFO] Marker found. Skipping dependency validation.>>"%LOG_FILE%"
  goto :run_application
)

:: ---------- Initial dependency check & installation ----------
call :ensure_python
if errorlevel 1 goto :setup_failed

call :ensure_virtualenv
if errorlevel 1 goto :setup_failed

call :install_requirements
if errorlevel 1 goto :setup_failed

:: ---------- Flag creation after successful setup ----------
(
  echo setup completed on %DATE% %TIME%
  echo python=%PYEXE%
) > "%FLAG_FILE%"
echo [INFO] Setup marker created at %FLAG_FILE%>>"%LOG_FILE%"

goto :run_application

:: ============================================================
:: Function: ensure_python — validates Python installation
:: Downloads embedded distribution if nothing suitable exists
:: ============================================================
:ensure_python
set "PYEXE="

for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if defined PYEXE goto :check_version

for /f "delims=" %%P in ('where python 2^>nul') do set "PYEXE=%%P"
if defined PYEXE goto :check_version

echo [WARN] Python not detected. Attempting to use embedded build...>>"%LOG_FILE%"
set "EMBED_DIR=%ROOT%python_embed"
set "PYEXE=%EMBED_DIR%\python.exe"
if not exist "%PYEXE%" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try{Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.6/python-3.12.6-embed-amd64.zip' -OutFile '%TEMP%\python-embed.zip';exit 0}catch{exit 1}" >>"%LOG_FILE%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Unable to download embedded Python.>>"%LOG_FILE%"
    echo [ERROR] Download failed. Verify your internet connection.
    exit /b 1
  )
  mkdir "%EMBED_DIR%" >nul 2>&1
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force '%TEMP%\python-embed.zip' '%EMBED_DIR%'" >>"%LOG_FILE%" 2>&1
  del "%TEMP%\python-embed.zip" >nul 2>&1
  for %%F in ("%EMBED_DIR%\python3*._pth") do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='%%~fF';$t=Get-Content -Raw $p; if($t -notmatch 'import\s+site'){ $t=$t + \"`r`nimport site`r`n\"}; $t=$t -replace '^\s*#\s*import\s+site','import site'; Set-Content -NoNewline $p $t -Encoding ASCII" >>'%LOG_FILE%' 2>&1
  )
  if not exist "%PYEXE%" (
    echo [ERROR] Embedded Python extraction failed.>>"%LOG_FILE%"
    exit /b 1
  )
)

echo [INFO] Using Python executable: %PYEXE%>>"%LOG_FILE%"
goto :verify_python_ready

:check_version
for /f "delims=" %%V in ('"%PYEXE%" -c "import sys;print('.'.join(map(str,sys.version_info[:2])))" 2^>nul') do set "PY_VERSION=%%V"
if not defined PY_VERSION (
  echo [WARN] Unable to read Python version.>>"%LOG_FILE%"
  set "PYEXE="
  goto :ensure_python
)
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do set "_MAJOR=%%a"&set "_MINOR=%%b"
for /f "tokens=1,2 delims=." %%a in ("%PY_EXPECTED_MIN%") do set "_REQ_MAJOR=%%a"&set "_REQ_MINOR=%%b"
if not "%_MAJOR%"=="%_REQ_MAJOR%" goto :verify_python_ready
if %_MINOR% LSS %_REQ_MINOR% (
  echo [WARN] Python version %PY_VERSION% below required %PY_EXPECTED_MIN%.>>"%LOG_FILE%"
  set "PYEXE="
  goto :ensure_python
)

echo [INFO] System Python %PY_VERSION% accepted.>>"%LOG_FILE%"

:verify_python_ready
if not defined PYEXE (
  echo [ERROR] Python could not be prepared.>>"%LOG_FILE%"
  exit /b 1
)
exit /b 0

:: ============================================================
:: Function: ensure_virtualenv — create dedicated venv when using system python
:: ============================================================
:ensure_virtualenv
echo %PYEXE% | find /I "%ROOT%python_embed\" >nul
if not errorlevel 1 (
  echo [INFO] Embedded Python detected. Skipping venv creation.>>"%LOG_FILE%"
  exit /b 0
)

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [INFO] Creating local virtual environment...>>"%LOG_FILE%"
  "%PYEXE%" -m venv "%ROOT%.venv" >>"%LOG_FILE%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Virtual environment creation failed.>>"%LOG_FILE%"
    exit /b 1
  )
)
set "PYEXE=%ROOT%.venv\Scripts\python.exe"
exit /b 0

:: ============================================================
:: Function: install_requirements — upgrade tooling & install packages once
:: ============================================================
:install_requirements
echo [INFO] Upgrading pip tooling...>>"%LOG_FILE%"
"%PYEXE%" -m pip install --upgrade pip setuptools wheel >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.>>"%LOG_FILE%"
  exit /b 1
)

if exist "%ROOT%requirements.txt" (
  echo [INFO] Installing requirements...>>"%LOG_FILE%"
  "%PYEXE%" -m pip install -r "%ROOT%requirements.txt" >>"%LOG_FILE%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed.>>"%LOG_FILE%"
    exit /b 1
  )
) else (
  echo [WARN] requirements.txt not found.>>"%LOG_FILE%"
)
exit /b 0

:: ============================================================
:: Runtime section — execute the application after setup
:: ============================================================
:run_application
echo [INFO] Launching application...>>"%LOG_FILE%"
set "APP_PYEXE=%PYEXE%"
if not defined APP_PYEXE set "APP_PYEXE=%ROOT%.venv\Scripts\python.exe"
if not exist "%APP_PYEXE%" set "APP_PYEXE=%ROOT%python_embed\python.exe"

if not exist "%APP_PYEXE%" (
  echo [ERROR] Runtime Python not found.>>"%LOG_FILE%"
  echo Application launch aborted.
  exit /b 1
)

start "" cmd /c ^
  "for /l %%i in (1,1,60) do (curl -I -s http://127.0.0.1:%APP_DEFAULT_PORT% >nul 2>&1 && start "" http://127.0.0.1:%APP_DEFAULT_PORT% & exit) & timeout /t 1 >nul"
"%APP_PYEXE%" -m product_research_app.web_app
set "RC=%ERRORLEVEL%"
echo [INFO] Application exited with code %RC%>>"%LOG_FILE%"
exit /b %RC%

:setup_failed
echo [ERROR] Initial setup failed. Dependencies remain unchecked.>>"%LOG_FILE%"
if exist "%FLAG_FILE%" del "%FLAG_FILE%" >nul 2>&1
echo Setup failed. Review %LOG_FILE% for details.
exit /b 1
