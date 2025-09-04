@echo off
:: Batch file to launch the Product Research Copilot
:: This script should be run from Windows.  It installs dependencies if
:: necessary and then starts the application.

REM Change to the directory containing this batch file
cd /d %~dp0

REM Install Python dependencies (requires pip).  Errors are ignored if already installed.
echo Comprobando e instalando dependencias de Python...
python -m pip install --quiet --no-input -r requirements.txt 2>NUL

REM Launch the web application in a new window and open the browser automatically
echo Iniciando la interfaz web de Ecom Testing App...
REM Start the Python server in a new window so the batch script can continue
start "Ecom Testing App" python -m product_research_app.web_app
REM Espera un momento para que el servidor arranque
timeout /t 2 >nul
REM Abrir automáticamente el navegador predeterminado
start "" http://127.0.0.1:8000

echo.
echo La interfaz web debería abrirse automáticamente en su navegador.  Si no es así, visite http://127.0.0.1:8000
echo Cierre esta ventana para detener el servidor.