-- Compilar con Editor de Scripts -> Archivo > Exportar... tipo "Aplicación"
set appPath to POSIX path of (path to me as alias)
set parentDir to do shell script "dirname " & quoted form of appPath
do shell script "cd " & quoted form of parentDir & " && /bin/chmod +x ./run_mac.command && ./run_mac.command >/dev/null 2>&1 &"
delay 2
open location "http://127.0.0.1:8000"
