do shell script "cd " & quoted form of POSIX path of (do shell script "dirname " & quoted form of POSIX path of (path to me as text)) & " && ./run_mac.command >/dev/null 2>&1 &"
delay 2
open location "http://127.0.0.1:8000"
