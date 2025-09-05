Set shell = CreateObject("Wscript.Shell")
shell.Run "pythonw.exe -m product_research_app.web_app", 0, False
WScript.Sleep 2000
shell.Run "http://127.0.0.1:8000", 1, False
