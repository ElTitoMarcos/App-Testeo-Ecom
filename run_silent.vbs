Set oShell = CreateObject("Wscript.Shell")
oShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
oShell.Run "pythonw.exe -m product_research_app.web_app", 0, False
WScript.Sleep 1000
oShell.Run "http://127.0.0.1:8000"
