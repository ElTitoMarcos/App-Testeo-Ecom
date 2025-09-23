' run_silent.vbs
Option Explicit

Dim fso, shell, folder, cmd
Set fso   = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

folder = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = Chr(34) & folder & "\run_app.bat" & Chr(34)

' 0 = oculto, False = no esperar a que termine
shell.Run cmd, 0, False
