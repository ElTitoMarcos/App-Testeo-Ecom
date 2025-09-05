Set shell = CreateObject("Wscript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
shell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)

' Start server hidden
shell.Run "pythonw.exe -m product_research_app.web_app", 0, False

' Wait for server to be ready (up to ~10s)
For i = 1 To 20
  WScript.Sleep 500
  On Error Resume Next
  Set http = CreateObject("MSXML2.XMLHTTP")
  http.Open "GET", "http://127.0.0.1:8000", False
  http.Send
  If Err.Number = 0 And http.Status = 200 Then Exit For
  Err.Clear
Next
On Error GoTo 0

' Open browser
shell.Run "http://127.0.0.1:8000", 1, False
