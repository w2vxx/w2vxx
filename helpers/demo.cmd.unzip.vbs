ExtractTo=WScript.Arguments(0)
ZipFile=WScript.Arguments(1)

set objShell = CreateObject("Shell.Application")
set FilesInZip=objShell.NameSpace(ZipFile).items
objShell.NameSpace(ExtractTo).CopyHere(FilesInZip)
Set objShell = Nothing
