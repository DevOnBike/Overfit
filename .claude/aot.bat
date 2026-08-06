@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
dotnet publish "D:\Overfit\Sources\Cli\Cli.csproj" -c Release -r win-x64 -p:PublishAot=true -p:TreatWarningsAsErrors=true -p:GenerateDocumentationFile=false -o "D:\Overfit\.claude\aot-cli"
