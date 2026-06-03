@echo off
rem Decodes an MP3 to a 16 kHz WAV with the pure-C# decoder. Optional args: input mp3, output wav.
rem   run.cmd                       :: decodes ..\materials\sample.mp3
rem   run.cmd C:\music\song.mp3      :: decodes your own file
setlocal
set "DIR=%~dp0..\materials"
set "IN=%~1"
if "%IN%"=="" set "IN=%DIR%\sample.mp3"
if not exist "%IN%" (
  echo MP3 not found: %IN%
  echo Run download-materials.cmd for a sample, or pass a path: run.cmd ^<input.mp3^> [output.wav]
  exit /b 1
)
dotnet run -c Release --project "%~dp0Mp3Demo.csproj" -- "%IN%" %2
endlocal
