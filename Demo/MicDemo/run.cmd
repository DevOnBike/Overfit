@echo off
rem Live microphone speech-to-text. Optional args: language (default en), seconds per round (default 5).
rem   run.cmd            :: English, 5 s rounds
rem   run.cmd pl 5       :: Polish, 5 s rounds
setlocal
set "DIR=%~dp0..\materials"
if not exist "%DIR%\ggml-tiny.bin" (
  echo Missing model. Run download-materials.cmd first.
  exit /b 1
)
set "LANGCODE=%~1"
if "%LANGCODE%"=="" set "LANGCODE=en"
set "SECS=%~2"
if "%SECS%"=="" set "SECS=5"
dotnet run -c Release --project "%~dp0MicDemo.csproj" -- "%DIR%\ggml-tiny.bin" %LANGCODE% %SECS%
endlocal
