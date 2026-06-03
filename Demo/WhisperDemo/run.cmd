@echo off
rem Transcribes the sample audio with the Whisper tiny model. Optional arg: language code (default en).
rem   run.cmd          -> transcribe jfk.wav in English
rem   run.cmd pl       -> transcribe in Polish
setlocal
set "DIR=%~dp0..\materials"
if not exist "%DIR%\ggml-tiny.bin" (
  echo Missing materials. Run download-materials.cmd first.
  exit /b 1
)
set "LANGCODE=%~1"
if "%LANGCODE%"=="" set "LANGCODE=en"
dotnet run -c Release --project "%~dp0WhisperDemo.csproj" -- "%DIR%\ggml-tiny.bin" "%DIR%\jfk.wav" %LANGCODE%
endlocal
