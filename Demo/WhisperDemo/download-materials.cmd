@echo off
rem Downloads the materials WhisperDemo needs (Whisper tiny ggml model + a sample WAV) into the shared
rem Demo\materials folder. Pure download via curl (built into Windows 10/11). Re-running skips existing files.
setlocal
set "DIR=%~dp0..\materials"
if not exist "%DIR%" mkdir "%DIR%"

if not exist "%DIR%\ggml-tiny.bin" (
  echo Downloading Whisper tiny model ^(~77 MB^) ...
  curl -L -o "%DIR%\ggml-tiny.bin" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
) else (
  echo Model already present: %DIR%\ggml-tiny.bin
)

if not exist "%DIR%\jfk.wav" (
  echo Downloading sample audio jfk.wav ...
  curl -L -o "%DIR%\jfk.wav" "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"
) else (
  echo Sample WAV already present: %DIR%\jfk.wav
)

echo.
echo Done. Materials in: %DIR%
endlocal
