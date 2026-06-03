@echo off
rem Downloads the Whisper tiny ggml model MicDemo needs into the shared Demo\materials folder.
rem Pure download via curl (built into Windows 10/11). Re-running skips an existing model.
setlocal
set "DIR=%~dp0..\materials"
if not exist "%DIR%" mkdir "%DIR%"

if not exist "%DIR%\ggml-tiny.bin" (
  echo Downloading Whisper tiny model ^(~77 MB^) ...
  curl -L -o "%DIR%\ggml-tiny.bin" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
) else (
  echo Model already present: %DIR%\ggml-tiny.bin
)

echo.
echo Done. Materials in: %DIR%
endlocal
