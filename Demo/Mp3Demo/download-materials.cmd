@echo off
rem Downloads a sample MP3 for Mp3Demo into the shared Demo\materials folder (a CC music clip — Mp3Demo
rem just decodes it to WAV, so any MP3 works). Pure download via curl. Re-running skips an existing file.
setlocal
set "DIR=%~dp0..\materials"
if not exist "%DIR%" mkdir "%DIR%"

if not exist "%DIR%\sample.mp3" (
  echo Downloading a sample MP3 ^(~9 MB^) ...
  curl -L -o "%DIR%\sample.mp3" "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
) else (
  echo Sample MP3 already present: %DIR%\sample.mp3
)

echo.
echo Done. Materials in: %DIR%
endlocal
