cls
@echo off
setlocal
REM ---------------------------------------------------------------------------
REM  download-bielik.cmd
REM
REM  Downloads the Polish Bielik-4.5B-v3.0-Instruct Q8_0 GGUF (~4.8 GB) so it can
REM  be loaded in pure .NET with OverfitClient.LoadGguf.
REM
REM  Source: speakleash/Bielik-4.5B-v3.0-Instruct-GGUF  (PUBLIC / ungated GGUF repo
REM          - no HuggingFace token required). The main *safetensors* repo is gated.
REM
REM  Uses HuggingFace Hub: resumable, integrity-checked, and a no-op if already
REM  downloaded. Requires Python with huggingface_hub  (pip install huggingface_hub).
REM
REM  Usage:  download-bielik.cmd [target-dir]      (default: C:\bielik)
REM  Tip:    for an fp16 reference build (~9 GB) swap the file name below to
REM          Bielik-4.5B-v3.0-Instruct-fp16.gguf
REM ---------------------------------------------------------------------------

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=C:\bielik"

set "REPO=speakleash/Bielik-4.5B-v3.0-Instruct-GGUF"
set "FILE=Bielik-4.5B-v3.0-Instruct.Q8_0.gguf"

echo Downloading %FILE% (~4.8 GB) to "%TARGET%"
echo (resumable - safe to re-run if interrupted)
echo.

python -c "from huggingface_hub import hf_hub_download; print('Saved to:', hf_hub_download(r'%REPO%', r'%FILE%', local_dir=r'%TARGET%'))"
if errorlevel 1 (
    echo.
    echo Download FAILED.
    echo If you saw "No module named huggingface_hub", run:  pip install huggingface_hub
    exit /b 1
)

echo.
echo Done. Load it in Overfit:
echo     var client = OverfitClient.LoadGguf(@"%TARGET%\%FILE%");
echo Or point the ASP.NET demo at it via appsettings ModelPath / OVERFIT_MODEL_DIR.
endlocal
