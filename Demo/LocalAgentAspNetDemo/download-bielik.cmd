cls
@echo off
setlocal
REM ---------------------------------------------------------------------------
REM  download-bielik.cmd
REM
REM  Downloads the Polish Bielik-4.5B-v3.0-Instruct Q4_K_M GGUF (~2.7 GB) so it can
REM  be loaded in pure .NET with OverfitClient.LoadGguf. Q4_K_M decodes ~1.24x faster
REM  than Q8_0 on CPU (smaller + mmap-friendly) and stays coherent in Polish — it is
REM  what the Bielik preset (appsettings.Bielik.json) points at by default.
REM
REM  Source: second-state/Bielik-4.5B-v3.0-Instruct-GGUF  (PUBLIC / ungated community
REM          quant - no token). The OFFICIAL speakleash repo only ships fp16 + Q8_0.
REM          The main *safetensors* repo is gated.
REM
REM  Uses HuggingFace Hub: resumable, integrity-checked, and a no-op if already
REM  downloaded. Requires Python with huggingface_hub  (pip install huggingface_hub).
REM
REM  Usage:  download-bielik.cmd [target-dir]      (default: C:\bielik)
REM  Variants (swap REPO/FILE below):
REM    Q8_0  (official, higher quality, ~4.8 GB): speakleash/...-GGUF  Bielik-4.5B-v3.0-Instruct.Q8_0.gguf
REM    fp16  (reference,                ~9 GB):   speakleash/...-GGUF  Bielik-4.5B-v3.0-Instruct-fp16.gguf
REM ---------------------------------------------------------------------------

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=C:\bielik"

set "REPO=second-state/Bielik-4.5B-v3.0-Instruct-GGUF"
set "FILE=Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf"

echo Downloading %FILE% (~2.7 GB) to "%TARGET%"
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
