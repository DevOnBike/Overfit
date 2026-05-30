@echo off
setlocal
REM ---------------------------------------------------------------------------
REM  download-embedder.cmd
REM
REM  Downloads the MiniLM sentence-embedding model (all-MiniLM-L6-v2, ~90 MB) used
REM  by the Local Agent demo's RAG endpoints (/documents/index, /rag/query).
REM
REM  This is the demo's SECOND model: the LLM answers, the embedder retrieves.
REM  (The LLM is fetched by download-bielik.cmd, or any GGUF you point ModelPath at.)
REM
REM  Source: sentence-transformers/all-MiniLM-L6-v2  (PUBLIC / ungated - no token).
REM  Requires Python with huggingface_hub  (pip install huggingface_hub).
REM
REM  Usage:  download-embedder.cmd [target-dir]     (default: C:\minilm)
REM  Then:   set EmbeddingModelPath (or OVERFIT_EMBEDDING_DIR) to that directory.
REM ---------------------------------------------------------------------------

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=C:\minilm"

set "REPO=sentence-transformers/all-MiniLM-L6-v2"

echo Downloading MiniLM (all-MiniLM-L6-v2, ~90 MB) to "%TARGET%"
echo.

python -c "from huggingface_hub import hf_hub_download as g; [g(r'%REPO%', f, local_dir=r'%TARGET%') for f in ('config.json','model.safetensors','vocab.txt','tokenizer.json')]; print('Saved to:', r'%TARGET%')"
if errorlevel 1 (
    echo.
    echo Download FAILED.
    echo If you saw "No module named huggingface_hub", run:  pip install huggingface_hub
    exit /b 1
)

echo.
echo Done. Enable RAG by pointing the demo at it, e.g. in appsettings:
echo     "EmbeddingModelPath": "%TARGET%"
echo or set the OVERFIT_EMBEDDING_DIR environment variable.
endlocal
