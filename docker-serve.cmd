@echo off
rem Build + run the Overfit server as a Docker image: a native-AOT Kestrel host on a chiselled base
rem (tiny image, fast cold start, no .NET runtime inside). The GGUF is NOT baked in — it is mounted from a
rem host directory at runtime and passed as the positional model arg.
rem
rem Usage:
rem   docker-serve.cmd                                   -> C:\qwen3b\qwen.q4km.gguf on port 8080
rem   docker-serve.cmd C:\qwen3b qwen.q4km.gguf 11434    -> models-dir, model-file, port
setlocal

set "MODELS=%~1"
if "%MODELS%"=="" set "MODELS=C:\qwen3b"

set "MODELFILE=%~2"
if "%MODELFILE%"=="" set "MODELFILE=qwen.q4km.gguf"

set "PORT=%~3"
if "%PORT%"=="" set "PORT=8080"

rem cd into the repo root (the build context) so the Dockerfile's `COPY . .` sees the whole tree.
rem Using "." as the context avoids the trailing-backslash-in-quotes trap of "%~dp0".
pushd "%~dp0"

echo.
echo Building image 'overfit' (first build downloads the .NET SDK image and AOT-compiles for linux - a few minutes)...
docker build -f "Sources\Cli\Dockerfile" -t overfit .
if errorlevel 1 (
  echo.
  echo Docker build failed. Is Docker Desktop running?
  popd
  exit /b 1
)

echo.
echo   Serving %MODELS%\%MODELFILE%
echo   API   http://127.0.0.1:%PORT%/v1
echo   Docs  http://127.0.0.1:%PORT%/docs
echo.
docker run --rm -p %PORT%:8080 -v "%MODELS%:/models" overfit /models/%MODELFILE%

popd
endlocal
