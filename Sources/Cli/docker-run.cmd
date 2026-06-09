cls
@echo off
REM Run the locally-built `overfit` server container with a model mounted from the host.
REM The model is NOT in the image — we bind-mount its directory to /models and point --model at it.
REM Usage:  docker-run.cmd  <path-to-model.gguf>  [port]   (default port: 8080, image: overfit:local)
setlocal
if "%~1"=="" (
    echo Usage: docker-run.cmd ^<path-to-model.gguf^> [port]
    echo Example: docker-run.cmd C:\qwen3-06b\Qwen3-0.6B-Q8_0.gguf 8080
    exit /b 1
)
if not exist "%~1" (
    echo Model file not found: %~1
    exit /b 1
)

set MODELDIR=%~dp1
REM strip the trailing backslash so the -v mount source is clean
if "%MODELDIR:~-1%"=="\" set MODELDIR=%MODELDIR:~0,-1%
set MODELFILE=%~nx1
set PORT=%~2
if "%PORT%"=="" set PORT=8080
set TAG=%OVERFIT_IMAGE%
if "%TAG%"=="" set TAG=overfit:local

echo Serving %MODELFILE% on http://localhost:%PORT%/v1  (Ctrl+C to stop)
REM `serve` takes the model as a POSITIONAL arg; it appends to the image's entrypoint.
docker run --rm -p %PORT%:8080 -v "%MODELDIR%:/models" %TAG% /models/%MODELFILE%
endlocal
