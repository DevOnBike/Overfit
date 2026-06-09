cls
@echo off
REM Quickly build the Native-AOT `overfit` server image locally.
REM The build context MUST be the repository root (the CLI references Sources\Main + Sources\Server).
REM Usage:  docker-build.cmd  [image-tag]      (default tag: overfit:local)
setlocal
set TAG=%~1
if "%TAG%"=="" set TAG=overfit:local
set ROOT=%~dp0..\..

echo Building %TAG% from context "%ROOT%" ...
docker build -f "%~dp0Dockerfile" -t %TAG% "%ROOT%"
if errorlevel 1 (
    echo.
    echo BUILD FAILED.
    exit /b 1
)
echo.
echo Built %TAG%:
docker images %TAG% --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"
echo.
echo Run it with:  docker-run.cmd ^<path-to-model.gguf^> [port]
endlocal
