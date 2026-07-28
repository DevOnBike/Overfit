@echo off
setlocal
REM Builds the local `overfit:latest` image the lab deployment uses.
REM
REM Build context is the REPO ROOT (the Dockerfile copies the whole solution), and this is a Native-AOT
REM publish inside the container — expect several minutes on a first run.
REM
REM Docker Desktop's Kubernetes shares this daemon's image store, so no registry or push is involved.

pushd "%~dp0\..\.."
echo === building overfit:latest from %CD% ===
docker build -f "Sources\Cli\Dockerfile" -t overfit:latest .
if errorlevel 1 (
    echo.
    echo Build failed.
    popd & endlocal & exit /b 1
)
echo.
docker image inspect overfit:latest --format "built {{.Created}}  size {{.Size}} bytes"
popd
endlocal
