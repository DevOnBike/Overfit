@echo off
REM Builds the MCP variant of the overfit image (same native binary, ENTRYPOINT `overfit mcp`,
REM stdio transport - no port). Build context = repo root, like docker-build.cmd.
REM Usage: docker-build-mcp.cmd [tag]    (default tag: overfit:mcp-local)
setlocal
set "TAG=%~1"
if "%TAG%"=="" set "TAG=overfit:mcp-local"

pushd "%~dp0..\.."
docker build -f Sources\Cli\Dockerfile --target mcp -t %TAG% .
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" exit /b %RC%

echo.
echo Built %TAG%. Try it (handshake smoke):
echo   Sources\Mcp\mcp-smoke-docker.cmd C:\models\model.gguf %TAG%
echo Or register it in Claude Code:
echo   Sources\Mcp\mcp-register-docker.cmd C:\models\model.gguf
