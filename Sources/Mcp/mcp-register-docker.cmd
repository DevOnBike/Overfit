@echo off
REM Registers the Overfit MCP server in Claude Code running FROM THE DOCKER IMAGE (no local build,
REM no .NET needed - the host pipes stdio into `docker run -i`). Mounts: the model's folder ->
REM /models, optional rag folder -> /docs, optional whisper model's folder -> /whisper.
REM
REM Usage: mcp-register-docker.cmd ^<model.gguf^> [rag-dir] [whisper-ggml]
REM Image: devonbikeit/overfit:mcp by default; override with  set OVERFIT_MCP_IMAGE=overfit:mcp-local
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 ^<model.gguf^> [rag-dir] [whisper-ggml]
    echo   e.g. %~nx0 C:\qwen3-06b\Qwen3-0.6B-Q4_K_M.gguf C:\docs C:\whisper\ggml-tiny.bin
    exit /b 1
)

set "IMAGE=%OVERFIT_MCP_IMAGE%"
if "%IMAGE%"=="" set "IMAGE=devonbikeit/overfit:mcp"
echo Using image: %IMAGE%

call claude mcp remove overfit >nul 2>&1

if not "%~3"=="" (
    call claude mcp add overfit -- docker run -i --rm -v "%~dp1:/models" -v "%~2:/docs" -v "%~dp3:/whisper" %IMAGE% "/models/%~nx1" --rag-dir /docs --whisper-model "/whisper/%~nx3"
) else if not "%~2"=="" (
    call claude mcp add overfit -- docker run -i --rm -v "%~dp1:/models" -v "%~2:/docs" %IMAGE% "/models/%~nx1" --rag-dir /docs
) else (
    call claude mcp add overfit -- docker run -i --rm -v "%~dp1:/models" %IMAGE% "/models/%~nx1"
)
if errorlevel 1 exit /b 1

echo.
echo Registered (dockerized). Verify with:  mcp-status.cmd
