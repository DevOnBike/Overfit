@echo off
REM Runs the Overfit MCP server in the foreground for debugging (stdio = this console:
REM type/paste JSON-RPC lines, responses print back; logs go to stderr). Ctrl+C / Ctrl+Z+Enter to stop.
REM Usage: mcp-run.cmd ^<model.gguf^> [extra overfit-mcp options, e.g. --rag-dir C:\docs]
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 ^<model.gguf^> [--rag-dir dir] [--whisper-model ggml]
    exit /b 1
)

call "%~dp0mcp-locate-exe.cmd"
if "%OVERFIT_EXE%"=="" (
    echo overfit.exe not found. Build it first:  dotnet build Sources\Cli\Cli.csproj -c Release
    exit /b 1
)

"%OVERFIT_EXE%" mcp %*
