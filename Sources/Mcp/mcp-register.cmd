@echo off
REM Registers the Overfit MCP server in Claude Code (run from the directory whose Claude
REM project should own it - registration scope is the CURRENT directory's project).
REM Re-registering is safe: any existing 'overfit' entry is removed first.
REM
REM Usage: mcp-register.cmd ^<model.gguf^> [rag-dir] [whisper-ggml]
REM   e.g. mcp-register.cmd C:\qwen3-06b\Qwen3-0.6B-Q4_K_M.gguf C:\docs C:\whisper\ggml-tiny.bin
REM
REM NOTE: registers the path of the locally-built exe - if you later run cleanup.cmd (purges
REM bin/), re-run this script after rebuilding. (PowerShell users: don't paste the underlying
REM 'claude mcp add' by hand into pwsh - it swallows the bare '--'. This .cmd avoids that.)
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 ^<model.gguf^> [rag-dir] [whisper-ggml]
    echo   e.g. %~nx0 C:\qwen3-06b\Qwen3-0.6B-Q4_K_M.gguf C:\docs C:\whisper\ggml-tiny.bin
    exit /b 1
)

call "%~dp0mcp-locate-exe.cmd"
if "%OVERFIT_EXE%"=="" (
    echo overfit.exe not found. Build it first:  dotnet build Sources\Cli\Cli.csproj -c Release
    exit /b 1
)
echo Using: %OVERFIT_EXE%

call claude mcp remove overfit >nul 2>&1

if not "%~3"=="" (
    call claude mcp add overfit -- "%OVERFIT_EXE%" mcp "%~1" --rag-dir "%~2" --whisper-model "%~3"
) else if not "%~2"=="" (
    call claude mcp add overfit -- "%OVERFIT_EXE%" mcp "%~1" --rag-dir "%~2"
) else (
    call claude mcp add overfit -- "%OVERFIT_EXE%" mcp "%~1"
)
if errorlevel 1 exit /b 1

echo.
echo Registered. Verify with:  mcp-status.cmd   (or just ask Claude to use the 'overfit' tools)
