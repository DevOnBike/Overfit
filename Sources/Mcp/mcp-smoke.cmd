@echo off
REM Smoke-tests the MCP server WITHOUT Claude: pipes a real initialize + tools/list handshake
REM into the server over stdio and prints the raw JSON-RPC responses (one per line).
REM Expect two lines: an initialize result (protocolVersion + serverInfo) and the tools array.
REM Usage: mcp-smoke.cmd ^<model.gguf^> [extra overfit-mcp options]
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

(
    echo {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"mcp-smoke","version":"0"}}}
    echo {"jsonrpc":"2.0","method":"notifications/initialized"}
    echo {"jsonrpc":"2.0","id":2,"method":"tools/list"}
) | "%OVERFIT_EXE%" mcp %*
