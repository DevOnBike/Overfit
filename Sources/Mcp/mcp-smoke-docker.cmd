@echo off
REM Smoke-tests the DOCKERIZED MCP server without Claude: pipes a real initialize + tools/list
REM handshake into `docker run -i` and prints the raw JSON-RPC responses.
REM Usage: mcp-smoke-docker.cmd ^<model.gguf^> [image]   (default image: devonbikeit/overfit:mcp)
setlocal

if "%~1"=="" (
    echo Usage: %~nx0 ^<model.gguf^> [image]
    exit /b 1
)

set "IMAGE=%~2"
if "%IMAGE%"=="" set "IMAGE=devonbikeit/overfit:mcp"

(
    echo {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"mcp-smoke-docker","version":"0"}}}
    echo {"jsonrpc":"2.0","method":"notifications/initialized"}
    echo {"jsonrpc":"2.0","id":2,"method":"tools/list"}
) | docker run -i --rm -v "%~dp1:/models" %IMAGE% "/models/%~nx1"
