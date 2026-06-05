@echo off
REM ---------------------------------------------------------------------------
REM  overfit serve - start the OpenAI-compatible local server (dev convenience).
REM
REM  Framework-dependent "dotnet run" (fast iteration); for the shippable single
REM  native binary use publish.cmd (Native AOT).
REM
REM  Defaults (from the CLI itself): --host 127.0.0.1 --port 11434.
REM  NOTE: 11434 is also Ollama's default port - if Ollama is running, pass a
REM        free port, e.g.  run.cmd qwen2.5-3b --port 18080
REM
REM  Usage:  run.cmd <model> [--host H] [--port P]
REM    <model> = a name from "overfit list" (in %USERPROFILE%\.overfit\models)
REM              or a path to a .gguf file.
REM ---------------------------------------------------------------------------
if "%~1"=="" (
    echo Usage: run.cmd ^<model^> [--host H] [--port P]
    echo   example: run.cmd qwen2.5-3b
    echo   list downloaded models with:  overfit list
    exit /b 1
)
dotnet run -c Release --project "%~dp0Cli.csproj" -- serve %*
