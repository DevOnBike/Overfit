@echo off
rem Overfit OpenAI-compatible server (ASP.NET / Kestrel, Native-AOT-ready host).
rem
rem Usage:
rem   serve.cmd                                  -> default model on port 8080
rem   serve.cmd C:\path\model.gguf               -> a specific GGUF
rem   serve.cmd C:\path\model.gguf 11434         -> and a specific port
rem   serve.cmd model.gguf 8080 --embed-model C:\minilm   -> extra flags pass straight through
setlocal

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=C:\qwen3b\qwen.q4km.gguf"

set "PORT=%~2"
if "%PORT%"=="" set "PORT=8080"

rem Drop the first two args so any remainder (e.g. --embed-model, --tts-model) passes through to serve.
if not "%~1"=="" shift
if not "%~1"=="" shift

echo.
echo   Overfit serve   model = %MODEL%   port = %PORT%
echo   API   http://127.0.0.1:%PORT%/v1
echo   Docs  http://127.0.0.1:%PORT%/docs
echo.

dotnet run -c Release --project "%~dp0Sources\Cli\Cli.csproj" -- serve "%MODEL%" --port %PORT% %1 %2 %3 %4 %5 %6

endlocal
