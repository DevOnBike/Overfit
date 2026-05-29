cls
@echo off
REM AgentDemo is a console app (no web server, no launch profiles). It loads a GGUF from
REM OVERFIT_MODEL_DIR (default C:\qwen3b) and runs the RAG + tool-calling + JSON walkthrough.
dotnet run -c Release --project "%~dp0AgentDemo.csproj"
