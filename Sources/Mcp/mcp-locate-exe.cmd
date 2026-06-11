@echo off
REM Shared helper: sets OVERFIT_EXE to the best available 'overfit' binary (empty if none).
REM Preference order: Native-AOT publish output, plain Release build, global tool on PATH.
set "OVERFIT_EXE=%~dp0..\Cli\bin\Release\net10.0\win-x64\publish\overfit.exe"
if exist "%OVERFIT_EXE%" goto :eof

set "OVERFIT_EXE=%~dp0..\Cli\bin\Release\net10.0\overfit.exe"
if exist "%OVERFIT_EXE%" goto :eof

where overfit >nul 2>&1
if not errorlevel 1 (
    set "OVERFIT_EXE=overfit"
    goto :eof
)

set "OVERFIT_EXE="
