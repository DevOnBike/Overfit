@echo off
REM ---------------------------------------------------------------------------
REM  Publish the overfit CLI as a single self-contained Native-AOT binary
REM  (no .NET runtime required on the target machine).
REM
REM  Prerequisite (Windows): the "Desktop development with C++" workload - the
REM  MSVC linker the AOT native-link step needs. Install once with:
REM    winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
REM  If link.exe is found but vswhere.exe is not, run this from a
REM  "Developer Command Prompt for VS 2022" so the MSVC environment is set up.
REM
REM  Usage:  publish.cmd [rid]
REM    [rid] = target runtime identifier (default win-x64; e.g. linux-x64, osx-arm64).
REM ---------------------------------------------------------------------------
setlocal
set RID=%~1
if "%RID%"=="" set RID=win-x64

echo Publishing overfit CLI ^(Native AOT, %RID%^) ...
dotnet publish "%~dp0Cli.csproj" -c Release -r %RID% -p:PublishAot=true
if errorlevel 1 goto :failed

echo.
echo Done. Native binary in:  %~dp0bin\Release\net10.0\%RID%\publish\
echo Put it on PATH and run:  overfit serve ^<model^>
endlocal
exit /b 0

:failed
echo.
echo Publish failed. If the error mentions a missing platform linker / vswhere,
echo install the "Desktop development with C++" workload and re-run from a
echo "Developer Command Prompt for VS 2022". See this script's header for details.
endlocal
exit /b 1
