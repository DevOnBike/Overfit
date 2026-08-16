@echo off
rem Native-AOT publish of a project, locally, with the C++ toolchain on PATH. This is the guard the CI
rem `aot-guard` job runs, made runnable on a dev box: ILCompiler actually executes, IL2026 / IL3050 / IL31xx
rem warnings on reachable code are promoted to errors, and the result is a native binary with no .NET runtime.
rem
rem A library cannot be Native-AOT compiled (no entry point), so the real AOT consumers are the executables:
rem Tests\AotSmokeTest is the minimal one CI publishes, Sources\Cli is the whole product surface.
rem
rem Usage:
rem   aot-publish.cmd                 - Sources\Cli, win-x64, into artifacts\aot-cli
rem   aot-publish.cmd smoke           - Tests\AotSmokeTest, which is what CI actually gates on
rem   aot-publish.cmd path\to.csproj  - any project
rem
rem Requires the "Desktop development with C++" workload. Any Visual Studio edition works: the location is
rem resolved with vswhere rather than hard-coded, because the version of this script that lived in .claude
rem assumed BuildTools 2022 and would have produced a confusing ILCompiler link failure on a machine with
rem Community or Professional.
rem
rem NOTE: keep CRLF line endings. cmd.exe mis-parses an LF-only batch file and the symptom is a cascade of
rem "'x' is not recognized" that points nowhere near the cause.
setlocal

set "REPO=%~dp0"
set "PROJECT=%REPO%Sources\Cli\Cli.csproj"
set "OUT=%REPO%artifacts\aot-cli"

if /i "%~1"=="smoke" (
    set "PROJECT=%REPO%Tests\AotSmokeTest\AotSmokeTest.csproj"
    set "OUT=%REPO%artifacts\aot-smoke"
) else if not "%~1"=="" (
    set "PROJECT=%~1"
    set "OUT=%REPO%artifacts\aot-custom"
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :no_vs

set "VCVARS="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"

if not defined VCVARS goto :no_cpp
if not exist "%VCVARS%" goto :no_cpp

echo --- toolchain: %VCVARS%
rem vcvars64.bat calls vswhere.exe UNQUALIFIED, so the Installer directory has to be on PATH BEFORE
rem the call, not after it. The script this replaced set it afterwards, which is why it printed
rem "'vswhere.exe' is not recognized" on every run and nobody noticed.
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VCVARS%" >nul
if errorlevel 1 goto :no_cpp

echo --- project:   %PROJECT%
echo --- output:    %OUT%
echo.

rem TreatWarningsAsErrors is the whole point: an IL2026/IL3050 on a reachable path must stop the publish
rem rather than appear in a log nobody reads. GenerateDocumentationFile off, because the XML doc has nothing
rem to say to a native binary and its warnings would drown the ones that matter.
dotnet publish "%PROJECT%" -c Release -r win-x64 -p:PublishAot=true -p:TreatWarningsAsErrors=true -p:GenerateDocumentationFile=false -o "%OUT%"
if errorlevel 1 goto :failed

echo.
echo --- ok: %OUT%
endlocal
exit /b 0

:no_vs
echo.
echo ERROR: vswhere.exe not found at "%VSWHERE%".
echo Visual Studio or the Build Tools are not installed, so ILCompiler has no linker to call.
exit /b 2

:no_cpp
echo.
echo ERROR: no Visual Studio installation carries the C++ toolchain (vcvars64.bat).
echo Install the "Desktop development with C++" workload; Native AOT links with MSVC's linker.
exit /b 2

:failed
echo.
echo --- AOT PUBLISH FAILED. A trim/AOT warning promoted to an error is a real finding: something on a
echo --- reachable path uses reflection, Activator, or an unrooted type. Fix the library, not this script.
exit /b 1
