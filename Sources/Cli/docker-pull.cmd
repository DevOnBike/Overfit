cls
@echo off
REM Pull the published Overfit OpenAI-server image from Docker Hub (https://hub.docker.com/r/devonbikeit/overfit)
REM at the version in Directory.Build.props — the same <Version> the NuGet packages + the published image are
REM tagged with, so the pulled image matches your source. No build needed (Native-AOT image from CI).
REM Override the repo with OVERFIT_IMAGE (default devonbikeit/overfit).
setlocal
set ROOT=%~dp0..\..
set REPO=%OVERFIT_IMAGE%
if "%REPO%"=="" set REPO=devonbikeit/overfit

REM Read <Version>x.y.z</Version> from Directory.Build.props (delims <> -> token 3 is the version number).
set VER=
for /f "tokens=3 delims=<>" %%v in ('findstr /c:"<Version>" "%ROOT%\Directory.Build.props"') do set VER=%%v
if "%VER%"=="" (
    echo Could not read ^<Version^> from "%ROOT%\Directory.Build.props".
    exit /b 1
)

echo Pulling %REPO%:%VER% from Docker Hub ...
docker pull %REPO%:%VER%
if errorlevel 1 (
    echo.
    echo PULL FAILED -- tag %VER% may not be published yet. Fallback: docker pull %REPO%:latest
    exit /b 1
)
echo.
echo Pulled:
docker images %REPO% --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"
echo.
echo Run it with a model mounted (model is a POSITIONAL arg of `serve`):
echo   docker run -p 8080:8080 -v C:\path\to\models:/models %REPO%:%VER% /models/model.gguf
echo.
echo Or via docker-run.cmd:
echo   set OVERFIT_IMAGE=%REPO%:%VER%
echo   docker-run.cmd C:\path\to\models\model.gguf 8080
endlocal
