@echo off
setlocal enabledelayedexpansion
rem ============================================================================
rem  Overfit Chat - build the APK and deploy it to the phone.
rem
rem  Usage:
rem    deploy.cmd                 - device already connected (USB, or wireless from before)
rem    deploy.cmd 192.168.1.174:PORT   - connect wireless first (PORT from the phone's
rem                                       "Wireless debugging" screen; it rotates each session)
rem    deploy.cmd 192.168.1.174:PORT aot   - same, but AOT-compile (slower build, faster decode)
rem
rem  First-time wireless pairing (do once, manually):
rem    "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" pair 192.168.1.174:PAIRPORT CODE
rem ============================================================================

set "SDK=%LOCALAPPDATA%\Android\Sdk"
set "ADB=%SDK%\platform-tools\adb.exe"
set "PROJ=%~dp0OverfitChatApp.csproj"
set "APK=%~dp0bin\Release\net10.0-android\android-arm64\com.devonbike.overthink-Signed.apk"
set "PKG=com.devonbike.overthink"

rem --- optional AOT flag (2nd arg = aot) ---
set "AOT="
if /I "%~2"=="aot" set "AOT=-p:RunAOTCompilation=true"

rem --- optional wireless connect (1st arg = ip:port) ---
if not "%~1"=="" (
    echo [1/4] Connecting to %~1 ...
    "%ADB%" connect %~1
) else (
    echo [1/4] Using already-connected device.
)

rem The bundled model is git-ignored (over GitHub's 100 MB limit) — fetch it if it's not on disk.
if not exist "%~dp0Assets\smollm2-135m.gguf" (
    echo.
    echo Fetching bundled model ^(~101 MB^) ...
    curl -L -f -o "%~dp0Assets\smollm2-135m.gguf" https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf
)

echo.
echo [2/4] Building APK %AOT% ...
dotnet build "%PROJ%" -c Release -f net10.0-android -p:AndroidSdkDirectory="%SDK%" -p:AcceptAndroidSDKLicenses=true %AOT%
if errorlevel 1 (
    echo.
    echo BUILD FAILED.
    exit /b 1
)

echo.
echo [3/4] Checking device ...
"%ADB%" get-state >nul 2>&1
if errorlevel 1 (
    echo.
    echo No device connected. Plug in USB ^(and accept the prompt^), or pass the wireless address:
    echo     deploy.cmd 192.168.1.174:PORT
    exit /b 1
)

echo.
echo [4/4] Installing + launching ...
"%ADB%" install -r "%APK%"
if errorlevel 1 (
    echo INSTALL FAILED.
    exit /b 1
)
"%ADB%" shell monkey -p %PKG% -c android.intent.category.LAUNCHER 1 >nul 2>&1

echo.
echo Done - "OverThink" is running on the phone.
rem  To pre-load a model so it auto-loads on launch:
rem    "%ADB%" push C:\path\to\model.gguf /sdcard/Android/data/%PKG%/files/model.gguf
endlocal
