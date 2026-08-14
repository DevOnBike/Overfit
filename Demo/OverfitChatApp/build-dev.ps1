# ============================================================================
#  OverThink - EVERY-DAY DEVELOPMENT build.
#
#  Fast inner loop: incremental, no AOT, install over the existing app, launch.
#  Typical turnaround is seconds; the release script takes minutes.
#
#    ./build-dev.ps1                       # device already connected (USB or wireless from before)
#    ./build-dev.ps1 -Device 192.168.1.124:45509
#    ./build-dev.ps1 -Log                  # follow the on-device app log afterwards
#    ./build-dev.ps1 -NoInstall            # compile only
#
#  Why no AOT here. Measured 2026-08-14 on a Snapdragon 7s Gen 2 (docs/measured-baselines.md):
#  full AOT changes decode throughput by nothing measurable (8.5 vs 8.6 tok/s, inside run-to-run
#  noise) and buys 246 ms of cold start (401 vs 647 ms median). That is worth having in a release
#  and not worth paying for on every edit - the AOT build takes minutes instead of seconds.
#  Use ./build-release.ps1 when the startup number matters.
# ============================================================================

param(
    [string]$Device,
    [switch]$Log,
    [switch]$NoInstall
)

$ErrorActionPreference = 'Stop'

$proj  = Join-Path $PSScriptRoot 'OverfitChatApp.csproj'
$sdk   = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$adb   = Join-Path $sdk 'platform-tools\adb.exe'
$model = Join-Path $PSScriptRoot 'Assets\smollm2-135m.gguf'
$pkg   = 'com.devonbike.overthink'
$apk   = Join-Path $PSScriptRoot 'bin\Release\net10.0-android\android-arm64\com.devonbike.overthink-Signed.apk'
# Not $log: PowerShell variable names are case-insensitive, so $log collides with the -Log switch.
$logPath = "/sdcard/Android/data/$pkg/files/overthink.log"

# The bundled model is git-ignored (over GitHub's 100 MB file limit) - fetch it if it isn't on disk.
if (-not (Test-Path $model)) {
    Write-Host 'Fetching bundled model (~101 MB) ...'
    Invoke-WebRequest 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf' -OutFile $model
}

if ($Device) {
    Write-Host "Connecting to $Device ..."
    & $adb connect $Device
}

Write-Host 'Building (Release, no AOT, incremental) ...'
dotnet build $proj -c Release -f net10.0-android `
    -p:RunAOTCompilation=false `
    -p:AndroidSdkDirectory=$sdk `
    -p:AcceptAndroidSDKLicenses=true
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)." }

if ($NoInstall) {
    Write-Host "Built: $apk"
    return
}

& $adb get-state *> $null
if ($LASTEXITCODE -ne 0) {
    throw "No device. Plug in USB (and accept the prompt), or pass -Device <ip:port> from the phone's Wireless debugging screen."
}

# Remember the last arm line BEFORE installing, so the check below can wait for a genuinely new one.
# Reading it too early prints the previous build's arm and looks exactly like a successful check.
$armBefore = & $adb shell "grep 'build:' $logPath | tail -1"

# `install -r` replaces the APK and its native libraries in place. Deliberately NOT `adb uninstall`
# first: uninstalling wipes the app's data, which deletes overthink.log and every model the user
# added by hand. If a stale native library is ever suspected, uninstall once, by hand, knowingly.
Write-Host 'Installing ...'
& $adb install -r $apk
if ($LASTEXITCODE -ne 0) { throw 'Install failed.' }

& $adb shell monkey -p $pkg -c android.intent.category.LAUNCHER 1 *> $null

# The app records which arm it is at startup, read from its own installed libaot-*.so files. Printing
# it here means the build you think you flashed is the build that is running - asserting the arm from
# the build flags instead is how a measurement silently ends up describing the other one. Wait for a
# line that differs from the pre-install one; a fixed sleep printed the PREVIOUS build's arm.
$arm = $null
foreach ($attempt in 1..20) {
    Start-Sleep -Milliseconds 500
    $arm = & $adb shell "grep 'build:' $logPath | tail -1"
    if ($arm -and $arm -ne $armBefore) { break }
    $arm = $null
}
if ($arm) {
    Write-Host "On device -> $arm   (expect aotLibs=0 from this script)"
}
if (-not $arm) {
    Write-Host 'Could not confirm the running build from the app log (it may not have started yet).' -ForegroundColor Yellow
}

Write-Host 'Done - OverThink is running on the phone.'

if ($Log) {
    Write-Host "Following $logPath  (Ctrl+C to stop)"
    while ($true) {
        & $adb shell "tail -5 $logPath"
        Start-Sleep -Seconds 3
    }
}
