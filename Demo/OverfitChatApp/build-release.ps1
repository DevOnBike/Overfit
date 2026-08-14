# ============================================================================
#  OverThink - LOCAL RELEASE build (signed, AOT).
#
#    ./build-release.ps1                      # signed APK for sideloading / a GitHub Release
#    ./build-release.ps1 -Format aab -VersionCode 3   # signed AAB for Play Console
#    ./build-release.ps1 -Install             # also flash it to the connected phone
#
#  Supersedes make-apk.ps1 and make-aab.ps1 (same signing approach, one script, and the AOT flags
#  corrected - see below). Those two are kept for now only so nobody's muscle memory breaks.
#
#  Why AOT here and not in build-dev.ps1. Measured 2026-08-14 on a Snapdragon 7s Gen 2, 6 launches
#  per arm, arms verified from the installed libaot-*.so files rather than from the build flags
#  (docs/measured-baselines.md):
#
#      decode      JIT 8.6 tok/s   vs  AOT 8.5 tok/s   -> no difference, inside noise
#      cold start  JIT 647 ms      vs  AOT 401 ms      -> 246 ms, ranges do not overlap
#
#  So AOT buys launch latency, not tokens. It costs ~7.3 MB of native libraries and a build measured
#  in minutes.
#
#  AndroidEnableProfiledAot=false is NOT optional. With the default (profiled AOT on), only the
#  startup profile is compiled: libaot-DevOnBike.Overfit.dll.so came out at 16 kB instead of 1155 kB
#  - the inference engine was ~1.4% compiled and the build was "AOT" in name only. make-apk.ps1 and
#  make-aab.ps1 shipped exactly that build until this was measured.
# ============================================================================

param(
    [ValidateSet('apk', 'aab')]
    [string]$Format = 'apk',

    # Android versionCode. For an AAB it MUST be strictly higher than any bundle already uploaded to
    # ANY Play track - internal, closed and production share one versionCode space.
    [int]$VersionCode = 0,

    [string]$Keystore = (Join-Path $PSScriptRoot 'overthink-upload.keystore'),
    [string]$Alias    = 'overthink',
    [switch]$Install
)

$ErrorActionPreference = 'Stop'

$proj  = Join-Path $PSScriptRoot 'OverfitChatApp.csproj'
$sdk   = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$adb   = Join-Path $sdk 'platform-tools\adb.exe'
$model = Join-Path $PSScriptRoot 'Assets\smollm2-135m.gguf'
$pkg   = 'com.devonbike.overthink'

if (-not (Test-Path $Keystore)) {
    throw "Keystore not found: $Keystore  (run generate-upload-key.ps1 first)"
}
if ($Format -eq 'aab' -and $VersionCode -le 0) {
    throw 'An AAB needs -VersionCode, and it must be higher than every versionCode already uploaded to Play.'
}
if ($Format -eq 'aab' -and $Install) {
    throw 'An AAB cannot be installed with adb. Build -Format apk for the phone, or upload the AAB to Play.'
}

if (-not (Test-Path $model)) {
    Write-Host 'Fetching bundled model (~101 MB) ...'
    Invoke-WebRequest 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf' -OutFile $model
}

$storeSecure = Read-Host 'Keystore (store) password' -AsSecureString
$keySecure   = Read-Host 'Key password (press Enter to reuse the store password)' -AsSecureString
$storePass = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($storeSecure))
$keyPass   = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure))
if ([string]::IsNullOrEmpty($keyPass)) { $keyPass = $storePass }

# Hand the passwords to the build through the `env:` indirection that .NET-for-Android understands,
# never literally on the command line. A literal -p:AndroidSigningStorePass=<pw> is re-parsed by
# PowerShell -> dotnet -> MSBuild -> jarsigner, so any special character ($, ", ;, space, !) silently
# corrupts it and jarsigner fails with a bare "exited with code 1". Via env: the raw string arrives intact.
$env:OVERTHINK_STOREPASS = $storePass
$env:OVERTHINK_KEYPASS   = $keyPass

# A release build starts clean: an incremental tree can carry native libraries from a previous
# (non-AOT, or profiled-AOT) build, and the result is a package whose arm nobody can vouch for.
Write-Host 'Cleaning ...'
dotnet clean $proj -c Release -f net10.0-android | Out-Null

$versionArgs = @()
if ($VersionCode -gt 0) { $versionArgs = @("-p:ApplicationVersion=$VersionCode") }

Write-Host "Building signed $($Format.ToUpper()) with full AOT - this takes minutes ..."
dotnet publish $proj -c Release -f net10.0-android `
    -p:AndroidPackageFormat=$Format `
    -p:RunAOTCompilation=true `
    -p:AndroidEnableProfiledAot=false `
    -p:AndroidSdkDirectory=$sdk `
    -p:AcceptAndroidSDKLicenses=true `
    -p:AndroidKeyStore=true `
    -p:AndroidSigningKeyStore=$Keystore `
    -p:AndroidSigningKeyAlias=$Alias `
    -p:AndroidSigningStorePass=env:OVERTHINK_STOREPASS `
    -p:AndroidSigningKeyPass=env:OVERTHINK_KEYPASS `
    @versionArgs

$code = $LASTEXITCODE
$env:OVERTHINK_STOREPASS = $null
$env:OVERTHINK_KEYPASS   = $null
if ($code -ne 0) { throw "Build failed (exit $code)." }

$artifact = Get-ChildItem (Join-Path $PSScriptRoot 'bin\Release\net10.0-android') -Recurse -Filter "*-Signed.$Format" |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $artifact) { throw "Build reported success but no *-Signed.$Format was produced." }

Write-Host ''
Write-Host '==================================================================' -ForegroundColor Green
Write-Host " Signed $($Format.ToUpper()): $($artifact.FullName)"
Write-Host " Size: $([math]::Round($artifact.Length / 1MB)) MB"
Write-Host '==================================================================' -ForegroundColor Green

if ($Format -eq 'aab') {
    Write-Host ' Upload in Play Console -> Internal testing -> Create release -> add the AAB.'
    return
}

Write-Host ' Attach it to a GitHub Release, or share the file directly.'
Write-Host ' Recipients: enable "Install unknown apps" for their browser/file manager, then tap the .apk.'

if (-not $Install) { return }

& $adb get-state *> $null
if ($LASTEXITCODE -ne 0) { throw 'No device connected.' }

# `install -r`, not uninstall-then-install: uninstalling wipes app data, which deletes the on-device
# log and any model the user added by hand.
& $adb install -r $artifact.FullName
if ($LASTEXITCODE -ne 0) { throw 'Install failed.' }
& $adb shell monkey -p $pkg -c android.intent.category.LAUNCHER 1 *> $null
Start-Sleep -Milliseconds 700

# Confirm the phone is running an AOT build, from the app's own probe of its installed libraries.
$arm = & $adb shell "grep 'build:' /sdcard/Android/data/$pkg/files/overthink.log | tail -1"
if ($arm) { Write-Host "On device -> $arm   (expect aotLibs > 0 for a release build)" }
