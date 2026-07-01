# Builds a SIGNED, shareable APK of OverThink (for sideloading / a GitHub Release — NOT Google Play; Play wants
# the AAB from make-aab.ps1). Recipients install it directly after allowing "install from unknown sources".
# Run from anywhere:  ./make-apk.ps1        (AOT, best on-device speed, slow build ~10 min)
#                     ./make-apk.ps1 -Aot:$false   (fast build, slower first-token on the phone)
# Needs the upload keystore (run generate-upload-key.ps1 first) + the JDK/.NET android workload set up.

param(
    [string]$Keystore = (Join-Path $PSScriptRoot 'overthink-upload.keystore'),
    [string]$Alias    = 'overthink',
    [bool]  $Aot      = $true
)

$ErrorActionPreference = 'Stop'
$proj  = Join-Path $PSScriptRoot 'OverfitChatApp.csproj'
$sdk   = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$model = Join-Path $PSScriptRoot 'Assets\smollm2-135m.gguf'

if (-not (Test-Path $Keystore)) { throw "Keystore not found: $Keystore  (run generate-upload-key.ps1 first)" }

# The bundled model is git-ignored (over GitHub's 100 MB repo-file limit) — fetch it if it isn't on disk.
if (-not (Test-Path $model)) {
    Write-Host 'Fetching bundled model (~101 MB) ...'
    Invoke-WebRequest 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf' -OutFile $model
}

$storeSecure = Read-Host 'Keystore (store) password' -AsSecureString
$keySecure   = Read-Host 'Key password (press Enter to reuse the store password)' -AsSecureString
$storePass = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($storeSecure))
$keyPass   = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure))
if ([string]::IsNullOrEmpty($keyPass)) { $keyPass = $storePass }

# Pass the passwords via the `env:` prefix .NET-for-Android understands, so a special character in the password
# isn't mangled by the PowerShell -> dotnet -> MSBuild -> jarsigner chain (that caused a bare "jarsigner exit 1").
$env:OVERTHINK_STOREPASS = $storePass
$env:OVERTHINK_KEYPASS   = $keyPass

Write-Host "Building signed APK (AOT=$Aot — AOT takes a few minutes) ..."
dotnet publish $proj -c Release -f net10.0-android `
    -p:AndroidPackageFormat=apk `
    -p:RunAOTCompilation=$Aot `
    -p:AndroidSdkDirectory=$sdk `
    -p:AcceptAndroidSDKLicenses=true `
    -p:AndroidKeyStore=true `
    -p:AndroidSigningKeyStore=$Keystore `
    -p:AndroidSigningKeyAlias=$Alias `
    -p:AndroidSigningStorePass=env:OVERTHINK_STOREPASS `
    -p:AndroidSigningKeyPass=env:OVERTHINK_KEYPASS

$code = $LASTEXITCODE
$env:OVERTHINK_STOREPASS = $null
$env:OVERTHINK_KEYPASS   = $null
if ($code -ne 0) { throw "Build failed (exit $code)." }

$apk = Get-ChildItem (Join-Path $PSScriptRoot 'bin\Release\net10.0-android') -Recurse -Filter '*-Signed.apk' |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host ''
Write-Host '==================================================================' -ForegroundColor Green
Write-Host " Shareable APK: $($apk.FullName)"
Write-Host " Size: $([math]::Round($apk.Length / 1MB)) MB"
Write-Host '==================================================================' -ForegroundColor Green
Write-Host ' Attach it to a GitHub Release (drag-drop into the assets box), or share the file directly.'
Write-Host ' Recipients: enable "Install unknown apps" for their browser/file manager, then tap the .apk.'
