# Builds a SIGNED release AAB of OverThink (AOT) for a manual upload to Play Console -> Internal testing.
# Run from anywhere:  ./make-aab.ps1
# Needs: the upload keystore (run generate-upload-key.ps1 first) + JDK/.NET android workload already set up.

param(
    [string]$Keystore = (Join-Path $PSScriptRoot 'overthink-upload.keystore'),
    [string]$Alias    = 'overthink'
)

$ErrorActionPreference = 'Stop'
$proj  = Join-Path $PSScriptRoot 'OverfitChatApp.csproj'
$sdk   = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$model = Join-Path $PSScriptRoot 'Assets\smollm2-135m.gguf'

if (-not (Test-Path $Keystore)) { throw "Keystore not found: $Keystore  (run generate-upload-key.ps1 first)" }

# The bundled model is git-ignored (over GitHub's 100 MB limit) — fetch it if it isn't on disk.
if (-not (Test-Path $model)) {
    Write-Host 'Fetching bundled model (~101 MB) ...'
    Invoke-WebRequest 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf' -OutFile $model
}

$storeSecure = Read-Host 'Keystore (store) password' -AsSecureString
$keySecure   = Read-Host 'Key password (press Enter to reuse the store password)' -AsSecureString
$storePass = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($storeSecure))
$keyPass   = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure))
if ([string]::IsNullOrEmpty($keyPass)) { $keyPass = $storePass }

Write-Host 'Building signed AAB (AOT — this takes a few minutes) ...'
dotnet publish $proj -c Release -f net10.0-android `
    -p:AndroidPackageFormat=aab `
    -p:RunAOTCompilation=true `
    -p:AndroidSdkDirectory=$sdk `
    -p:AcceptAndroidSDKLicenses=true `
    -p:AndroidKeyStore=true `
    -p:AndroidSigningKeyStore=$Keystore `
    -p:AndroidSigningKeyAlias=$Alias `
    -p:AndroidSigningStorePass=$storePass `
    -p:AndroidSigningKeyPass=$keyPass

if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)." }

$aab = Get-ChildItem (Join-Path $PSScriptRoot 'bin\Release\net10.0-android') -Recurse -Filter '*-Signed.aab' |
       Select-Object -First 1
Write-Host ''
Write-Host '==================================================================' -ForegroundColor Green
Write-Host " Signed AAB: $($aab.FullName)"
Write-Host '==================================================================' -ForegroundColor Green
Write-Host ' Upload it in Play Console -> Test wewnętrzny -> Utwórz wersję -> dodaj plik AAB.'
