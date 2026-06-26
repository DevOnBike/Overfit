# Generates the Google Play UPLOAD keystore for OverThink, then prints everything you need for the
# GitHub Actions secrets. Run it once, in PowerShell, from anywhere:
#
#   ./make-upload-key.ps1
#
# Keep the resulting .keystore file and the passwords SOMEWHERE SAFE (a password manager) — if you lose
# the upload key you can reset it via Play Console, but losing it + the passwords is a headache. With Play
# App Signing, Google holds the real app-signing key; this upload key only signs what you send to Google.

param(
    [string]$Alias    = "overthink",
    [string]$Keystore = "overthink-upload.keystore",
    [int]   $Validity = 10000,                                  # ~27 years
    [string]$Dname    = "CN=Maciej Rychter, O=DevOnBike, C=PL"
)

$ErrorActionPreference = "Stop"

# --- locate keytool (ships with the JDK) ---
$keytool = $null
if ($env:JAVA_HOME -and (Test-Path "$env:JAVA_HOME\bin\keytool.exe")) {
    $keytool = "$env:JAVA_HOME\bin\keytool.exe"
} elseif (Get-Command keytool -ErrorAction SilentlyContinue) {
    $keytool = (Get-Command keytool).Source
} else {
    $guess = Get-ChildItem "C:\Program Files\Microsoft\jdk-*\bin\keytool.exe" -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if ($guess) { $keytool = $guess.FullName }
}
if (-not $keytool) { throw "keytool not found. Install a JDK (e.g. Microsoft OpenJDK 17) or set JAVA_HOME." }
Write-Host "Using keytool: $keytool"

if (Test-Path $Keystore) { throw "$Keystore already exists — delete it first or pass -Keystore <other>." }

# --- passwords ---
$storeSecure = Read-Host "Keystore (store) password" -AsSecureString
$keySecure   = Read-Host "Key password (press Enter to reuse the store password)" -AsSecureString
$storePass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                 [Runtime.InteropServices.Marshal]::SecureStringToBSTR($storeSecure))
$keyPass   = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                 [Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure))
if ([string]::IsNullOrEmpty($keyPass)) { $keyPass = $storePass }
if ($storePass.Length -lt 6) { throw "Keystore password must be at least 6 characters." }

# --- generate ---
& $keytool -genkeypair -v -keystore $Keystore -alias $Alias `
    -keyalg RSA -keysize 2048 -validity $Validity `
    -storepass $storePass -keypass $keyPass -dname $Dname
if ($LASTEXITCODE -ne 0) { throw "keytool failed (exit $LASTEXITCODE)." }

# --- base64 for the GitHub secret ---
# Resolve to an absolute path: .NET file APIs use [Environment]::CurrentDirectory, NOT PowerShell's $PWD,
# so a relative path here looks in the wrong folder.
$ksFull  = (Resolve-Path -LiteralPath $Keystore).Path
$b64File = "$ksFull.base64.txt"
[Convert]::ToBase64String([IO.File]::ReadAllBytes($ksFull)) | Set-Content -NoNewline $b64File

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host " Keystore created: $Keystore"
Write-Host " Base64 written to: $b64File"
Write-Host "==================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Add these GitHub repo Secrets (Settings -> Secrets and variables -> Actions):"
Write-Host "  ANDROID_KEYSTORE_BASE64    = contents of $b64File"
Write-Host "  ANDROID_KEYSTORE_PASSWORD  = your store password"
Write-Host "  ANDROID_KEY_ALIAS          = $Alias"
Write-Host "  ANDROID_KEY_PASSWORD       = your key password (same as store if you pressed Enter)"
Write-Host "  PLAY_SERVICE_ACCOUNT_JSON  = the Play Console service-account JSON"
Write-Host ""
Write-Host "Then DELETE $b64File and store the .keystore + passwords in a password manager." -ForegroundColor Yellow
