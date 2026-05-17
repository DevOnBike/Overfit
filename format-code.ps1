param (
    [string]$SolutionPath = ""
)

# Terminal configuration (runs after the parameters are declared)
Set-ExecutionPolicy Unrestricted -Scope Process -Force
cls

if ([string]::IsNullOrWhiteSpace($SolutionPath)) {
    $slnFiles = Get-ChildItem -Filter *.sln -File
    if ($slnFiles.Count -eq 0) {
        Write-Host "Error: No .sln file found in the current directory." -ForegroundColor Red
        exit 1
    }
    $SolutionPath = $slnFiles[0].FullName
}

Write-Host "Starting code formatting..." -ForegroundColor Cyan

Write-Host "Step 1/3: Whitespace correction..." -ForegroundColor Yellow
dotnet format whitespace $SolutionPath

Write-Host "Step 2/3: Applying general style rules..." -ForegroundColor Yellow
dotnet format style $SolutionPath

Write-Host "Step 3/3: Verifying file headers (IDE0073)..." -ForegroundColor Yellow
dotnet format style $SolutionPath --diagnostics IDE0073

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSuccess! The entire solution has been formatted." -ForegroundColor Green
} else {
    Write-Host "`nErrors occurred during formatting." -ForegroundColor Red
    exit $LASTEXITCODE
}
