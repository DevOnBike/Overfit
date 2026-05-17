param (
    [string]$SolutionPath = ""
)

# Konfiguracja terminala (wykonuje się po zadeklarowaniu parametrów)
Set-ExecutionPolicy Unrestricted -Scope Process -Force
cls

if ([string]::IsNullOrWhiteSpace($SolutionPath)) {
    $slnFiles = Get-ChildItem -Filter *.sln -File
    if ($slnFiles.Count -eq 0) {
        Write-Host "Błąd: Nie znaleziono pliku .sln w bieżącym katalogu." -ForegroundColor Red
        exit 1
    }
    $SolutionPath = $slnFiles[0].FullName
}

Write-Host "Rozpoczynam formatowanie kodu..." -ForegroundColor Cyan

Write-Host "Krok 1/3: Korekta białych znaków (whitespace)..." -ForegroundColor Yellow
dotnet format whitespace $SolutionPath

Write-Host "Krok 2/3: Aplikowanie ogólnych reguł stylu..." -ForegroundColor Yellow
dotnet format style $SolutionPath

Write-Host "Krok 3/3: Weryfikacja nagłówków plików (IDE0073)..." -ForegroundColor Yellow
dotnet format style $SolutionPath --diagnostics IDE0073

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSukces! Cała solucja została sformatowana." -ForegroundColor Green
} else {
    Write-Host "`nWystąpiły błędy podczas formatowania." -ForegroundColor Red
    exit $LASTEXITCODE
}