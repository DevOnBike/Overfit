$ErrorActionPreference = "Stop"

Write-Host "[>] Executing NuGet update skill ..." -ForegroundColor Cyan

claude "run dotnet-nuget-updates"