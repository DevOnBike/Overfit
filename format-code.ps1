param (
    [string]$SolutionPath = "",

    # CI mode: report formatting drift instead of rewriting files (non-zero exit if anything would change).
    [switch]$Check
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($SolutionPath)) {
    $slnFiles = Get-ChildItem -Filter *.sln -File
    if ($slnFiles.Count -eq 0) {
        Write-Host "Error: No .sln file found in the current directory." -ForegroundColor Red
        exit 1
    }
    $SolutionPath = $slnFiles[0].FullName
}

# --verify-no-changes turns each step into a check instead of a rewrite.
$extraArgs = @()
if ($Check) { $extraArgs += '--verify-no-changes' }

$mode = if ($Check) { "Checking" } else { "Formatting" }
Write-Host "$mode $SolutionPath" -ForegroundColor Cyan

# Each step is verified on its own: $LASTEXITCODE only ever reflects the LAST command, so a single
# check at the end would report success even when an earlier step had failed.
function Invoke-FormatStep {
    param([string]$Label, [string]$Subcommand)

    Write-Host "  $Label..." -ForegroundColor Yellow
    dotnet format $Subcommand $SolutionPath @extraArgs

    if ($LASTEXITCODE -ne 0) {
        $why = if ($Check) { "would be reformatted" } else { "failed" }
        Write-Host "`n$Label $why (exit $LASTEXITCODE)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# whitespace = indentation / spacing only.
# style      = the .editorconfig IDE rules, INCLUDING the IDE0073 file header (it is severity=warning in
#              .editorconfig and `dotnet format style` fixes warn-and-above by default), so the header
#              needs no separate pass.
# `dotnet format analyzers` is deliberately NOT run here: it applies analyzer code fixes, which would mix
# behavioural rewrites into what should be a formatting-only change. Run it by hand if you want it.
Invoke-FormatStep -Label "Step 1/2: whitespace" -Subcommand "whitespace"
Invoke-FormatStep -Label "Step 2/2: style rules (incl. IDE0073 headers)" -Subcommand "style"

if ($Check) {
    Write-Host "`nSuccess! Formatting is already correct." -ForegroundColor Green
} else {
    Write-Host "`nSuccess! The entire solution has been formatted." -ForegroundColor Green
}
