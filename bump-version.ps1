# Copyright (c) 2026 DevOnBike.
#
# Bumps the SINGLE-SOURCE package/assembly version in Directory.Build.props. Every project inherits it
# (Main / Server / Extensions.AI library packages + the Cli global tool), so this one edit re-versions the
# whole solution and keeps them in lockstep — no per-project <Version> to drift.
#
#   .\bump-version.ps1                # increment the PATCH segment (10.0.22 -> 10.0.23)
#   .\bump-version.ps1 minor          # 10.0.22 -> 10.1.0
#   .\bump-version.ps1 major          # 10.0.22 -> 11.0.0
#   .\bump-version.ps1 10.1.0         # set explicitly
#   .\bump-version.ps1 -DryRun        # show what would change, write nothing
#
# Does NOT commit / tag / push — that stays in your hands. The Docker Hub + NuGet workflows publish on a
# `v<version>` tag, so the script prints the matching tag command.
param(
    [string]$Bump = 'patch',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$props = Join-Path $root 'Directory.Build.props'

if (-not (Test-Path $props))
{
    throw "Directory.Build.props not found at $props"
}

$text = [System.IO.File]::ReadAllText($props)
$match = [regex]::Match($text, '<Version>\s*(\d+)\.(\d+)\.(\d+)\s*</Version>')
if (-not $match.Success)
{
    throw "No numeric <Version> (x.y.z) found in Directory.Build.props"
}

$current = "$($match.Groups[1].Value).$($match.Groups[2].Value).$($match.Groups[3].Value)"
$major = [int]$match.Groups[1].Value
$minor = [int]$match.Groups[2].Value
$patch = [int]$match.Groups[3].Value

if ($Bump -match '^\d+\.\d+\.\d+$')
{
    $target = $Bump
}
else
{
    switch ($Bump.ToLowerInvariant())
    {
        'major' { $target = "$($major + 1).0.0" }
        'minor' { $target = "$major.$($minor + 1).0" }
        'patch' { $target = "$major.$minor.$($patch + 1)" }
        default { throw "Unknown bump '$Bump'. Use: major | minor | patch | x.y.z" }
    }
}

if ($current -eq $target)
{
    Write-Host "Already at $target — nothing to do."
    exit 0
}

if (-not $DryRun)
{
    $text = [regex]::Replace($text, '<Version>\s*[\d][^<]*</Version>', "<Version>$target</Version>")
    $text = [regex]::Replace($text, '<FileVersion>\s*[\d][^<]*</FileVersion>', "<FileVersion>$target</FileVersion>")
    [System.IO.File]::WriteAllText($props, $text)   # UTF-8 (no BOM), preserves line endings
}

$verb = if ($DryRun) { 'Would bump' } else { 'Bumped' }
Write-Host "$verb version: $current -> $target   (Directory.Build.props — all projects inherit)" -ForegroundColor Green
if (-not $DryRun)
{
    Write-Host "next: git commit -am `"chore: release v$target`"  &&  git tag v$target  &&  git push --tags"
}
