# Copyright (c) 2026 DevOnBike.
#
# Bumps the package <Version> in every *.csproj that declares one (packable projects:
# DevOnBike.Overfit, DevOnBike.Overfit.Extensions.AI, ...). Test/benchmark/demo/CLI projects
# have no <Version> and are skipped automatically.
#
#   .\bump-version.ps1                # increment the PATCH segment of each project's version
#   .\bump-version.ps1 10.1.0         # set every versioned project to 10.1.0
#   .\bump-version.ps1 -DryRun        # show what would change, write nothing
#
param(
    [string]$NewVersion,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$projects = Get-ChildItem -Path $root -Recurse -Filter *.csproj -File |
    Where-Object { $_.FullName -notmatch '[\\/](bin|obj)[\\/]' }

$pattern = '<Version>(\d+)\.(\d+)\.(\d+)</Version>'
$changed = @()

foreach ($project in $projects)
{
    $text = [System.IO.File]::ReadAllText($project.FullName)
    $match = [regex]::Match($text, $pattern)
    if (-not $match.Success)
    {
        continue
    }

    $current = "$($match.Groups[1].Value).$($match.Groups[2].Value).$($match.Groups[3].Value)"
    if ($NewVersion)
    {
        $target = $NewVersion
    }
    else
    {
        $target = "$($match.Groups[1].Value).$($match.Groups[2].Value).$([int]$match.Groups[3].Value + 1)"
    }

    if ($current -eq $target)
    {
        continue
    }

    if (-not $DryRun)
    {
        $updated = [regex]::Replace($text, $pattern, "<Version>$target</Version>")
        [System.IO.File]::WriteAllText($project.FullName, $updated)   # UTF-8 (no BOM), preserves line endings
    }

    $changed += "  $($project.Name): $current -> $target"
}

if ($changed.Count -eq 0)
{
    Write-Host 'No <Version> changes (no versioned csproj found, or already at target).'
    exit 0
}

$verb = if ($DryRun) { 'Would bump' } else { 'Bumped' }
Write-Host "$verb $($changed.Count) project(s):"
$changed | ForEach-Object { Write-Host $_ }
