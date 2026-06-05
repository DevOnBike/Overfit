@echo off
rem Bumps the package <Version> in every versioned *.csproj. See bump-version.ps1 for options.
rem   bump-version.cmd            increment the patch of each versioned project
rem   bump-version.cmd 10.1.0     set every versioned project to 10.1.0
rem   bump-version.cmd -DryRun    preview, write nothing
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0bump-version.ps1" %*
