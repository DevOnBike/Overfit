---
name: dotnet-nugets-update
description: Claude Code skill for checking and updating NuGet packages in a .NET 10 solution using native CLI commands.
---

# .NET 10 NuGet Updates

Use this skill when the user asks to check, list, update, upgrade, or audit NuGet package dependencies in a .NET 10 repository.

## Instructions

1. **Find the Solution:** Locate the `.sln` file in the current repository. If there are multiple, ask the user which one to use.
2. **Check for Updates:** Run `dotnet package list --outdated --format json --output-version 1` against the solution. 
   * *Note: If the user specifically asks to check for security vulnerabilities, use `--vulnerable` instead of `--outdated`.*
3. **Report:** Parse the output and show the user a clean, readable table of packages that have available updates (Current Version vs Latest Version).
4. **No Updates?** If everything is up to date, simply reply in green text: `no updates needed` and exit.
5. **Ask for Confirmation:** If updates are found, **YOU MUST ASK** the user if they want to proceed with the updates. Do not run any update commands without explicit consent.
6. **Update:** Once confirmed, use the native .NET 10 command: `dotnet package update --project <path-to-project>` for each project in the solution that needs updates.
7. **Verify:** After updating, run `dotnet restore`, followed by `dotnet build`, and inform the user if the build succeeded.

## Rules
- Strictly use the .NET CLI (`dotnet package list` / `dotnet package update`). Do not manually edit XML `.csproj` files unless CLI fails.
- Default to stable package versions unless the user explicitly asks for prerelease.