---
name: reference-commit-lands-mid-mutation
description: A commit can land WHILE a mutation harness has a file mutated — measured 2026-08-15; verify HEAD's copy after every mutation run, not just the working tree.
metadata:
  type: reference
---

The tree here is shared and the user commits while agents work. Measured 2026-08-15 during `XC-54`:
commit `1d1eff8` (22:47:36) landed in the middle of my run and swept in my in-progress
`DecodePoolIdleBurnTests.cs`. It happened to catch the golden version, and
`Sources/Main/Runtime/OverfitParallel.cs` was mutated only in a later window — but nothing prevented the
mutant from being committed.

**So the restore check has a second half**: after restoring, verify the *committed* copy too, not only the
working tree — `git show HEAD:<path> | grep <anchor>`. A working tree that matches your saved bytes says
nothing about what got committed while you were mutated.

Second, smaller trap from the same run: a file rewritten with **identical** bytes shows as ` M` in
`git status` (mtime refreshed) while `git diff` is empty. Content equality is the claim; check it with
`git diff --stat`, not with `git status`.

Related: [[reference-navigator-index-stale-after-move]] — the navigator's held DLL makes every
solution-wide `dotnet build` exit 1 on MSB3021/3027 in `Tools/SemanticNavigator`; build
`./Tests/Tests.csproj` instead, which pulls `Main`.
