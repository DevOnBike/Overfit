---
name: raw-string-literals-keep-file-eol
description: C# raw string literals ("""...""") carry the FILE's line endings verbatim; git autocrlf on this box makes a repo-LF .cs check out CRLF on Windows, so escaped "\n" find-strings stop matching.
metadata:
  type: reference
---

A C# raw string literal does **not** normalise line endings — the runtime string carries whatever the
`.cs` file has on disk. Measured 2026-08-16 by converting
`Tests/TestSupport/Assemblies/BreakingChangeClassifierTests.cs` (LF in git, 775 lone LF, 0 CRLF) to CRLF:
`BreakingChangeClassifierTests.NewTypeAdded_IsAdditive` went red with
`String: "namespace Sample\r\n{\r\n..."` / `Not found: "public class Store\n    {\n    }"`, reproducing the
`windows-latest` CI failure exactly (1 of 33 in the class).

Why it bites here: `.gitattributes` pins `*.ps1` CRLF and `*.sh` LF and **says nothing about `*.cs`**, and
git on this box warns `LF will be replaced by CRLF the next time Git touches it` — so a Windows checkout of
an LF-in-repo `.cs` is CRLF. Any test that mixes a raw literal (file endings) with an escaped literal
containing `\n` is then platform-dependent.

Fix shape that worked: normalise once in the helper —
`source.Replace("\r\n", "\n", StringComparison.Ordinal)` — and pin it with a test whose CRLF is written
**explicitly** rather than inherited from the file, otherwise nothing guards it on an LF checkout (measured:
with the normalisation removed, 0 of the 33 pre-existing tests fail on LF; the new one does).

Related: [[reference_test_output_and_anchors]] (anchors need CRLF in a CRLF tree — same family, opposite
direction).
