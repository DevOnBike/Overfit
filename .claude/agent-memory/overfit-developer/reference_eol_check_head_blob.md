---
name: eol-check-head-blob
description: this tree is genuinely mixed — Anomalies.csproj is LF on disk and in HEAD despite core.autocrlf=true — so never blanket-normalise a file to CRLF; compare against `git show HEAD:<path>` first
metadata:
  type: reference
---

**"Normalise a newly written file to CRLF" is wrong as a blanket rule here.** Measured 2026-08-16:
`core.autocrlf = true`, and `Sources/Anomalies/BannedSymbols.txt` and the `.cs` files beside it are CRLF
on disk — but `Sources/Anomalies/Anomalies.csproj` is **LF on disk and LF in the HEAD blob** (CRLF=0,
LF=84). Applying the rule converted the whole file, which `git diff` did **not** show (autocrlf normalises
on read), so the only way to see it was counting bytes.

**Procedure before touching line endings:** read `git show HEAD:<path>` as bytes, count `\r\n` versus lone
`\n`, and match that. The Edit tool already matches the file's dominant ending for inserted text, so in
practice a normalisation pass after an Edit is unnecessary and is itself the risk.

**A NEW `.cs` written LF-only needs no CRLF pass — the committed form here is LF.** Measured 2026-08-17
on four blobs (`HotPathStringAnalyzer.cs`, `JaggedFloatArrayTypeAnalyzer.cs`, `AnalyzerHarness.cs`,
`JaggedFloatArrayTypeAnalyzerTests.cs`): every one is **CRLF=0** in `git show HEAD:<path>`, while some of
the same files are CRLF *on disk* after checkout. So a file the Write tool creates LF-only matches HEAD
exactly and produces a clean `git diff`; converting it to CRLF is the thing that would create noise.
**Consequence for a mutation harness on a file you just created: anchors use `\n`, not `\r\n`** — the
opposite of the tree-wide default in [[test-output-and-anchors]]. Check the file's bytes, not the rule.

`.gitattributes` pins only `*.ps1` (crlf) and `*.sh` (lf) — everything else is whatever happened to be
written. Related: [[raw-string-literals-keep-file-eol]].
