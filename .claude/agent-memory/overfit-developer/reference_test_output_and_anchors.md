---
name: test-output-and-anchors
description: Two instrument traps on this box — the dotnet test summary language (Polish until DOTNET_CLI_UI_LANGUAGE=en was set on 2026-08-13; check, do not assume), and mutation anchors written with \n matching 0 times against this tree's CRLF files
metadata:
  type: reference
---

**The `dotnet test` summary language is not fixed — check it, do not assume either way.** It was Polish
until `DOTNET_CLI_UI_LANGUAGE=en` was set on 2026-08-13, and English is what six runs printed that day:

```
Passed!  - Failed:     0, Passed:  2610, Skipped:   273, Total:  2883
Powodzenie!    — niepowodzenie:     0, powodzenie:  2597, pominięto:   273, łącznie:  2870
```

A filter matching **both** costs nothing and survives the setting being changed back.

A Python filter keeping lines containing `Passed`/`Failed`/`Total` therefore prints **nothing at all**, and
an empty filtered output is indistinguishable from a clean run — which is the exact failure
`CLAUDE.md` warns about ("twice now a real failure has been lost because the filter kept only the summary").
Match `niepowodzenie:` / `Powodzenie!`, or keep both languages. Cost 2026-08-12: two round trips on `AN-D14`
before a single number appeared.

**Mutation anchors must be normalised for CRLF.** `Sources/Anomalies/Incidents/AnomalyGuard.cs` (and most of
this tree) is CRLF, while a multi-line anchor typed into the harness carries `\n` — so the count printed by
guard 2 is **0**, six arms in a row, and nothing runs. The fix, and it keeps the byte-verified restore intact:

```python
ORIGINAL = TARGET.read_bytes()
CRLF = b"\r\n" in ORIGINAL
text = ORIGINAL.decode("utf-8").replace("\r\n", "\n")     # match and replace here
TARGET.write_bytes((text.replace("\n", "\r\n") if CRLF else text).encode("utf-8"))
```

The [[overfit-mutate]] template does not do this; its pitfall table names the symptom but the code does not
guard it. Related: [[mutation-theory-test-names]].

**Line endings are per-file, not per-tree — including inside one directory.** Measured 2026-08-14 while
editing five citation sites for `PB-12`: `docs/performance-discipline.md` is **CRLF** (196 lines) while
`docs/measured-baselines.md` (503) and `docs/code-patterns.md` (171) in the same directory are **LF-only**,
as is `Sources/Main/Runtime/README.md` next to `OverfitParallel.cs`'s 982 CRLF. Detect per file
(`b"\r\n" in raw`) and write back with that file's own ending; a blanket "this tree is CRLF" produces a
whole-file diff on roughly half of what you touch.

**A file created with the `Write` tool lands LF-only in this CRLF tree, and `.gitattributes` will not fix
it** — it pins only `*.ps1 text eol=crlf` and `*.sh text eol=lf`, nothing for `*.cs`. Measured 2026-08-14 on
`XC-50`: two brand-new `.cs` files were LF-only (135 and 152 LF, 0 CRLF) next to `OverfitParallel.cs`'s 954
CRLF. They compile and test fine, so nothing catches it; the cost is that they become the two files where a
`\n` anchor works and everything else's does not. Normalise a newly-Written `.cs` before finishing:
`data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")` written back in binary mode. Single-line anchors
sidestep the whole question and are worth preferring when the mutation allows.
