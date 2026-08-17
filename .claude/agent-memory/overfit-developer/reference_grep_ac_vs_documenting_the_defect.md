---
name: reference-grep-ac-vs-documenting-the-defect
description: An acceptance criterion of the form "grep X finds nothing" collides with the same plan's order to document defect X; both times the surviving match was correct text.
metadata:
  type: reference
---

A plan that says **"`rg X` must find no match in <dir>"** collides with any instruction to *write down*
what X was — and on `XC-51` (2026-08-15) it collided **twice in one task**, both times with text that
should stay:

| AC | surviving match | why it stays |
|---|---|---|
| `rg "Has\{0\}" Sources/Analyzers` -> 0 | `Sources/Analyzers/README.md` authoring-notes bullet | the same plan's D3 ordered that bullet, and it quotes the dead composition to make the rule legible |
| `rg -i "msbuild guard" Sources/Analyzers` -> 0 | `JaggedFloatArrayTypeAnalyzer.cs` doc: *"where the old MSBuild guard **used to** fail the build"* | past tense — a record, and D2 forbids correcting records |

**Do not reword to beat the grep and do not weaken the AC.** Report the match, classify it (advice vs
record vs deliberate documentation), and give the narrowed check that does hold — here
`git grep "Has{0}" -- 'Sources/Analyzers/*.cs'` returns nothing, which is the property the AC was
actually written to pin.

The general shape: **a text-absence criterion cannot distinguish the defect from a description of the
defect.** When an AC is phrased that way, the scope it means is nearly always the *mechanism* file
(`*.cs`), not the whole directory. See [[reference-doc-diagnostics-on-arrival]] for the other class of
"prose is checked for the first time" surprise.
