---
name: reference-culture-sensitive-diagnostics
description: CA1305 does NOT flag plain $"" interpolation; invariant P0 renders "25 %" with a space; the repo form is string.Create(CultureInfo.InvariantCulture, $"...")
metadata:
  type: reference
---

Measured on `XC-64` (2026-08-16), fixing culture-sensitive diagnostic text in `Sources/Main/Statistics`
and `Sources/Anomalies`.

**The renderings, so an assertion can be written right the first time.** `0.25:P0` is **`25 %`** (ordinary
space) under `InvariantCulture` — which is what a Linux container gets from `LANG=C.UTF-8` — **`25%`** under
`pl-PL`, and **`%25`** under `tr-TR`. `1.5:G4` is `1.5` invariant, `1,5` under both. So `pl-PL` alone covers
decimal separator + percent spacing, `tr-TR` adds percent *position*. `F0` on a whole number is identical
everywhere, so a mutation on an `F0` site cannot redden a culture test.

**CA1305 would NOT have caught this, and that is measured, not assumed.** Building `Anomalies.csproj` with
`-p:AnalysisMode=All -p:TreatWarningsAsErrors=false -t:Rebuild` produced exactly **2** CA1305 diagnostics in
the whole project, both on `StringBuilder.Append(ref AppendInterpolatedStringHandler)` — and **neither** was
one of the ~23 defective `$"{x:P0}"` sites. The BCL rule sees the StringBuilder overload and is blind to
plain string interpolation. A guard for site 27 therefore has to be an `OVERFIT0xx` analyzer.

**The form this repo uses is `string.Create(CultureInfo.InvariantCulture, $"…")`**, not
`FormattableString.Invariant`: it keeps the `DefaultInterpolatedStringHandler` lowering (no boxing, no
`object[]`), it covers holes with no format specifier, and it already existed at
`Sources/Main/Audio/AudioSimilarityReport.cs`. Concatenated interpolated strings (`$"a" + $"b"`) bind to the
handler fine; make every operand `$"…"` rather than mixing in a plain literal.

Related: [[reference-test-output-and-anchors]], [[reference-coverage-runsettings-scope]].
