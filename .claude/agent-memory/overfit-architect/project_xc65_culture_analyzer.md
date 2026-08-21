---
name: xc65-culture-analyzer
description: 2026-08-17 XC-65 OVERFIT047 culture-sensitive interpolation analyzer — signed; measured Roslyn operation shapes and culture formatting, both of which overturn the brief's premises
metadata:
  type: project
---

`docs/specs/xc-65-culture-invariant-interpolation-analyzer-plan.md`, `STATUS: APPROVED`, signed 2026-08-17.
Rule `OVERFIT047`, category Reliability, in `Sources/Analyzers`.

**Two measurements that overturned the brief, both executed (throwaway console probe outside the repo,
`Microsoft.CodeAnalysis.CSharp` 5.0.0, .NET 10 / ICU, Windows dev box; probe deleted after).**

1. **`sb.Append($"{d:F2}")` and `string.Create(InvariantCulture, $"{d:F2}")` have the IDENTICAL Roslyn
   operation shape** — both become `IInterpolatedStringHandlerCreationOperation` whose parts are
   `InterpolatedStringAppendFormatted`, **not** `IInterpolationOperation`. The obvious implementation
   (walk `Parts`, handle `IInterpolationOperation`) gets the `string.Create` exemption *by accident* and
   is blind to every `StringBuilder.Append`/`AppendLine` — **12 of ~24 remaining `Sources/Main` sites**.
   Distinguish by asking whether the enclosing invocation's `TargetMethod` has an `IFormatProvider`
   parameter, not by the operation kind.
2. **"An `int` renders identically under every culture" is FALSE for negatives.** `$"{-5}"` is `-5`
   invariant/`pl-PL`/`tr-TR` but `−5` (`U+2212`) under `sv-SE`/`lt-LT`/`fi-FI` and `U+061C`-prefixed under
   `ar-SA`. Accepted as a residual (flagging every integral hole is noise), written into the rule's own
   `description`. `:D`/`:X` and positive ints ARE invariant; `:N0` is not.

**MY §3 WAS WRONG AND WAS AMENDED THE SAME DAY — the mistake shape is the one to remember.** I measured one
specifier per type and then generalised to "fires regardless of format specifier" for reals, dates AND
`TimeSpan`. Re-measured (twice, independently, invariant vs `pl-PL`/`ar-SA`/`sv-SE`/`fi-FI`/`tr-TR`,
ordinal): dates are **byte-identical under `o O s u R r`**; `TimeSpan` moves **only under `g`/`G`** (none,
`c`, `t`, `T` are the invariant constant format, and it writes ASCII `-`, so no `U+2212` residual there);
reals move under every specifier. It produced a **false positive on `SuppressionStore.cs:58`**
(`{suppression.Until:u}`) — correct code, in the directory armed at `error`. **`R` is the proof the
predicate cannot be simplified**: culture-invariant for a date, culture-sensitive for a `double`. So the
key is the PAIR (type family × specifier). *Generalising from one measured cell to a whole row is the
failure; measure the row.*

**Also corrected: a mutation that can only SUBTRACT diagnostics cannot be pinned by an `Assert.Empty`.** My
M1 said "if rows 9/11/15 also move, the exemption was working by accident" — unfalsifiable by construction,
since those rows expect no diagnostics. The real arm is forcing the `IFormatProvider` check to `false`
(M2b), which reddens exactly the exemption rows. Pair every "reports less" mutation with a test expecting a
diagnostic.

**`double?` reports `SpecialType.None`** — the nullable unwrap is a separate step or the whole nullable
family is silently missed. `TimeSpan`/`DateTimeOffset` also `None`; resolve by metadata name against the
compilation (no repo reference needed — every type involved is BCL).

**Severity ladder was decided by `aot-guard`, not taste.** `warning` is unavailable as an intermediate
step anywhere with live sites: `aot-guard` publishes with `TreatWarningsAsErrors=true` across the whole
graph, and the escape hatch is barred because an id in `WarningsNotAsErrors`
(`Directory.Build.props:47`) **also overrides a directory-scoped `error`** (measured 2026-07-19, it
neutered OVERFIT001/002/009/900). `suggestion` is invisible in build output. So: `none` under `[*.cs]`,
`error` in the two zero-site directories (`Sources/Anomalies`, `Sources/Main/Statistics`).

Also: the two `CA1305` hits `XC-64` judged benign really are benign under this predicate
(`AnomalyGuardConfigReader.cs:347` = `int` `Count`; `MetricMap.cs:240` = strings + enums) — a cheap
cross-check that the predicate is not just CA1305 inverted. And the exception path must NOT be exempt
(3 of XC-64's 42 sites are inside `throw new ArgumentException`), unlike OVERFIT006/014 which do exempt it
— copying a neighbouring analyzer's skeleton wholesale would have dropped exactly those three.

**AMENDMENT 2 (same day, from `XC-72`'s sweep): `F0` on a real hole is NOT invariant, and the tie-breaker
that was supposed to settle such questions returns the WRONG answer.** Proposal was "F is separator-free at
precision 0, so it is invariant for every non-negative real — exempt it". Measured (.NET 10.0.11 / ICU, this
box, ordinal vs invariant, `pl-PL ar-SA sv-SE fi-FI tr-TR lt-LT`): finite non-negative values ARE identical,
but **`+Infinity` → `∞` in all six** and **`NaN` → `epäluku` (fi-FI) / `ليس رقم` (ar-SA)** — neither is
negative. Also `-0.0` and `-0.4` render `-0` → `−0`. `f0`/`F00`/`F0000` identical to `F0`; **bare `F` is not
a candidate** (precision comes from `NumberDecimalDigits`, 2 invariant vs 3 in five cultures). Rejected as
D8; the two sites get pragmas justified as **"finite and non-negative"** — non-negativity alone does not
exclude `+Infinity`.

Two durable pieces:

1. **The tie-breaker's culture set decides the answer, and `pl-PL` alone is blind.** `pl-PL`'s
   `NegativeSign` is `U+002D` and its `NaNSymbol` is `NaN`, so §7 as first written ("invariant vs `pl-PL`")
   plus the obvious value set declares `F0` invariant. Minimum set is invariant + `pl-PL` + `fi-FI`
   (`U+2212` **and** `epäluku`) + `ar-SA`; values must include a negative, a zero, a large one, and for
   reals `NaN` and `±Infinity`.
2. **The criterion that separates a rule defect from a pragma**, and it subsumes the `0.##`/`G6` question:
   *a hit is a rule defect only when the (type × specifier) pair is invariant for **every value the type can
   represent**; invariance for the values that site produces is a pragma.* `SuppressionStore.cs:58` was the
   first kind, `{ParameterCount/1e6:F0}` is the second.
3. **D2 is not a precedent for every residual.** D2 = one culture datum, on a family with no non-finite
   values, buying off thousands of `{count}` sites. D8 = three residual classes buying off two sites. Ratio
   inverted at both ends — say which, do not just cite the earlier decision.

Also: a **file-based app (`dotnet run probe.cs`) placed under `D:\Overfit\` picks up `Directory.Build.props`
and tries to rebuild `Sources/Analyzers`** — which fails MSB3027 when another agent's build holds the dll.
Probes go outside the repo (`D:\_scratch\...`, never `C:`), and are deleted after.

See [[reference-measured-baselines]] for the CA1305 capability row (`docs/measured-baselines.md:534-553`).
