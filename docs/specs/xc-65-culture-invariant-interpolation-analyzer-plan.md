STATUS: APPROVED
Author: overfit-architect (both halves — see §0)
Architecture review: overfit-architect, 2026-08-17 — SIGNED
Amended: 2026-08-17 by overfit-architect — §3's predicate was WRONG for `DateTime`/`DateTimeOffset` and
  `TimeSpan` and produced a false positive; refuted by `overfit-developer` during implementation and
  re-measured independently before the amendment landed. Changes: Finding 8 (new), §3 rewritten as four
  families, §5 description bullet, §7 precondition, §9 rows 20-21, §10 M1 corrected + M2b/M8/M9 added,
  D1 corrected, D7 added. **Nothing else in the plan is affected and no earlier reasoning was deleted.**
Date: 2026-08-17
Slug: xc-65-culture-invariant-interpolation-analyzer-plan

GATES:
  performance:        NOT_REQUIRED — no performance target, claim or comparison enters this plan. The
                       artefact is a Roslyn analyzer in `Sources/Analyzers`, which runs in the compiler and
                       ships nothing; it touches no execution path in the product. Analyzer *compile-time*
                       cost is not a product performance claim, but if a build-time figure is produced for
                       any reason it must not be reported as one — hand it to `overfit-perf-claim-auditor`
  security:           NOT_REQUIRED — no parser, endpoint, gateway, tokenizer or externally-fed surface. The
                       analyzer reads syntax trees the compiler already parsed
  leak-scan:          NOT_REQUIRED — no config, log, fixture, host name, path or token touched
  AOT:                NOT_REQUIRED — `Sources/Analyzers` targets `netstandard2.0`, is wired as
                       `OutputItemType="Analyzer"` with `ReferenceOutputAssembly="false"`
                       (`Directory.Build.props:80-82`) and is therefore never in any published output.
                       Nothing new becomes reachable from `Tests/AotSmokeTest`. **But see Finding 6** —
                       the *severity* chosen interacts with the `aot-guard` job, which is an AOT-adjacent
                       constraint on this plan even though no AOT surface changes
  API-compatibility:  NOT_REQUIRED — nothing in `DevOnBike.Overfit.*` is added, removed or changed. The
                       analyzer assembly is not packed (`IsPackable=false`)
  release-readiness:  NOT_REQUIRED — build-time guard plus tests. No shipped behaviour, no packaging, no
                       dependency, no version-affecting surface. If the follow-up sweep of §8 phase 3 edits
                       product strings, that is a separate row and carries its own gate

`verifier`, `reviewer` and `mutation-proof` are **intentionally absent rather than `NOT_REQUIRED`**: a
missing line means "not yet asked". Nothing is implemented. `mutation-proof` is not optional here —
§10's table is part of the acceptance criteria.

---

# XC-65 — a build-time guard against culture-sensitive interpolation (`OVERFIT047`)

Plan file for task `XC-65` (`docs/TASKS.md:191`). The durable half of `XC-64` (`docs/TASKS.md:192`),
which fixed 42 sites and pinned them with an enumerated test that cannot see site 43.

## 0. Why the architect wrote both halves

There is no analysis for an analyst to do that is not already in the two task rows. The problem, the
measurement that rules out `CA1305`, the two traps and the scope candidate are all stated there and in
`docs/measured-baselines.md:534-553`. No business rule, acceptance threshold or value judgement is
missing, so there is nothing to send to a client and nothing to stop for. What was genuinely undecided is
technical and is decided below.

**Everything in §2 and §3 marked "measured" was executed on 2026-08-17**, on this Windows dev box, .NET 10
SDK, ICU globalization, using a throwaway console project outside the repository holding
`Microsoft.CodeAnalysis.CSharp` **5.0.0** — the same version `Sources/Analyzers` references
(`Directory.Packages.props`). The probe was deleted afterwards; the outputs are quoted here because the
quotation is the evidence, not the project.

---

## 1. Review of the problem statement — findings

**Finding 1 — the brief's premise "an `int` renders identically under every culture" is FALSE for negative
values, and I measured it.** Formatting `-5` with no specifier, .NET 10 / ICU:

| culture | `$"{-5}"` | `NumberFormatInfo.NegativeSign` |
|---|---|---|
| invariant, `pl-PL`, `tr-TR` | `-5` | `U+002D` |
| `sv-SE`, `lt-LT`, `fi-FI` | `−5` | **`U+2212`** |
| `ar-SA` | `‏-5` | `U+061C U+002D` |

Positive integers, `:D` and `:X` are identical across all seven cultures tested; `:N0` differs in every
one of them. **This does not change the decision** — see D2 — but it does change what the rule may claim,
and it goes in the rule's own `description` rather than being left implicit. Firing on every integral hole
would flag thousands of `{count}` sites to catch a defect no machine this project runs on can produce (the
dev box is `pl-PL`, CI is invariant, neither uses `U+2212`). That is the noise `XC-64` deliberately
avoided, and `.editorconfig` records the same failure twice.

**Finding 2 — the brief's design ("an interpolated string holding a numeric hole") silently misses the
largest single group of remaining sites, and this is the one thing that would have made the rule useless.**
Measured operation shapes:

| written | operation shape of the parts | current culture used? |
|---|---|---|
| `string s = $"{d:F2}";` | `Interpolation` | yes |
| `string.Create(CultureInfo.InvariantCulture, $"{d:F2}")` | `InterpolatedStringAppendFormatted` | no |
| `sb.Append($"{d:F2}")` | **`InterpolatedStringAppendFormatted`** | **yes** |
| `sb.Append(CultureInfo.InvariantCulture, $"{d:F2}")` | `InterpolatedStringAppendFormatted` | no |

The obvious implementation — walk `IInterpolatedStringOperation.Parts`, handle `IInterpolationOperation`,
ignore the rest — gets the `string.Create` exemption **for free and for the wrong reason**, and by the same
accident is blind to every `StringBuilder.Append`/`AppendLine` interpolation. Of the ~24 candidate sites I
counted in `Sources/Main` (§8), **12 are `sb.Append`/`sb.AppendLine`**: `Retrieval/Evaluation/RagAssert.cs`
6 sites, `LanguageModels/Runtime/DecodeProfiler.cs` and `PrefillProfiler.cs`, `Onnx/OnnxGraphModel.cs`. A
rule that shipped with that hole would have been proved "red on real defective code" against the other half
and would have looked correct.

**Finding 3 — the specifier-less holes ARE covered by the type check, verified rather than assumed.** The
three `AlertEngine` sites are `_config.AlertThreshold` / `_config.CriticalThreshold`
(`Sources/Anomalies/Alerting/AlertEngine.cs:97,106,107`). `find_references` on
`AlertEngineConfig.AlertThreshold` resolves the declaration at
`Sources/Anomalies/Contracts/AlertEngineConfig.cs:35`, and `Tests/Monitoring/AlertEngineTests.cs:27`
constructs it via `MakeConfig(float, float, TimeSpan?)` — the type is `float`. Measured: `$"{1.5}"` is
`1.5` invariant and `1,5` under `pl-PL`/`tr-TR`/`sv-SE`/`lt-LT`/`fi-FI`, `1٫5` under `ar-SA`. So a
type-keyed rule reaches exactly the sites a specifier-keyed rule and a human reviewer both miss. **The
answer to the brief's question 3 is: yes, the type check covers it, with no syntax on format specifiers
required for floating-point, `decimal` or `Half`.** *(Amended 2026-08-17 — this sentence originally
extended to `TimeSpan`, `DateTime` and `DateTimeOffset` and was wrong for them. See Finding 8.)*

**Finding 4 — the two `CA1305` hits `XC-64` judged benign are benign, and the new rule agrees with that
judgement.** `docs/measured-baselines.md:540-542` names them. Read this run:
`Sources/Anomalies/Monitoring/AnomalyGuardConfigReader.cs:347` interpolates `problems.Count` (an `int`, no
specifier) and `Sources/Anomalies/Monitoring/MetricMap.cs:240-241` interpolates strings and enums. Under
§3's predicate `OVERFIT047` fires on neither. That is a cheap cross-check that the predicate is not merely
"whatever `CA1305` does with the sign flipped": the two rules disagree about which sites matter and agree
about these two.

**Finding 5 — the exception path must NOT be exempt, unlike every neighbouring rule.** `OVERFIT006` and
`OVERFIT014` exempt `throw` construction via `OverfitPerfAnalysis.IsOnExceptionPath`, correctly, because
they are per-call allocation rules. Three of `XC-64`'s 42 fixed sites are inside `throw new
ArgumentException(...)`. Copying the neighbouring analyzers' skeleton wholesale would exempt exactly those
three. `OVERFIT047` is a determinism rule, not an allocation rule: it must not use
`OverfitPerfAnalysis.Report`, must not list `OverfitPerfAnalysis.HotPathRule` in `SupportedDiagnostics`,
and must not call `IsOneTimeAllocationContext` or `IsOnExceptionPath`. It reports through
`context.ReportDiagnostic` directly.

**Finding 6 — the severity ladder is constrained by `aot-guard`, not by taste, and this rules out the
obvious "warning while the backlog is worked down".** `Directory.Build.props:47` carries
`WarningsNotAsErrors` for the advisory `OVERFIT0xx` ids, because the `aot-guard` job publishes with
`TreatWarningsAsErrors=true` and that switch flows through the whole graph including `Sources/Main`. So a
`warning` severity on `OVERFIT047` anywhere in `Main`, with live sites, **fails `aot-guard`** — and the
escape hatch is barred: the same file records (measured 2026-07-19) that an id placed in
`WarningsNotAsErrors` **also overrides a directory-scoped `error`**, which silently neutered
`OVERFIT001/002/009/900`. `OVERFIT047` therefore must never enter that list. §7 takes the only remaining
consistent ladder.

**Finding 7 — no disagreement with the brief's scope instinct, one addition.** `Sources/Cli`,
`Sources/Server`, `Sources/Benchmark`, `AndroidBench` and `Demo/**` carry ~44 further candidate sites and
are console output read by a human at a terminal, where the ambient culture is the correct choice. They
stay at `none`, permanently and by decision rather than by backlog. That is recorded in §7 so nobody later
reads their absence as an unfinished sweep.

**Finding 8 — AMENDMENT 2026-08-17, after implementation. §3 as originally signed was WRONG for two of its
three families, and it produced a false positive on the only site in a directory §7 arms at `error`.**
Raised by `overfit-developer` during implementation and **re-executed independently by me** before this
amendment landed (same probe method as §0, .NET 10 / ICU, invariant baseline against `pl-PL`, `ar-SA`,
`sv-SE`, `fi-FI`, `tr-TR`, ordinal comparison; both runs agree on every cell):

| hole type | specifiers that MOVE | specifiers that are byte-identical in all five |
|---|---|---|
| `DateTime`, `DateTimeOffset` | none, `G g F d D T` | **`o O s u R r`** |
| `TimeSpan` | **`g G` only** | none, `c`, `t`, `T` (positive *and* negative values) |
| `float double decimal Half BigInteger` | every one tested — none, `F2 N0 G R E2 P1` | — |

The cost of getting this wrong was concrete, not theoretical:
`Sources/Anomalies/Monitoring/SuppressionStore.cs:58` writes `$"…{suppression.Until:u}…"`, which is
**already correct**, and the first build of the rule reported it. Arming at `error` on the signed
predicate would have forced a pragma onto correct code — the exact "teaches people the rule is noise"
failure Finding 1 rejects for integers, arriving through a family I had not questioned.

**The transferable part is why the mistake was reachable at all.** I measured the *types* (§0's culture
table used one specifier per type) and then generalised to *all* specifiers, which is a claim I never
tested. `R` is the case that makes the generalisation indefensible in both directions at once:
culture-**invariant** for a date, culture-**sensitive** for a `double`. So the predicate cannot key on the
specifier character alone, and it cannot key on the type alone either — it is (type family × specifier),
and §3 now says so.

**Also corrected: my M1 in §10 could not test what its second clause claimed.** See §10.

---

## 2. What the rule sees, and how it sees a hole's type with no reference to repository code

**It asks the compilation, and every type involved is a BCL type.** `Sources/Analyzers` holds no project
reference to repository code and must not gain one (`Directory.Build.props:57` — it runs inside the
compiler and ships nothing). Nothing in this design needs one:

- the hole's type comes from `IInterpolationOperation.Expression.Type`, an `ITypeSymbol` the semantic model
  already resolved. Measured: `HOLE-TYPE=double special=System_Double`, `HOLE-TYPE=System.TimeSpan
  special=None`, `HOLE-TYPE=double? special=None`, `HOLE-TYPE=T special=None` for an unconstrained type
  parameter;
- the four non-`SpecialType` types are resolved by metadata name against the compilation being analyzed —
  `System.TimeSpan`, `System.DateTimeOffset`, `System.Half`, `System.Numerics.BigInteger` — via
  `Compilation.GetTypeByMetadataName`, cached in a `CompilationStartAction`. A `null` result (the type is
  not in that compilation) simply means the rule cannot fire for it there;
- `System.IFormatProvider` likewise, for the exemption in §4.

**The two shapes, both measured.** Register once on `OperationKind.InterpolatedString` and handle both
kinds of part:

1. **`IInterpolationOperation`** (a plain `$"…"` whose target is `string`). `Expression.Type` is the hole
   type; `FormatString` is the specifier, whose `ConstantValue` is `"F2"` (the syntax text is `:F2` — use
   the constant, not the text). Raw interpolated literals (`$$"""… {{d:F2}} …"""`) produce the same shape;
   measured.
2. **`IInterpolatedStringAppendOperation` of kind `InterpolatedStringAppendFormatted`** (the literal was
   converted to an interpolated-string *handler*). `AppendCall` is an `IInvocationOperation` targeting
   `AppendFormatted<T>(…)`; the hole type is `Arguments[0].Value.Type` with conversions unwrapped, and the
   specifier is the argument bound to the parameter **named `format`** — located by parameter name, because
   the alignment overload `AppendFormatted<T>(T, int, string?)` shifts the position. `AppendLiteral` parts
   are skipped.

`IInterpolatedStringAdditionOperation` (`$"a{x:F1}" + $"b{x:F2}"`) contains one `IInterpolatedStringOperation`
per segment, so the action fires per segment inside it — measured, both as a bare concat and inside a
`string.Create`.

---

## 3. The predicate — (type family × format specifier)

**Amended 2026-08-17 after implementation. The original text said "fires regardless of format specifier"
for the real-number, date and `TimeSpan` families together; that is true only of the first. See Finding 8
for the measurement and for the false positive the original wording produced.**

**Family A — fires on every format specifier, and on none.** `float`, `double`, `decimal`, `System.Half`,
`System.Numerics.BigInteger`. Measured: no specifier, `F2`, `N0`, `G`, `R`, `E2`, `P1` all move between
the invariant culture and each of `pl-PL`, `ar-SA`, `sv-SE`, `fi-FI`, `tr-TR`.

**Family B — `System.DateTime`, `System.DateTimeOffset`: fires EXCEPT on the round-trip and
interchange specifiers `o O s u R r`**, which the BCL formats against `DateTimeFormatInfo.InvariantInfo`
whatever provider it is handed. Measured byte-identical in all five cultures. Everything else — no
specifier, `G g F d D T`, and any custom pattern — fires.

**Family C — `System.TimeSpan`: fires ONLY on `g` and `G`.** No specifier, `c`, `t` and `T` all render the
invariant constant format, measured byte-identical in all five cultures for a positive *and* a negative
value. (The constant format also writes an ASCII `-` rather than the culture's `NegativeSign`, so Finding
1's `U+2212` residual does not reach `TimeSpan`.)

**`R` is why this cannot be simplified back**: culture-invariant for family B, culture-sensitive for
family A. The decision is the pair, never the specifier alone and never the type alone.

`Nullable<T>` of any of the above unwraps to the same answer — measured: `double?` reports
`special=None`, so the implementation must unwrap `INamedTypeSymbol { OriginalDefinition.SpecialType:
System_Nullable_T }` and re-test `TypeArguments[0]`, or the whole nullable family is silently missed.

**Family D — integral types (`sbyte` … `ulong`, `nint`, `nuint`, `char`) and enums: fires only when a
culture-sensitive format specifier is present.** Culture-sensitive specifiers are, case-insensitively, the standard
`c e f g n p r`, and any custom specifier containing `.` `,` `%` or `‰`. `d`, `x`, `b` and an absent
specifier do not fire. Measured: `$"{1234567:N0}"` is `1,234,567` invariant, `1 234 567` under `pl-PL`,
`1.234.567` under `tr-TR`, `1٬234٬567` under `ar-SA`; `:D` and `:X` are byte-identical in all seven.

**Never fires**: `string`, `object`, `bool`, `Guid`, unconstrained type parameters, and any type the rule
does not name. A hole of type `string` is the shape produced by `{d.ToString(CultureInfo.InvariantCulture)}`
— measured `HOLE-TYPE=string` — so the recommended per-hole fix is exempted for free.

---

## 4. Exemptions, and what each one gives up

| exempt | why | measured shape |
|---|---|---|
| the enclosing invocation/object-creation of a handler-converted literal has a parameter of type `System.IFormatProvider` | the caller chose a provider; choosing the *wrong* one is a different rule | `string.Create(IFormatProvider?, ref DefaultInterpolatedStringHandler)`, `StringBuilder.Append(IFormatProvider?, ref AppendInterpolatedStringHandler)` |
| the literal is converted to `System.FormattableString` or `System.IFormattable` | the culture is chosen by the consumer, and the analyzer cannot see where | `FormattableString.Invariant($"…")` → `Conversion [System.FormattableString]` wrapping a plain `InterpolatedString` |
| `#pragma warning disable OVERFIT047` | the standard escape hatch, as everywhere else here | — |

**What the first row gives up, stated rather than discovered**: passing `CultureInfo.CurrentCulture`
explicitly is exempt. That is deliberate — the rule's subject is *"nobody chose"*, not *"somebody chose
badly"* — and it matches `CA1305`'s own semantics.

**Not exempt, deliberately**: the exception path (Finding 5), `ToString` overrides, one-time contexts,
`[OverfitHotPath]` escalation. None of those has anything to do with determinism.

**Reporting granularity — one diagnostic per interpolated-string expression, not per hole.** The fix is to
wrap the whole literal once, so one diagnostic matches one edit. When the parent is an
`IInterpolatedStringAdditionOperation`, walk to the outermost addition and report there once — the same
shape `HotPathStringAnalyzer.cs:57-61` already uses for concatenation chains. The message names the count
of offending holes and the first offending type, so a reader can find them.

---

## 5. The descriptor

```
Id           OVERFIT047                      (next free: 47 rules exist, OVERFIT001-046 + OVERFIT900)
Category     Reliability
Title        Culture-sensitive interpolation
Severity     Warning (default; the real severity is per-directory in .editorconfig — see §7)
Message      Interpolated string formats {0} with the ambient culture — the same build emits different
             bytes on different machines. Wrap it: string.Create(CultureInfo.InvariantCulture, $"…").
```

The suggested replacement must resolve — `XC-51`'s standing rule, and the reason
`DiagnosticMessageNamesResolveTests` exists. `string.Create(IFormatProvider, ref
DefaultInterpolatedStringHandler)` is a BCL method and `CultureInfo.InvariantCulture` a BCL property; both
are in use at all 42 sites `XC-64` fixed, and `Sources/Main/Audio/AudioSimilarityReport.cs:100` is the
precedent that predates it.

**The `description` must state what the rule does not catch** (the brief's question 3, generalised), in
these terms:

- a **negative** integral hole with no specifier — `U+2212` under `sv-SE`/`lt-LT`/`fi-FI`, `U+061C` prefix
  under `ar-SA`, measured 2026-08-17. Not flagged, because flagging every integral hole is noise;
- `{d.ToString("F2")}` — the hole's type is `string`, so this rule cannot see it. **`CA1305` does catch this
  one**; the two rules are complementary and neither subsumes the other;
- an unconstrained generic hole (`$"{v:F2}"` inside `M<T>`) — measured `special=None`, type `T`; the rule
  cannot know;
- **(added by the Finding 8 amendment)** a `DateTime`/`DateTimeOffset` under `o O s u R r`, or a `TimeSpan`
  under no specifier / `c` / `t` / `T`, is **not flagged and does not need to be** — those render against
  the invariant format info whatever provider is passed. The `description` must say this in the positive
  ("these specifiers are already invariant"), not as a limitation, or a reader will add a wrap that buys
  nothing. Note in the same breath that `R` is invariant for a date and culture-sensitive for a `double`;
- a value formatted through a provider the caller chose badly (§4).

---

## 6. Boundaries — the standing questions, answered so nobody guesses

| | |
|---|---|
| Execution path | **neither.** `Sources/Analyzers` runs inside the compiler; it is not inference and not training, and it touches no product execution path |
| Assembly | `Sources/Analyzers` (`netstandard2.0`, `DevOnBike.Overfit.Analyzers`). No new project, no new dependency anywhere |
| Dependency direction | unchanged. `Analyzers` references nothing in the tree and **must not**; §2 shows nothing here needs it |
| Public API surface | none. `IsPackable=false`; the analyzer is wired as `OutputItemType="Analyzer"`, `ReferenceOutputAssembly="false"` |
| AOT reachability | not reachable, and cannot become so. See the gate manifest for the one real AOT interaction (Finding 6) |
| Allocation policy | neither hot nor load path. Analyzer code follows the Roslyn convention (cache symbol lookups in a `CompilationStartAction` rather than per operation) — that is an authoring habit, not an Overfit allocation rule, and `Sources/Analyzers` is excluded from the `OVERFIT0xx` set anyway (`Directory.Build.props:63` — it would reference itself) |
| Ownership / disposal | no buffers, no `AutogradNode`, no tag needed |
| Threading | `EnableConcurrentExecution()`, as every analyzer here. All state is per-compilation and immutable after `CompilationStart` |
| Moat side | open. A build-time guard on this repository's own source is not a product capability |
| Source of truth | `.editorconfig` for severity, `AnalyzerReleases.Unshipped.md` for the rule's existence. Both are already the repository's convention |

---

## 7. Scope and severity — the ladder

```editorconfig
[*.cs]
dotnet_diagnostic.OVERFIT047.severity = none

[Sources/Anomalies/**.cs]
dotnet_diagnostic.OVERFIT047.severity = error

[Sources/Main/Statistics/**.cs]
dotnet_diagnostic.OVERFIT047.severity = error
```

Placement follows the `OVERFIT033`/`OVERFIT034` block (`.editorconfig:518-528`): `none` globally, `error`
exactly where the backlog is zero. The developer writes it with a comment block in that section's style
carrying (a) the `CA1305` measurement by reference to `docs/measured-baselines.md:534-553`, not restated,
(b) Finding 6, and (c) the list of directories deliberately excluded per Finding 7.

**Why `error` straight away and not `warning` first.** The repository's rule — a directory goes to `error`
only at zero sites — is satisfied for these two, and Finding 6 shows `warning` is not available as an
intermediate step anywhere with live sites without either breaking `aot-guard` or entering
`WarningsNotAsErrors`, which would silently disarm the `error` set here. `suggestion` is not an option
either: `.editorconfig:544-546` records that a build does not print suggestions, so counting a rule at
suggestion measures the config's silence. This is the `OVERFIT027`/`OVERFIT046` posture — a tripwire at
zero sites on a shape that has already produced one real CI failure.

**A precondition on both `error` lines, and it is not ceremony.** `XC-64`'s sweep was driven by reading
`$"` lines; this rule is semantic and reaches shapes that sweep did not enumerate — integral holes with an
`:N0`, and handler-converted `sb.Append` sites. **Both directories must be measured at zero with the rule
built and temporarily raised, before the `error` lines land.** If a site is found, it is fixed in the same
task; if more than three are found, stop and report — that would mean the predicate is wider than this plan
believes and the ladder needs re-deciding.

**Amended 2026-08-17, because this step fired and the plan mis-stated what to do with it.** The text above
assumes a hit is a defective *site*. The one hit in `Sources/Anomalies` —
`Monitoring/SuppressionStore.cs:58`, `{suppression.Until:u}` — was a defective *predicate* (Finding 8), and
the correct action was to narrow the rule, not to wrap correct code. **So the first question on any hit
here is which of the two it is, and the tie-breaker is a measurement, not a reading**: format the hole's
type and specifier under the invariant culture and under `pl-PL`, and compare ordinally. If the bytes are
identical the rule is wrong. That check takes a minute and it is the only thing standing between this
precondition and a pragma on correct code.

---

## 8. Order of work, and the backlog gate

**Phase 1 — the rule and its unit tests.** `Sources/Analyzers/CultureSensitiveInterpolationAnalyzer.cs`,
`AnalyzerReleases.Unshipped.md` entry, `Tests/Analyzers/CultureSensitiveInterpolationAnalyzerTests.cs`.
No `.editorconfig` change yet. `AnalyzerHarness.Run` is the harness; its reference set
(`Tests/Analyzers/AnalyzerHarness.cs:38-42`) is `System.Runtime.dll` + `System.Console.dll` plus
`typeof(object).Assembly`, which resolves `double`, `TimeSpan`, `DateTime`, `CultureInfo`,
`DefaultInterpolatedStringHandler` and `string.Create` — but **not `System.Text.StringBuilder`'s
`AppendInterpolatedStringHandler` unless `System.Runtime` forwards it**, which is unverified. If the
handler-shape snippets fail to compile, add `System.Text.StringBuilder`'s assembly to `Referenced` by file
name, following that field's own doc comment; do not fake the type with a hand-written stand-in, which
would pin the test's own type name rather than the rule (same reasoning as the `System.Console.dll` note
there).

**Phase 2 — prove it red on real defective code, before anything is fixed.** Build the solution with
`OVERFIT047` temporarily raised to `warning` under `[Sources/**.cs]` (a local edit, reverted; `aot-guard`
is not run in this step). Capture the full diagnostic list. **Acceptance: the run names at least the 17
sites `XC-65`'s row records, including at least one `sb.Append`/`sb.AppendLine` site and at least one site
with no format specifier.** This step is also the only trustworthy census — my §1 count of ~24 candidate
sites in `Sources/Main` is a *text* scan and is therefore wrong in both directions by construction.

For orientation only, the text scan (2026-08-17, `Sources/**`, excluding lines within eight lines of an
`InvariantCulture`/`FormattableString.Invariant`): `Main` 24 — `LanguageModels/Runtime` 7,
`LanguageModels/Retrieval/Evaluation` 6, `Onnx` 3, `Data/Interpretation` 2, `Kernels` 2, and one each in
`Audio/Tts/Orpheus`, `DeepLearning`, `Optimizers`, `Statistical`; `Cli` 21, `Benchmark` 12, `AndroidBench`
9, `Server` 2 (68 in total); **`Anomalies` 0 and `Main/Statistics` 0**. Treat every number here as a hint that phase 2
replaces, not as a result.

**Phase 3 — arm the two clean directories.** The `.editorconfig` block of §7, after the zero measurement
of §7's precondition. This is where `XC-65` ends.

**Phase 4 — NOT this task.** The sweep of `Sources/Main`'s remaining directories, one at a time, each
escalating to `error` at zero, exactly as `OVERFIT043`/`OVERFIT044` were done. It needs its own row
because it edits product strings in nine directories and each edit changes bytes something might assert on
— which is how `XC-64` started. See question A2.

---

## 9. The verification oracle

**The oracle is `AnalyzerHarness.Run` returning an exact diagnostic-id sequence per snippet**, one snippet
per shape, plus the phase-2 red run against real unfixed source. There is no numerical parity or model
involved. Minimum shapes, each an independent test:

| # | snippet | expected |
|---|---|---|
| 1 | `$"{d:F2}"`, `double` | OVERFIT047 |
| 2 | `$"{d}"`, `double`, **no specifier** | OVERFIT047 — the Finding 3 shape |
| 3 | `$"{f}"`, `float`; `$"{m:F2}"`, `decimal` | OVERFIT047 each |
| 4 | `$"{ts:g}"` `TimeSpan`; `$"{dt}"` `DateTime`; `$"{dto}"` `DateTimeOffset` | OVERFIT047 each |
| 5 | `$"{n}"` `double?` | OVERFIT047 — the nullable unwrap |
| 6 | `$"{i}"`, `int`/`long`/`byte`/`nint`/`char`/`bool`/`enum`/`string`/`object` | **none** |
| 7 | `$"{i:N0}"`, `int` | OVERFIT047 |
| 8 | `$"{i:D}"`, `$"{i:X}"`, `int` | **none** |
| 9 | `string.Create(CultureInfo.InvariantCulture, $"{d:F2}")` | **none** |
| 10 | `sb.Append($"{d:F2}")` | OVERFIT047 — the Finding 2 shape |
| 11 | `sb.Append(CultureInfo.InvariantCulture, $"{d:F2}")` | **none** |
| 12 | `FormattableString.Invariant($"{d:F2}")` | **none** |
| 13 | `$"{d.ToString(CultureInfo.InvariantCulture)}"` | **none** |
| 14 | `$"a{d:F1}" + $"b{d:F2}"` assigned to `string` | OVERFIT047 **once** |
| 15 | `string.Create(CultureInfo.InvariantCulture, $"a{d:F1}" + $"b{d:F2}")` | **none** — the `AlertEngine` shape |
| 16 | `throw new ArgumentException($"{d}")` | OVERFIT047 — Finding 5 |
| 17 | `$"{v:F2}"` inside `static string M<T>(T v)` | **none**, and the test says why |
| 18 | `$$"""…{{d:F2}}…"""` raw literal | OVERFIT047 |
| 19 | `$"{d:F2}"` with `#pragma warning disable OVERFIT047` | **none** |
| 20 | `$"{dt:o}"`, `$"{dt:u}"`, `$"{dto:R}"`, `$"{dt:s}"` | **none** — added by the Finding 8 amendment |
| 21 | `$"{ts}"`, `$"{ts:c}"`, `$"{ts:T}"` | **none** — added by the Finding 8 amendment |

Rows 9, 11, 12, 13 and 15 are the ones that decide whether the rule is usable: a false positive on any of
them condemns all 42 sites `XC-64` just fixed. **Rows 20 and 21 join that set** — row 20 is
`SuppressionStore.cs:58` in miniature, the site the signed predicate wrongly reported. Row 4 is unaffected
by the amendment: `ts:g`, bare `dt` and bare `dto` all still fire.

---

## 10. Mutations — a green mutation is a finding

Run after the suite is green, per the repository's standing rule. Each names its predicted victim; a
mutation whose victim stays green is a hole in the tests, not a pass.

| # | mutation | predicted victim |
|---|---|---|
| M1 | drop the `IInterpolatedStringAppendOperation` branch | row 10 only |
| M2 | treat any handler creation as exempt regardless of `IFormatProvider` | row 10 only |
| **M2b** | make the "enclosing call takes an `IFormatProvider`" test return `false` unconditionally | rows 9, 11, 12, 15 — **added 2026-08-17, see below** |
| M3 | drop the `Nullable<T>` unwrap | row 5 only |
| M4 | fire on integral holes regardless of specifier | row 6 only |
| M5 | require a non-empty format specifier before reporting | rows 2 and 3's `float` case |
| M6 | drop the outermost-addition walk | row 14 (two diagnostics instead of one) |
| M7 | re-add `IsOnExceptionPath` | row 16 only |
| **M8** | treat every `DateTime`/`DateTimeOffset` specifier as culture-sensitive | row 20 — the signed predicate, as a mutation |
| **M9** | treat every `TimeSpan` specifier as culture-sensitive | row 21 |

**M1's second clause was unfalsifiable as signed, and `overfit-developer` was right to reject it.** It
read *"…and if rows 9/11/15 also move, the exemption was working by accident"*. Those rows assert
`Assert.Empty`, and M1 only ever **removes** diagnostics — an emptiness assertion cannot be reddened by a
mutation that reports less, so the clause could never have fired whatever the implementation did. **M2b is
the arm that actually tests it**: forcing the `IFormatProvider` check to `false` must redden exactly rows
9, 11, 12 and 15, which is what proves the exemption comes from the provider check rather than from the
operation-kind branch. Corrected here rather than left to be re-derived — and the general shape is worth
keeping: **a mutation that can only subtract diagnostics cannot be pinned by an assertion that expects
none.** Pair every "reports less" mutation with a test that expects a diagnostic.

**M8 and M9 are the signed §3 predicate re-entering as mutations**, which is the right place for it: the
plan was wrong, the tests now say so, and a future edit that "simplifies" the family tables back will
redden rather than quietly restore a false positive on `SuppressionStore.cs:58`.

---

## 11. Risks, and the cheapest experiment that retires each

| risk | retired by | when |
|---|---|---|
| the handler branch cannot be tested because `AnalyzerHarness`'s reference set cannot resolve `AppendInterpolatedStringHandler` | write rows 10/11 **first** and add the assembly by file name if they fail to compile | phase 1, first hour |
| the predicate is wider than believed and one of the two "clean" directories is not clean | §7's precondition measurement, before the `error` lines are written | phase 3 |
| the rule fires somewhere unexpected in `Sources/Main` at a volume that makes phase 4 unaffordable | phase 2 produces the exact count before any commitment is made to phase 4 | phase 2 |
| `Microsoft.CodeAnalysis.CSharp` 5.0.0 does not expose `IInterpolatedStringAppendOperation` on `netstandard2.0` | **already retired** — the probe of §0 ran on exactly 5.0.0 and printed `InterpolatedStringAppendFormatted` | done |

---

## 12. Operability

Nothing runs. The rule surfaces in `dotnet build` output and in the IDE. Two operator-facing notes:

- **`XC-68` applies**: no `.editorconfig` section reaches a source-generated tree, so `OVERFIT047` is inert
  in generated code however the glob is written. That is correct here rather than a gap — generated
  formatting is not this repository's text — and it should be stated in the `.editorconfig` comment so
  nobody later "fixes" it with a `[*.g.cs]` section that cannot work.
- **`RS2\d{3}` must be grepped out of the build log** after the `AnalyzerReleases.Unshipped.md` edit.
  Nothing promotes the release-tracking analyzers to error here, so "the build succeeded" is not evidence
  that the file's format was accepted — `XC-51`'s finding, and it applies verbatim to this row's new entry.
  `AnalyzerReleases.Shipped.md` remains empty; that is the known `RS2001` gap and is not this task's.

---

## 13. Decisions

- **D1** — `OVERFIT047`, category `Reliability`, keyed on the **pair** (type family, format specifier);
  neither alone is sufficient. Rationale: Finding 3 for the type half, **Finding 8 for the specifier half**
  — *as originally signed this decision read "keyed by type-of-hole and not by format-specifier syntax",
  which was wrong for families B and C and produced a false positive.* No ADR: this is neither public API,
  nor an assembly choice, nor an on-disk format, nor AOT reach, nor a dependency, nor a moat boundary. It
  is one file in an existing project, reversible by deleting it.
- **D2** — integral holes fire only with a culture-sensitive specifier; the negative-sign residual is
  accepted and written into the rule's own `description`. Rationale: Finding 1 plus the noise argument
  `.editorconfig:459-461` already makes twice.
- **D3** — `error` in `Sources/Anomalies` and `Sources/Main/Statistics`, `none` everywhere else;
  `OVERFIT047` never enters `WarningsNotAsErrors`. Rationale: Finding 6.
- **D4** — `Cli`, `Server`, `Benchmark`, `AndroidBench`, `Demo` are out of scope permanently, by decision.
  Rationale: Finding 7.
- **D5** — the exception path is not exempt. Rationale: Finding 5.
- **D6** — one diagnostic per interpolated-string expression, not per hole; outermost addition only.
  Rationale: §4, matching `HotPathStringAnalyzer.cs:57-61`.
- **D7 (2026-08-17, amendment)** — the round-trip and interchange specifiers are exempt for dates
  (`o O s u R r`) and the constant format is exempt for `TimeSpan` (none, `c`, `t`, `T`). Rationale:
  Finding 8, measured twice by two agents independently. This decision is pinned by test rows 20/21 and
  mutations M8/M9, so it cannot be silently reverted by a later simplification.

**Explicitly not decided here, and left to the developer**: the file and test class names, the exact
message wording beyond the constraint in §5, whether the symbol cache is a static lookup or a
`CompilationStartAction` closure, and the internal structure of the predicate. Those are local and
reversible.

---

## 14. Out of scope, recorded so it is not re-derived

`PrometheusHistoricalSource.cs:191` and `PrometheusMetricWindowSource.cs:343` interpolate `long`
timestamps into **URLs**. Under §3 those are integral holes with no specifier and `OVERFIT047` does not
fire — correctly, because a positive `long` is invariant on every culture measured. `XC-65`'s row already
flags this as a related class needing its own decision, and the reason it is genuinely different is the
failure mode: a broken query, not ugly text. Not this rule's job; do not widen the predicate to reach it.

---

## BLOCKING QUESTIONS

**For the client** — none. This is an internal build guard with no business rule, no acceptance threshold
and no value judgement outstanding.

**For the analyst / main session** — neither of these blocks phase 1, and I state what I will assume:

- **A1.** Is phase 3 (arming the two directories) inside `XC-65`, or does `XC-65` end at a rule plus tests
  with the ladder as a follow-up? *Assumption if unanswered: phase 3 is in, because a rule at `none`
  everywhere guards nothing and `XC-65`'s row exists specifically to make `XC-64` durable.*
- **A2.** Who owns phase 4 — the sweep of `Sources/Main`'s nine remaining directories? *Assumption if
  unanswered: a new row, filed with phase 2's measured count, not folded into this task. It edits product
  strings in nine directories, and `XC-64` is the evidence that such a sweep is its own piece of work.*

---

## Outcome

Not yet measured — nothing is implemented.

- Success metric: a new culture-sensitive interpolation written in `Sources/Anomalies` or
  `Sources/Main/Statistics` fails the build, and `XC-64`'s 42 fixed sites produce zero diagnostics.
- Measured: —
- Verdict: not yet measurable.
