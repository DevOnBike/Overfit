# XC-25 — OVERFIT040's console exclusion never fires

**Status:** drafted 2026-08-12 by the main session, pending architect sign-off.
**Scope:** `Sources/Analyzers/SynchronousIslandAnalyzer.cs` and its tests, then the pragmas the fix makes redundant.

## The defect

`OVERFIT040` (`SynchronousIslandAnalyzer`) flags a synchronous method that calls an API with a
compatible asynchronous sibling. It carries an exclusion whose doc-comment reads:

> *Console writers only. `Console.Out` and `Console.Error` are synchronised wrappers whose async
> members write synchronously and hand back a completed task, so switching changes nothing but the
> syntax — measured, and the reason eight of eight CA1849 hits here were noise.*

The implementation is `SynchronousIslandAnalyzer.cs:250`:

```csharp
return called.ContainingType.ToDisplayString() is "System.Console";
```

**For `Console.Out.WriteLine(...)` the containing type is `System.IO.TextWriter`, not
`System.Console`.** So the exclusion never matches the case its own comment names as the reason it
exists. This is not a missed edge case — it is a stated behaviour the code does not have, which is the
same shape as the `.editorconfig` severity block corrected earlier the same day.

## What it costs, measured

**18 of the 35 `OVERFIT040` sites in the service projects were this case**, and every one of them was
answered with a pragma on 2026-08-12 — 13 of them under a single file-scoped pragma in
`Sources/Cli/Commands.cs`, plus a type-scoped one in `Sources/Mcp`. The rule spent a sweep reporting
exactly what it was written to skip, and the suppressions now record a constraint that the analyzer
should have known itself.

## The change

`IsExcluded` currently takes only the called `IMethodSymbol`, which is why the receiver is invisible to
it. The fix needs the invocation's receiver, so the signature has to widen — pass the
`InvocationExpressionSyntax` (or its `MemberAccessExpressionSyntax` receiver) and the `SemanticModel`.

**Exclude when the receiver resolves to `System.Console.Out`, `System.Console.Error` or
`System.Console.In`** — resolve the receiver to an `IPropertySymbol` and require its containing type to
be `System.Console`. Do **not** exclude on the containing type being `System.IO.TextWriter` /
`TextReader` alone: a `StreamWriter` over a file is a real synchronous island, and excluding all
`TextWriter` calls would silence the rule exactly where it should speak.

Keep the existing `Dispose` and `System.Console` cases unchanged.

## Tests — the negative is what matters

Add to the existing analyzer test file, in the shape `OVERFIT039`/`OVERFIT046` use (assert silence as
well as detection):

1. `Console.Out.WriteLine(...)` in a sync method → **not reported**. This is the defect.
2. `Console.Error.WriteLine(...)` → not reported.
3. `Console.In.ReadLine()` → not reported.
4. `Console.WriteLine(...)` (the static, already excluded) → still not reported. A regression guard.
5. **`new StreamWriter(path).WriteLine(...)` in a sync method → STILL REPORTED.** The rule must keep
   firing on a real file writer, or the fix has traded 18 false positives for an unknown number of
   false negatives. This is the test that decides whether the change is correct rather than merely
   quieter.
6. A `TextWriter` parameter passed into a method and written to → still reported, same reason.

**Mutation, required:** make the new receiver check always return `true` (exclude everything reached
through any member access) and confirm tests 5 and 6 go red. A change that only ever removes
diagnostics passes a suite that only tests for silence.

## After the analyzer is fixed

Re-inventory `OVERFIT040` across the solution — raise it to `warning` in a scoped `.editorconfig`,
because a build does not print suggestions — and **delete the pragmas the fix makes redundant**,
including the file-scoped block in `Sources/Cli/Commands.cs` and the type-scoped one in `Sources/Mcp`.
A pragma that suppresses a diagnostic that no longer fires is worse than none: it documents a
constraint that is not there and the next reader believes it.

Expect the tree-wide count to fall from 12 by however many of those 12 are console writers; **the 4
sites in `Sources/Anomalies` have never been assigned to anyone** and are out of scope here.

## Out of scope

- Raising `OVERFIT040`'s severity. It stays at `suggestion`: most of its remaining hits are one-shot
  loaders where synchronous is correct, and the 2026-08-12 pass through `Sources/Main` ended in 32
  pragmas and zero conversions for that reason.
- `XC-26` (`OverfitResourcePool.TryRent` blocking 30 s) — a different defect, and the rule cannot see it.
