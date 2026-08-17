---
name: mutating-an-armed-analyzer
description: mutating a Roslyn rule that is `error` in a product directory fails the BUILD before any test runs; build with -p:RunAnalyzersDuringBuild=false so the in-process analyzer tests still decide the victim
metadata:
  type: reference
---

**A mutation to an analyzer that is armed at `error` somewhere in `Sources/**` cannot be measured by the
normal cycle**: the mutated rule fires during the build of `Main`/`Anomalies`, `dotnet build` exits non-zero,
and the run reports "did not compile" for every arm — which is indistinguishable from a broken mutation.
Hit on 2026-08-17 building `OVERFIT047`, whose M8/M9 arms would have reddened `Sources/Anomalies` itself.

**The fix is one flag:** `dotnet build ./Tests/Tests.csproj -c Release -p:RunAnalyzersDuringBuild=false`,
then `dotnet test --no-build --filter ...`. The analyzer *unit* tests construct the analyzer directly and
run it over an in-memory compilation, so they are unaffected by the flag while the product build stops
consulting it. Measured: the source generator still runs, so `Anomalies` still compiles.

**Two harness details from the same run.** A mutation that can only *remove* diagnostics cannot redden a
test that asserts `Assert.Empty` — so an arm written as "drop branch X, and if the exemption rows also move
then the exemption was accidental" is unfalsifiable by construction. Prove an exemption is real with the
opposite arm: make the exemption's own predicate return `false` and require the `Empty` rows to go red.
And build the two-arm probe per project — `Anomalies` references `Main`, so a probe in `Main` fails the
build first and the `Anomalies` arm never runs.

Related: [[decode-dispatcher-mutation-arms]], [[eol-check-head-blob]].
