# Contributing to Overfit

Thanks for the interest. A few things to read before opening a PR.

## Quick reference

| Topic | Where |
|---|---|
| Code style | `.editorconfig` + `CLAUDE.md` |
| AOT discipline (what's banned in `Sources/Main`) | `Sources/Main/BannedSymbols.txt` |
| Hot-path conservatism rules | `Sources/Main/README.md` |
| Test layout + `[LongFact]` policy | `Tests/README.md` |
| File header template (run before PR) | `update-code-headers.cmd` |
| Architecture overview | `docs/TECHNICAL.md`, `CLAUDE.md` |

## Code style

- Block-scoped namespaces. Sorted `using` directives. No separate import groups.
- File headers are templated via `.editorconfig` IDE0073 — run `update-code-headers.cmd` before submitting.
- `IDisposableAnalyzers` warnings in `Sources/Main` are effectively errors — this codebase leans on explicit `using` blocks and pooled buffers.

## Native-AOT discipline

`Sources/Main` is published under `PublishAot=true` in CI via the `aot-guard` job. The following are banned and enforced by `Microsoft.CodeAnalysis.BannedApiAnalyzers` with `RS0030` promoted to error:

- `System.Linq` (the namespace is also `<Using Remove="System.Linq" />` in `Main.csproj`).
- `System.Reflection` and `System.Linq.Expressions.Expression`.
- `System.Activator`.
- `Array.Copy` — use `Span<T>.CopyTo` instead.
- Raw `ArrayPool<T>.Shared` — use `PooledBuffer<T>` (scoped via `using`) or `PooledBuffer<T>.RentArray` + `ReturnArray` (class-lifetime).

See `Sources/Main/BannedSymbols.txt` for the authoritative live list. Tests and benchmarks are unconstrained — LINQ is fine there.

Use explicit `for` / `foreach` over `Span<T>`, delegates over reflection, explicit `new` over `Activator`.

## Tests

- All test commands use `-c Release` — never Debug.
- Tests must stay fast. Long-running ones use `[LongFact]` (auto-skipped by default; flip back to `[Fact]` to run locally).
- See `Tests/README.md` for layout conventions and the `[LongFact]` vs `[Fact(Skip=...)]` policy. The two have different semantics — `[LongFact]` is for runtime, `[Fact(Skip=...)]` is for tracked bugs / numerical instability with a preserved "why".
- One public class per test file, named after the subject under test.

## Commits

- Keep commits focused and reviewable.
- Match the one-line subject style of existing history (no enforced format).
- Sign-off (`Signed-off-by:`) is **not** required, but a Contributor License Agreement (CLA) is — see below.

## Contributor License Agreement (CLA)

Overfit is **dual-licensed** under AGPLv3 and a commercial license (see [`LICENSE.md`](LICENSE.md)). To accept external contributions while preserving the ability to offer commercial licenses, all contributors must sign a CLA before their pull request is merged.

The CLA grants the project:

- The right to distribute the contribution under AGPLv3.
- The right to distribute the contribution under the commercial license.
- A confirmation that the contribution is the contributor's own work, or properly attributed.

The CLA is signed **once per contributor**, not per PR. CLA Assistant will prompt you on your first PR with a one-click sign-off. If you decline the CLA, you can still file bug reports, feature requests, and discussions — only merged code contributions require it.

If your employer holds copyright on your work, your employer must sign the CLA, not you personally.

## What we welcome

Yes please:

- **Bug fixes** — with a regression test that fails without the fix.
- **New model architecture loaders** — follow the existing GGUF / safetensors loader patterns; validate against a real checkpoint with bit-parity or coherent-generation evidence.
- **Performance improvements** — BenchmarkDotNet evidence required, both single-thread and (where relevant) multi-thread.
- **Documentation** — scenario doc updates, README clarifications, capability table updates that match shipped reality.
- **Diagnostics / test fixtures** — under `Tests/**/Diagnostics/` with `[LongFact]`.

Please discuss first (open an issue or discussion):

- **Large refactors** — anything touching multiple subsystems.
- **New public API** — must fit the existing facade style; breaking the public surface needs a versioning plan.
- **New NuGet dependencies** — `NuGetAudit` is strict; every dep is weighted on attack surface and trim/AOT compatibility.

Probably not accepted:

- Code introducing LINQ / reflection / `Activator` / `Expression` / `Array.Copy` / raw `ArrayPool<T>.Shared` into `Sources/Main`.
- Mocked dependencies in tests where a real component is feasible (see `Tests/README.md`).
- Half-finished features, scaffolding without implementation, or speculative abstractions.

## Reporting security issues

Do not open public issues for security vulnerabilities. See [`SECURITY.md`](SECURITY.md).

## Questions

Open a [GitHub discussion](https://github.com/DevOnBike/Overfit/discussions) or email **devonbike@gmail.com**.
