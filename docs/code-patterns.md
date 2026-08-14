# How code is written here

For anyone — human or agent — about to write C# in this repository. `CLAUDE.md` carries the boundaries;
this file carries the idiom.

**Read this first, because it is not how a web or a line-of-business application is written.** Almost every
rule below exists because this library runs **inside somebody else's process**, on their CPU, in their
memory, often compiled ahead of time with no JIT to rescue it. A framework app can allocate freely, use
reflection, throw on a bad file and let the host restart. None of those are available here: an uncaught
stack overflow is the customer's application dying, a hidden allocation in decode is their latency, and a
reflection call is a Native-AOT publish failure.

Three claims that shape everything: **zero allocation on the inference path**, **Native-AOT compatible**,
**no Python, no ONNX Runtime, no native binaries**. Every one is a promise somebody is entitled to check.

---

## 1. Hard rules — a build gate fails, not a reviewer

These are not preferences and there is nothing to discuss at review time. They fail `dotnet build`.

| Never | Instead | Enforced by |
|---|---|---|
| `System.Linq` in `Sources/Main` | explicit `for` / `foreach` over `Span<T>` | `RS0030` error + `<Using Remove>` |
| `System.Reflection`, `System.Activator`, `Expression` | delegates; explicit `new` | `RS0030` error |
| `Array.Copy` | `Span<T>.CopyTo` | `BannedSymbols.txt` |
| raw `ArrayPool<T>.Shared` | `PooledBuffer<T>` scoped (`using`), `.RentArray`/`.ReturnArray` for class lifetime | `BannedSymbols.txt` |
| `Stopwatch.StartNew` / `new Stopwatch()` | `ValueStopwatch.StartNew` → `GetElapsedTime` (alloc-free) | `BannedSymbols.txt` |
| jagged `float[][]` in `Sources/Main` | flat `float[]` sliced per row, or `PooledBuffer<float>` / `TensorStorage<float>` | MSBuild task `OVERFIT-JAGGED` |
| two top-level types in one file | one file per type, named after it. Nested types and same-name `partial` are fine | MSBuild task `OVERFIT-ONETYPE` |
| `else` / `else if` | guard clause, early `return`/`continue`, ternary | `OVERFIT021` |
| recursion without a stated bound | an explicit stack/worklist, or `#pragma warning disable OVERFIT022` whose comment starts `BOUND:` and names the bound | `OVERFIT022` |
| `while (true)` / a loop with no exit in its header | a counter or condition in the header, or the same `BOUND:` pragma | `OVERFIT023` |
| `stackalloc` over the byte budget, or with a variable length | `PooledBuffer<T>`, or the `BOUND:` pragma naming what limits it | `OVERFIT025` / `OVERFIT026` |
| `async void` | `async Task` | `OVERFIT027` |
| unchecked `int` arithmetic on sizes | `checked`, or widen to `long` before multiplying | `OVERFIT028` |

**These are `error` in `Sources/Main` and, since 2026-08-10, in `Sources/Anomalies`, `Server.AspNet`, `Mcp`
and `Cli` too** — the assemblies that parse Prometheus text, HTTP bodies, JSON-RPC frames and files. The
backlog was measured at zero before arming, so nothing was grandfathered in.

**The pragma is the escape hatch and it has a contract.** `BOUND: cols is validated to [1,16] by the throw
above, so 2560 B` is acceptable. If you cannot write that sentence, the code is not safe and the pragma
would be a lie. **Never widen a directory's budget to clear a backlog** — that waves through every future
file nobody has looked at.

---

## 2. Parsing anything that came from outside the process

This is the largest and most dangerous surface in the product: GGUF, ONNX (hand-rolled protobuf),
safetensors, `.bin`, `tokenizer.json`, `.repack` sidecars, WAV, MP3, Prometheus exposition text, HTTP
bodies. A model downloaded from a public hub is attacker-influenceable input.

- **Validate a size against what remains in the file BEFORE allocating**, never after. A header claiming a
  40 GB tensor must be refused by arithmetic, not by `OutOfMemoryException`.
- **A loop whose trip count comes from the file needs a stated bound.** That is what `OVERFIT023` is for.
- **Integer overflow in offset/size arithmetic** is how a bounds check passes and an out-of-range read
  happens on the next line.
- **Joining a path from file content goes through `ContainedPath.Resolve`.** Three checks, each closing a
  different door: empty, rooted (this is the one people miss — `Path.Combine` looks like it constrains the
  result and does not), and `..` traversal tested **after** `Path.GetFullPath` normalises. On Windows a UNC
  name is rooted, so this is also what stops a model pointing the loader at a remote share.
- **Throw `OverfitFormatException`** for malformed input, not `ArgumentException` — a caller that handles
  bad models must be able to catch one type.

---

## 3. Allocation and the two execution paths

**Mixing these two is the most common architectural mistake here.**

- **Inference** — `InferenceEngine.Run(input, output)` with caller-owned buffers, zero allocation per call.
  No `AutogradNode`, no `ComputationGraph`. Do not call `model.Forward(...)` on the inference hot path.
- **Training** — `ComputationGraph` records a tape, `graph.Backward(loss)` walks it, `graph.Reset()` reclaims
  by ownership.

Every `AutogradNode` carries an ownership tag that decides who disposes it: `GraphTemporary` and
`GraphAuxiliary` → `graph.Reset()`; `Parameter` → the owning layer; `ExternalBorrowed` → the caller; `View`
→ nobody. Getting this wrong leaks or double-frees, and neither shows up in a unit test.

**On the hot path**: no `.ToArray()`, no hidden allocation, no LINQ, no closures capturing locals. Prefer
`readonly struct` for new value types and change it only when the build forces you.

**Minimise PEAK memory during load, not just steady state** — the target includes low-end hardware. Prefer
unpooled weight buffers and avoid scratch `byte[]` in read paths.

---

## 4. Things that are measured here, not assumed

These read like micro-optimisation folklore elsewhere. Here they have numbers behind them, and several
contradict the obvious answer.

- **`for` vs `foreach` over an array is NOT a lever** (~2 ns, and the direction reverses with size). The
  lever is the **declared type**: an interface costs 2.4× (`foreach`) to 4.6× (indexing) plus 32 B for the
  enumerator. Do not "tidy" a `T[]` into `IReadOnlyList<T>`.
- **`OverfitParallelFor` in decode** — 455 µs / 0 B against `Parallel.For` at 2059 µs / 925 KB. **Everywhere
  else use `Parallel.For`**: migrating `Conv2D` to it measured +13% wall and was reverted.
- **`TensorPrimitives` beats a hand-written micro-kernel**, and the simple register-blocked GEMM beat the
  cache-blocked one. The structure of the data around a technique decides, not the technique — so **never
  extrapolate a win from one kernel to another** without measuring it there.
- **Extracting a method costs 2.25× when the JIT does not inline it**; ternary, `continue` and inverting a
  condition are free. That is why the `else` ban is free and why "just extract a helper" is not.
- `stackalloc` does not put anything in registers.

Full list of reverted "wins", including two that were parity-correct: [`performance-discipline.md`](performance-discipline.md).

---

## 5. Style, and why it is not arbitrary

- Block-scoped namespaces. Braces always. **Bodies on their own line** — no `if (x) { return; }` one-liners.
- File header (AGPL/commercial notice) on every file; `update-code-headers.cmd` applies it.
- `dotnet_sort_system_directives_first`, no separate import groups.
- One public class per file, named after the subject under test or under implementation.
- Exceptions: `OverfitException` is the base; malformed data → `OverfitFormatException`; invalid operation
  and unsupported → `OverfitRuntimeException`. `ArgumentException` and `ObjectDisposedException` stay BCL.
- `IDisposableAnalyzers` is wired in — heed it; the project leans on `using` plus pooled `TensorStorage<T>`.

**Comments carry the WHY and, where one exists, the incident.** This is the repository's most distinctive
habit and it is deliberate: a doc comment saying *what* a method does duplicates the signature, while
*"three, chosen to sit between the cases that have actually been measured — nothing measured falls between
1 and 18, so the exact value inside that range is not load-bearing"* tells the next reader whether they may
change it. **A number without the measurement that produced it is a number nobody can ever safely touch.**

---

## 6. Tests

- **`-c Release` always.** Debug numbers and Debug behaviour are not this project's contract.
- `dotnet test` must stay fast and contain **only correctness checks**. Anything long-running is
  `[LongFact]` (auto-skipped). Keep `[Fact(Skip = "...")]` — with the specific reason — for bug-tracker
  skips and flaky timing tests, because the *why* is the point of the distinction.
- Layout is domain first, purpose second. Fixtures under `Tests/test_fixtures/`.
- **A test that cannot fail is worse than no test.** Both of these shipped here: an assertion satisfied by
  built-in channels before the new one existed, and a fixture whose value coincided with the fallback so
  both sides read 24. **Prove a new test can fail by breaking the code under it** and watching it go red.
- **Bound every wait.** A hung test has no message and blocks everything behind it; two regression tests
  here hung instead of failing under exactly the regression they existed to catch.
- **New code carries at least 80% line coverage, measured on the lines you wrote**, with
  `--settings coverlet.runsettings` — never a bare `--collect`. Those exclusions are load-bearing:
  instrumenting the hot loops costs 10x–900x, so `Ops`, `Kernels`, `Maths`, `Intrinsics`, `Autograd`,
  `Optimizers`, `Tensors` and `LanguageModels.Runtime` are excluded and code added there reports as
  uncovered however well tested it is — cover those with a named test per behaviour and say so.
  **80% is a floor on effort, not a target**: a covered line proves execution, not that anything asserted
  on it. Coverage plus a mutation that turns the test red is evidence; coverage alone is a percentage.

---

## 7. Before you claim anything about performance

Correctness first, in a **separate** pass, pinned by a parity test. Only then a second pass for speed,
A/B-ed against that baseline. Write the benchmark before the argument. Best-of-N on **both** sides, one
lever isolated, ABAB interleaving in one process, and an untouched canary path that tells you whether the
box moved instead of your code.

**If a result is backwards, suspect the benchmark before the runtime. If it is a flat 1.00, suspect that the
lever is not live** — that has happened here. `overfit-perf-claim-auditor` owns the verdict on any
performance claim and must not be substituted for.

---

## 8. What NOT to build

- **No exporters.** Loading is one-directional: external formats → Overfit, all native. Do not propose or
  build Overfit → anything converters.
- **No "real-time" promises in public docs.** Offline and batch correctness is the open story; real-time and
  GPU are the commercial differentiator.
- **Nothing that reintroduces a Python or ONNX Runtime dependency at runtime.** Conversion scripts under
  `Scripts/` are build-time tooling and are not part of the product.
