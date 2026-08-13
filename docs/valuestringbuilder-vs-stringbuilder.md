# `ValueStringBuilder` vs `StringBuilder` — measured, 2026-08-13

**A prediction was written into the benchmark file before the run. It came out one-third right, and that is
why this document exists**: a confirmed prediction teaches nothing, a refuted one moves a rule.

## The run

`[SimpleJob(warmupCount: 8, iterationCount: 20)]` with `MemoryDiagnoser` — deliberately **not** the shared
`BenchmarkConfig`, whose `InvocationCount=1` would leave the 64-character arm measuring timer resolution
rather than work. Ryzen 9 9950X3D (32 logical / 16 physical), .NET SDK 10.0.111, Release, 12.4 min wall.
Full log: `Tests/bin/vsb-bench.log`. Benchmark: `Sources/Benchmark/ValueStringBuilderBenchmark.cs`.

The stack buffer is **256 chars**, and `Grow` doubles, so the lengths below are chosen to sit at
**0, 0, 1, 3 and 6 growth steps** — the crossover axis, not a single ratio.

## Producing a `string`

All four arms end in `ToString()`, so the final string allocation is common to every row and the ratio
isolates the *building*. The pre-sized arm is separate on purpose, so "`ValueStringBuilder` is faster"
cannot be confused with "pre-sizing is faster".

| length | growths | `StringBuilder` default | `StringBuilder` pre-sized | `ValueStringBuilder` (256-char stack) | `ValueStringBuilder` (pooled exact) |
|---|---|---|---|---|---|
| 64 | 0 | 35.85 ns · 496 B · **1.00** | 22.46 ns · 352 B · 0.63 | **15.01 ns · 152 B · 0.42** | 16.92 ns · 152 B · 0.47 |
| 256 | 0 | 104.91 ns · 1408 B · **1.00** | 55.19 ns · 1120 B · 0.53 | **52.77 ns · 536 B · 0.50** | 53.95 ns · 536 B · 0.51 |
| 512 | 1 | 149.26 ns · 2504 B · **1.00** | 124.96 ns · 2144 B · 0.84 | 105.64 ns · 1048 B · 0.71 | **100.82 ns · 1048 B · 0.68** |
| 2048 | 3 | 650.55 ns · 8792 B · **1.00** | 471.47 ns · 8288 B · 0.72 | 408.05 ns · 4120 B · 0.63 | **396.81 ns · 4120 B · 0.61** |
| 16384 | 6 | 4241.72 ns · 82040 B · **1.00** | 2808.14 ns · 65632 B · 0.66 | 2852.71 ns · 32792 B · 0.67 | **2675.73 ns · 32792 B · 0.63** |

## Writing into a caller-owned `Span<char>`

**Its own baseline. Not comparable to the table above** — no string is produced here, so the two tables
answer different questions and the ratios must not be read across them.

| length | `StringBuilder` → `CopyTo` | `ValueStringBuilder` → `TryCopyTo` |
|---|---|---|
| 64 | 26.85 ns · 296 B | **10.17 ns · 0 B · 0.38** |
| 256 | 74.08 ns · 824 B | **33.00 ns · 0 B · 0.45** |
| 512 | 116.68 ns · 1408 B | **66.71 ns · 0 B · 0.57** |
| 2048 | 417.07 ns · 4624 B | **271.74 ns · 0 B · 0.65** |
| 16384 | 3218.13 ns · 49200 B | **2191.17 ns · 0 B · 0.68** |

## The prediction, and what happened to it

Written into the benchmark file **before** the run: *the two types grow by opposite mechanisms —
`StringBuilder` chains a chunk and copies nothing until `ToString`, while `ValueStringBuilder` rents and
copies everything on every growth, ~2n against ~n — so `ValueStringBuilder` wins on bytes at every size,
wins on time only where it does not grow, and the time ratio moves against it as growths rise.*

| claim | verdict |
|---|---|
| fewer bytes at every size | **CONFIRMED** — roughly half throughout, and **exactly zero** on the span path |
| faster **only** where it does not grow | **REFUTED** — faster at every length measured, including six growths |
| ratio degrades as growths rise | **holds for three points** (0.42 → 0.50 → 0.71) then **breaks**, settling at 0.63-0.67 |

**The mechanism behind the prediction is real; the conclusion drawn from it was not.** The extra copying on
growth exists — that is what bends the curve from 0.42 to 0.71 across the first three points — but it is
smaller than what `StringBuilder` pays for chunk bookkeeping and the final consolidating copy, so it never
overtakes.

## What this refutes in our own code

`ValueStringBuilder.cs` said: *"It is not free, and small builds can be slower than `StringBuilder`."*
**Small builds are where its advantage is largest** — at 64 characters it is 2.4x faster than a default
`StringBuilder` and 1.5x faster than a pre-sized one. The comment has been corrected in place, marked as a
correction rather than silently rewritten.

The reasoning behind the old claim was not wrong in its parts — a stack buffer cannot be
register-allocated, and `ToString` does pay a pool return. Both are true and both are smaller than what is
saved. **This is the ordinary shape of a wrong performance claim here: correct components, unmeasured
sum.**

## What it does NOT license

- **It measures one call shape**: repeated `Append` into a single build, on one box, at these lengths. A
  different shape can still lose. Every use stays a hypothesis until measured — see
  [`performance-discipline.md`](performance-discipline.md).
- **It is not a licence to sweep the tree.** An inventory the same day found **71 `StringBuilder`
  constructions in `Sources/**` and zero migration candidates** under the type's stated criteria. What the
  measurement does change is one of those criteria — see below.
- **Two structural blockers that no measurement can move**: a `ref struct` cannot be a class field, so a
  reused-buffer shape like `StopSequenceDetector`'s cannot use it at all; and it cannot be captured by a
  lambda or local function, which rules out several chat-path candidates.
- **Where the result is consumed as characters, a caller-owned `Span<char>` destination beats both** — and
  the span table above prices that: zero bytes, 0.38-0.68 of `StringBuilder`. Swapping such a path onto the
  string-producing form would be a regression dressed as an optimisation.

## The criterion this reopens

The type's guidance had two conjunctive conditions: a `string` must really be produced **and** the build
must take **several growth steps**. **The second is refuted by the table above** — the advantage at zero
growths (0.42, 0.50) is larger than at six (0.63-0.67).

That matters because the inventory rejected most per-call sites for exactly that reason: they pre-size to
the answer and therefore grow 0-1 times. **They are candidates again on these numbers.** What has not
changed is that the win must be shown to matter *in the enclosing operation*: at several of those sites the
`StringBuilder` is not the dominant allocation (`Redactor` allocates a `List`, a `Dictionary` and one
interpolated placeholder per matched span; `JsonLinesAuditSink` allocates two strings per category plus a
`ToString("o")`). **A 20 ns saving inside a 40 ms request is not a finding.**

So the honest next step is not a migration but a second pass with a narrower question: *which per-call,
string-producing sites are on a path where nanoseconds and bytes are actually visible* — chat decode and
the gateway request path — and then measure the operation, not the builder.
