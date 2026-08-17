---
name: audit-pb12-decode-pool
description: PB-12 verdict — the four published decode-pool figures re-measured at HEAD e21e7c3 on 2026-08-14; two hold, two do not.
metadata:
  type: project
---

# PB-12 — decode-pool published numbers, audited 2026-08-14

**Measured on:** Ryzen 9 9950X3D (32 logical / 16 physical), Windows 11 26200, .NET 10.0.8, Release,
HEAD = `e21e7c3` (XC-50 committed 2026-08-14 20:05 — the task brief called it uncommitted; it is not).
Harness: `.claude/pb12` (see [[harness-pb12]]). ABAB at process level, best-of-3 within each process,
canary in every process, dispatch-counter liveness probe in every process.

| # | Published claim | Where it is written | Measured at HEAD | Verdict |
|---|---|---|---|---|
| 1 | `455 µs / 0 B` vs `Parallel.For` `2059 µs / 925 KB` = **4.5x** | `OverfitParallel.cs:782-783`, `measured-baselines.md:375`, `performance-discipline.md:145`, `code-patterns.md:98`, `Runtime/README.md:16` | **3.3-3.6x** vs uncapped TPL, **2.3-2.7x** vs the capped path that is today's actual fallback; 0 B confirmed; 862 KB-1.13 MB confirmed | NOT SUPPORTED as stated (ratio + no recorded workload) |
| 2 | Qwen3-0.6B **+28%** (56.7→72.3) | `OverfitParallel.cs:359` | **+25.1%** Q8_0 (70.75/56.56), **+23.0%** Q4_K_M; paired 1.251/1.274/1.226 | SUPPORTED (re-stamp number + quant) |
| 3 | Phi-3.5-3.8B **+3%** (13.27→13.69) | `OverfitParallel.cs:359` | **−1.9%** (13.12/13.38); paired 0.986/0.968/0.965/1.005 | NOT SUPPORTED (sign reverses, magnitude under floor) |
| 4 | Bielik-4.5B decode-worker cap **+11%** (12.55→14.0) | `OverfitParallel.cs:178` | **+3.8%** in its original mechanism (pool OFF, cap 10 vs 32); **+87%** at HEAD default because the cap now also sizes the spin pool | NOT SUPPORTED as stated (confounded lever) |

**Key facts that make the above evidence, not opinion:**

- **The lever is live and I proved it.** `OverfitParallel.CountDispatches` is incremented only by the capped
  parking path, so with the pool ON it reads 0/token and with it OFF it reads the real dispatch census:
  Qwen3-0.6B Q8_0 **183.1/token**, Q4_K_M 137.6, Bielik-4.5B **383.6**, Phi-3.5 **209.1**. This is the
  cheapest liveness probe on this path — use it every time.
- **On Phi-3.5 the pool only moves 75% of dispatches**: 52.0 dispatches/token still go through
  `OverfitParallel.For` with the pool ON (GQA `For(0, KvHeadCount, …)` at `CachedMultiHeadAttention.cs:284`).
  On Qwen3 and Bielik the pool takes all of them.
- **The untouched arm reproduces its published value; the treated arm does not.** Qwen OFF 56.56 vs published
  56.7 (+0.2%); Phi OFF 13.38 vs published 13.27 (+0.8%). Qwen ON 70.75 vs 72.3 (−2.1%); Phi ON 13.12 vs
  13.69 (−4.2%). Suggestive, not decisive — the original runs' build and date are not recorded anywhere.
- **`0 B/token` and `bit-identical` both hold at HEAD.** Every run reported 0 B/tok; FNV hash of generated
  token ids is identical pool ON vs OFF on both Qwen3 quants.
- **The sync-protocol change's own cost is not measurable here** (DERIVED, not executed): one uncontended CAS
  plus one uncontended monitor enter/pulse per dispatch is tens of ns against a ~3.4 µs dispatch and
  ~14 ms/token — 0.02-0.05% of token time, three orders below the ±3-4% floor.

**What none of the five citation sites records:** model, quantisation, dispatch count, comparator
configuration (capped vs uncapped `Parallel.For`), date or build — for a file whose own rule is "every number
here carries what it was measured on".

## Did the 2026-08-14 claim/park protocol change cost anything? Measured, not derived — 2026-08-14

The client asked whether the numbers moved because the fix broke something. **It did not.** EXECUTED, 36
processes, AB/BA alternated at process level so build position is not confounded with build.

**How the pre-change build was made, and why not a plain `f8dd016` checkout:** `git diff f8dd016 e21e7c3 --
Sources/Main` lists **17 files**, not 2 — the path-filtered log hid intermediate commits, and one of the
extras is `SingleTokenProjectionKernel.cs`, a decode hot-path file. A whole-tree old build would confound the
protocol change with those. The isolated lever is **HEAD's tree with `Runtime/OverfitParallel.cs` alone
reverted to `f8dd016`** (`DecodeChunkClaim.cs` deleted; nothing else references it). Each binary prints its
own MVID and auto-detects its protocol, and the summary asserts the arm label matches what the binary
reports — a mislabelled arm cannot pass silently.

| model | dispatches/token | old/new, pool ON (treated) | old/new, pool OFF (canary, untouched) |
|---|---|---|---|
| Qwen3-0.6B Q8_0 | 183.1 | 1.007 (+0.7%) | 0.999 (−0.1%) |
| Phi-3.5 Q4_K_M | 209.1 | 1.004 (+0.4%) | 0.995 (−0.5%) |
| Bielik-4.5B Q4_K_M | 383.6 | **0.995 (−0.5%)** | 1.008 (+0.8%) |

**The treated arm moves no more than the untouched one**, and the per-cycle sign flips in every model. The
per-dispatch-cost hypothesis predicts the damage should scale with the census — it predicts Bielik worst.
Bielik is the model where the **new** protocol is fastest. At dispatch level (25 reps, both censuses) the
pool arm is 512/536 µs new vs 515/540 old at d=183, and 1083/1082 new vs 1049/1045 old at d=384 — a sign
flip with shape, bounding any per-dispatch cost below ~90 ns against a 2.7 µs dispatch.

**The published ratios do not reproduce on the OLD protocol either:** Qwen +23.3% (published +28%),
Phi −0.8% (published +3%), Bielik pool +8.8%. So the documentation correction stands and there is no
regression finding.

**Phrasing correction the client made and I accept:** the figures were not "unmeasurable". The `[ModelFact]`
diagnostics exist and somebody ran them by hand. The defensible statement is that **nothing in
`Sources/Benchmark` covers this path and nothing runs in CI**, so the method was single-arm, one process, no
canary — a weak method, not an absent one.

**Unresolvable from history:** the pool and the cap both first appear in the same squashed commit `8a28bdb`
(2026-06-11), so I cannot tell whether the `+11%` was measured with the pool present. Measured today on
Bielik: pool worth +8.8…+10.2%, cap alone (pool off) +3.8%, cap at HEAD default (pool on) +87%.
