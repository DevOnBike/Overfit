# `[LongFact]` release gate — backlog, scored

Compiled 2026-08-07, from the **first execution these tests have ever had**. Until 2026-08-06 the attribute
skipped unconditionally and its own message told the reader to edit source in order to run one, so nothing
in this file is a regression: it is a first measurement of code that had been accumulating unobserved.

**Scoring.** ROI is value per unit of work, not value. Risk is the chance the change makes the suite report
something untrue — which is the failure mode that matters here, because every item below was invisible
precisely *because* something reported green.

**Provenance and its limits.** The numbers come from one chunked run on one box (Windows, 61.6 GB, a local
`docker-desktop` Kubernetes cluster on the same machine). At the time of writing that run had completed
**82 of 175 chunks in 31.9 minutes with 9 failures** and was still going; anything marked *(partial)* will
move. Durations include a cold process start and reading multi-GB weights off disk, so a second run on a
warm file cache is faster and this is not an average of anything.

---

## A. Fixed on 2026-08-07, recorded because of what each cost

| # | What | What it cost before it was found |
|---|---|---|
| A1 | **`longfact.py` could not report failing test names on this box** | The SDK here is Polish-localised: a passing run prints `Powodzenie! — niepowodzenie: 0, powodzenie: 8`, a failing one `Niepowodzenie`. The English patterns matched neither, so the gate would have printed an **empty failing-test list from a run that failed**. Fixed by pinning `DOTNET_CLI_UI_LANGUAGE=en` and taking counts from the TRX the runner writes rather than from console text in any language. The TRX also distinguishes "0 failed" from "nothing ran"; an exit code of 0 does not. |
| A2 | **The gate misreported its own scope by 40%** | `count_longfacts()` counted occurrences of the text `[LongFact`, which includes every mention in a comment or doc block — 103 of them. It reported **359 where there are 256**. The same wrong number had been written into `Tests/LongFact.cs` and into the script's own header ("358 against 1733 ordinary facts, 277 under LanguageModels"; the measured values are **256, 1715, 195**). Fixed to match an attribute followed by a method signature, and the reconciliation now closes exactly: 256 `[LongFact]` + 5 deliberate `[Fact(Skip=...)]` = the 261 the runner reports skipped. |
| A3 | **The whole suite in one process took the machine down** | Peak working set **21.7 GB**, leaving **1.0 GB free of 61.6**. Models loaded by earlier tests are not released while the process lives. Prometheus, kube-state-metrics and node-exporter were all evicted and restarted; the anomaly guard logged a cycle failure on `connection refused`. Killing the run returned the box to **35.3 GB free within seconds**, which is what identifies the cause rather than merely correlating with it. Now chunked — one process per chunk, memory flat at ~37 GB free for the whole run — at a cost of one model reload per chunk. |
| A4 | **Five lab diagnostics failed on `connection refused` and the documented fix did not work** | `k8s/monitoring/forward.cmd` forwarded **only 9090**, while the diagnostics default to **9090, 9098 and 9099**. Four of the five already documented that they need a port-forward; two even printed the exact command. Following the documentation still left two of them red, with a bare `HttpRequestException` that names no cause. The script now forwards all three and carries a map of which port serves which test; each test's doc comment now states its port, the symptom, and how to verify the forward before believing a red result. |
| A5 | **`OVERFITPRERELEASE` turned an accepted package into a content-free warning** | Adding the only prerelease pin to the accept-list left `_OverfitPrereleaseUnaccepted` correctly empty — and the `Warning` task still ran once, batching over the empty list to produce `" is pinned at a PRERELEASE version () and is not on the accepted list."` with no package name. Worse than having no accept-list: accepting a package deliberately left the build shouting about nothing. Fixed with an explicit emptiness condition. Verified in five steps including the negative control. |

---

## B. Open

| # | Task | Why now | ROI | Difficulty | Risk |
|---|---|---|---|---|---|
| T1 | ⏳ **65 of 72 converted 2026-08-07; 7 left, each with a stated reason.** Silent passes now skip. Three mechanisms: `ModelFact` (28) for a `const` path, `FixtureFact` + `TestFixture` (31) for a path resolved at runtime, `ProductionAnomalyBaseFact` (2). See the section below for the seven that remain and why. | **highest** — it is the credibility of the gate itself | the remaining 7 need a judgement each, not a sweep | low; the change can only turn silent passes into visible skips |
| T2 | **Weight initialisation is not seedable, and two tests assert through it** | `MathUtils.Rng` seeds from `Guid.NewGuid().GetHashCode()`, `LSTMCell` and `FastTensorExtensions` draw from `Random.Shared`. Neither `GPT1Model` nor `Crnn` takes a seed, so every run starts from a different network. Two failures below are consequences. | high | medium — touches `Sources/Main`, so it goes through the delivery chain | medium — a seed parameter changes behaviour for every caller that omits it |
| T3 | ✅ **RESOLVED 2026-08-07 by experiment — and it split in two.** Both parity tests pass once the kernel layout is held constant, so they are test bugs (see the section below). The half that did not dissolve is now **T8**. | — | — | — |
| T8 | **The repacked/tiled Q4_K GEMM does not reproduce run to run** | Measured 2026-08-07 and the contrast is the evidence: two invocations of the **per-row** kernel in one process produced identical greedy output; two invocations of the **repacked** kernel produced output diverging from token 0 (`matched 0/24`). Most likely a reduction order that follows the parallel work split. **This is not an exotic path** — `IsPrepacked` makes it the default wherever a `*.gguf.repack` sidecar sits next to the model, which is ordinary usage here. | **high** — it decides what a coherence assertion can mean anywhere in the repo | medium — needs a look at how `GemmTiled` accumulates | low to investigate |
| T9 | **`GgufLlamaLoaderIntegrationTests` compares two different files** | `Max diff 7.83`, **mean diff 1.387** over 151936 logits, and a completely different top-1 (22043 vs 40) — orders past numerical noise, when the repo's own threshold for "enough to flip an argmax" is 0.44. The arithmetic says why: `qwen.gguf` is 6.18 GB (÷2 B = 3.09 B params) and `qwen.bin` is 13.59 GB (÷4 B = 3.40 B), a gap of 0.31 B — which is exactly `151936 × 2048`, one language-model head. The test's comment claims "Both go through identical FP32 kernels — only loader differs"; the file sizes contradict it. Same class as the already-resolved Q4_K_M parity bug, which was **a test-premise bug, not code**. | medium | low — read both headers and diff the tensor lists | low |
| T4 | **`BatchedQuantProjection.UseTiledPrefillQ4K` is left mutated** | `TinyBlasTiledPrefillE2EPhase3Tests` sets the static flag and never restores it, so it stays `true` for every test that follows in the same process. Masked today only because each chunk is its own process. | medium | trivial (`try/finally`) | none |
| T5 | **`AnomalyLabLoadGeneratorTests` needs a second forward nobody documented** | Fails with *"no request succeeded — the lab is reachable but not serving"*. It needs the workload replicas forwarded (`k8s/overfit/forward-replicas.cmd`), which is a different route from the Prometheus one fixed in A4. | medium | trivial | none |
| T6 | ⏳ **Light half measured, heavy half deferred.** 245 test results, **101 minutes** of measured runtime. The distribution is why the gate had to split: median chunk 5 s, but **one test takes 27.4 min** (`QwenGgufKnowledgeInjectionDemoTests`), the next 15.8 and the next 9.6. The 34-test heavy group has not been run and 29 of its entries are still name-guesses. | medium | hours of clock for the heavy half | none |
| T7 | **Split answered by measurement; the skip policy still open.** The heavy/light split exists and is driven by timing (`Scripts/longfact_heavy.txt`). What is still undecided is the environment half: a gate that goes red on a machine without a lab teaches people to ignore it. **Do not decide before T1** — the same mechanism answers both. | medium | low | medium — the wrong split hides real failures |

---

## T1, measured and largely fixed 2026-08-07 — a quarter of the gate could pass without running

**65 of 72 sites converted. 7 remain, listed at the end with the reason for each.**

The pattern, in the shapes it actually took:

```csharp
if (!File.Exists(Path)) { _out.WriteLine("missing gguf"); return; }   // a PASS
if (path is null)       { _out.WriteLine("not found");    return; }   // a PASS
var tok = TryLoad(); if (tok is null) { return; }                     // a PASS
if (!Avx2.IsSupported) { return; }                                    // a PASS
```

Three mechanisms, because one would not have fitted:

| | | |
|--:|---|---|
| **28** | `ModelFact(path)` | the path is a `const string`, so the attribute can take it directly |
| **31** | `FixtureFact(TestFixture.X)` | the path is built at runtime — `Path.Combine(TestModelPaths…)`, or a walk up the tree — which no attribute argument can express, so an **enum** travels instead and resolution lives in one place |
| **2** | `ProductionAnomalyBaseFact` | one fixture with its own search across three directories |

`TestFixture.Avx2AndFma` is in that enum although it is not a fixture at all. It is deliberate: the reader's
question is identical — *was this actually checked?* — and it deserves the same answer in the same place
rather than a second mechanism nobody remembers exists.

### Why the proof is five ordinary `[Fact]` and not a run of the converted tests

Every fixture named by those 65 tests is present on the development box, so all of them genuinely execute
here and always did. **The defect was only ever observable where the models are missing — on CI** — which
is also the one place nobody reads a per-test outcome rather than a colour. `ModelFactTests` therefore
proves the mechanism against a path chosen to be absent, deterministically, inside the fast suite: a
missing fixture must produce a `Skip` naming the path, a present one must not skip, and without
`OVERFIT_RUN_LONG` the long-running skip must still win. It also asserts the wording contains
*"SKIPPED, not passed"* — "fixture not present" on its own reads as an explanation for a pass, and that
reading is what let this survive.

### What it cost to do, which is the part worth remembering

Three attempts to do this with a pattern, three different failures:

1. a condition-matching bug meant ten sites were never touched, and the script reported the shortfall as
   **"converted 9"** — no match is indistinguishable from nothing to do;
2. a non-greedy regex ended a method at the wrong brace and left two files unparseable — 32 syntax errors,
   which is the **loud** kind of failure and the one to prefer;
3. deleting `if (!TryLoad(out var engine, out var tok))` as "a guard" also deleted the only **declaration**
   of both variables. The condition had never done anything — the helper's own documentation says it
   "returns true unconditionally" — so the whole substance of that line was its side effect.

The fourth attempt drove from an explicit table and worked. And while consolidating two byte-identical
copies of `LocateOverfitExe`, the replacement hardcoded `overfit.exe` and dropped the platform check —
which would have made that fixture **permanently absent on Linux**, silently, by skipping instead of
passing. Caught only because the wreckage from (2) was being read line by line.

### The seven left, and why none of them is a sweep away

| test | guard | why it needs a decision |
|---|---|---|
| `BuildFromGguf_RealQwen3B_ProducesOpenableSidecar` | `!File.Exists(gguf)` | `gguf` is built in the body |
| `Load_RealGpt2Safetensors_BitParity_WithBinFixture` | `safe is null \|\| bin is null` | two locals from two separate helpers |
| `Hybrid_VsDense_OnRealDocsCorpus` | `indexed is null` | what `BuildIndex()` needs has not been established |
| `Fusion_KSweep_OnRealDocsCorpus` | `indexed is null` | as above |
| `DraftModel_Speculative_BitIdentical_AndSpeedup_OnNovelText` | `Path.Combine(DraftDir, …)` | not a constant |
| `Qwen05B_Agentic_Probe` | `Path.Combine(Dir, …)` | not a constant |
| `RealMixtralVocab_RoundTrips` | `!File.Exists(path)` | guards on a local |

Each turns on *what this test actually requires*, which is a question for the code. Guessing would attach an
attribute that skips for the wrong reason — a defect harder to notice than the one being removed, because
it looks like a fix.

### The original diagnosis, kept because the reasoning is the reusable part

The pattern, in three forms, all equivalent:

```csharp
if (!File.Exists(Path)) { _out.WriteLine("missing gguf — skipping"); return; }   // green
if (path is null)       { _out.WriteLine("not found — skipping");   return; }   // green
var tok = TryLoad(); if (tok is null) { return; }                               // green
```

**62 of the 256** do this. Three were spot-checked against source by hand after the sweep, because a regex
overcounted twice already today (A2, and a comment-language detector that reported 154 violations where
there were 4). The list includes `Phi4_Loads_And_Generates_Coherent_English`,
`Gemma2_Loads_And_Generates_Coherent_English`, `Bielik_Loads_And_Generates_Polish`, the whole
`*_DecodeThroughput_BestOfN` family, and `Gateway_RedactsOutbound_ForwardsRestores_Audits`.

The repository already has the correct pattern and it predates all of these: `SmallModelFact` and
`Gpt2ModelFact` set `Skip` when the fixture is absent, so the result is a **skip**, which is visible in the
TRX and in any report. A `return` is a **pass**, which is indistinguishable from having checked the thing.

One instance was fixed on 2026-08-07 as a worked example —
`GptAnomalyLoRATargetComparisonProductionTests` now calls `Assert.Skip` and its message names all three
directories it searched, because "not found" without a search path is not actionable. That test is the one
whose entire purpose is to confirm the tiny-base LoRA finding **on the real trained artifact**; it had been
passing in under a second without loading anything.

---

## T3 / T8, settled 2026-08-07 — the A/B was dead, the accidental A/A failed, and only half of it was the test

**The experiment, and the prediction was written down before it ran.** Set
`BatchedQuantProjection.DisableRepackedKernelsForParity = true` in both failing parity tests, so every path
runs the per-row kernel, and see what survives.

| | with the layout held constant |
|---|---|
| `PrefixKvCacheParityTests` | **Passed**, 8 s |
| `TinyBlasTiledPrefillE2EPhase3Tests` | **Passed**, 42 s |

**Half the alarm was wrong and it is worth saying so plainly.** The row above used to read "if greedy decode
is not reproducible, every coherence gate in the repo is measuring luck". Greedy decode *is* reproducible —
on the per-row kernel two runs produced identical output. That sentence was too broad.

**The half that survives is sharper than the original claim, because it is localised.** Both arms of the
tiled test had been running the repacked kernel all along (`IsPrepacked` overrides the flag, and the sidecar
exists), so that test was an unintentional A/A — and it failed. It passes on per-row. The contrast is the
finding:

| path | two invocations, one process |
|---|--:|
| per-row | identical |
| repacked / tiled | diverged from token 0, `matched 0/24` |

If state were leaking between the two arms, per-row would diverge too. It does not. What is left is the
repacked GEMM itself, most likely accumulating in an order that follows the parallel work split. That is
**T8**, and it matters beyond these tests because `IsPrepacked` makes the repacked path the default wherever
a `*.gguf.repack` sidecar sits beside the model.

This is the second time the same trap has been paid for. The documentation of the very field involved records
the first: *"a `*.gguf.repack` sidecar sets `IsPrepacked` and therefore turns the repacked path on regardless
of the env flag — which is exactly how `BatchedPrefillParityTests` came to be failing unnoticed for two days,
being `[LongFact]`."*

### The original diagnosis, kept because the reasoning is the reusable part

`TinyBlasTiledPrefillE2EPhase3Tests.Phase3_Ttft_And_Coherence_RealModel` flips
`BatchedQuantProjection.UseTiledPrefillQ4K` and asserts the greedy continuation is identical across the two
settings. It failed with `matched: 0/24` — divergence from the very first token.

That reads as "the tiled kernel breaks coherence". It is not what happened.

```csharp
// BatchedQuantProjection.cs:407
var tiled = (w.IsPrepacked || UseTiledPrefillQ4K) && !DisableRepackedKernelsForParity
```

A `qwen.q4km.gguf.repack` sidecar sits next to the model (1.33 GB, confirmed present), so `IsPrepacked` is
true and `tiled` is true **in both arms**. The test toggles a flag that decides nothing. Its own timing
agrees: **1203 ms vs 1156 ms, a 1.04x tie**, which is what two runs of the same kernel look like.

So the test unintentionally ran the control this repository's measurement discipline demands — two
identical arms — and **the control failed**. Two identical configurations produced completely different
greedy output.

Two candidate causes, and one run cannot separate them:

1. **Greedy decode is not reproducible.** The repacked GEMM associates its reduction differently under
   parallelism, and the field's own documentation records `maxAbsLogitDiff ≈ 0.44` on Qwen-3B as
   *"enough to flip an argmax"*. One flipped argmax at position 0 explains everything downstream.
2. **State leaks between the arms.** Both use one `engine`; the first arm consumes five sessions before the
   second starts, so the arms are not symmetric.

**The experiment that settles it is trivial and absent: run one configuration twice.** If the two OFF runs
differ, it is (1). If they agree and divergence appears only on the second arm of a shared engine, it is (2).

This is the second time this exact trap has been paid for. The documentation of the very field involved,
twenty lines above the flag, records the first: *"a `*.gguf.repack` sidecar sets `IsPrepacked` and therefore
turns the repacked path on regardless of the env flag — which is exactly how `BatchedPrefillParityTests` came
to be failing unnoticed for two days, being `[LongFact]`."*

---

## The three code failures, separated from the six environment ones

| Test | Failure | Reading |
|---|---|---|
| `GptAnomalyLoRATargetComparisonTests` | `Stage1 LMHead: did not flatten benign (7.9282 >= base 7.3430)` | **Flaky by construction**, and the test says so itself: its own comment records *"LM-head-only flattened to 0.05 one run and only 6.65 the next"* because the base is unseeded random init — then line 91 asserts the invariant anyway, for every stage. A comment two lines above even claims single-stage magnitude is *"not asserted"*. Blocked on **T2**. |
| `CtcOcrLettersDemoTests` | `LM-beam recognition 5/24 too low` (threshold 22/24) | **Training did not converge**, not a decoder fault: the sibling assertion `lmOk >= greedyOk` passed, so greedy was equally bad. The test *looks* seeded — `new Random(20260527)` — but that seeds the data order; the CRNN's LSTM initialises from `Random.Shared`. Partially addressed 2026-08-07: non-finite CTC losses are now counted and asserted (they used to be swallowed by a bare `continue`, so a batch that produced no gradient still called `optimizer.Step()` silently), and a loss-decrease assertion now runs **before** the recognition assertions, so a collapsed run reports the cause instead of the symptom. The 0.5x factor there is a collapse detector, **not a measured threshold** — tighten it once T2 lands. |
| `TinyBlasTiledPrefillE2EPhase3Tests` | `Assert.Equal() Failure: Collections differ` at position 0 | Passes once the layout is held constant, so the test is at fault — but **do not fix it with that flag**: it disables tiling in *both* arms, leaving the test comparing nothing. Its real defect is that tiling cannot be turned off while a sidecar exists. Until then its reported **1.04x is a measurement of two identical arms**, i.e. noise, and must not be quoted. The residue is **T8**. |
| `PrefixKvCacheParityTests` | `Assert.Equal() Failure: Expected 34, Actual 322` | **Test bug, settled by experiment.** It compares prefilling 32 tokens at once against 24+8 separately and asserts bit-identity, without holding the kernel layout constant — which this codebase documents as producing `maxAbsLogitDiff ≈ 0.44`, *"enough to flip an argmax"*. `BatchedPrefillParityTests`, which was already caught by this, sets both parity flags; this one sets neither. Passes with `DisableRepackedKernelsForParity`. |
| `GgufLlamaLoaderIntegrationTests` | `Max diff = 7.83 at vocab[28974]` | See **T9** — the two files differ by one language-model head. |

---

## Order

**T1 first**, because until the gate stops reporting success for work it did not do, no other number from it
means anything — including the runtime in T6 and the pass rate anyone would quote. **T3's diagnosis next**,
because it is one run of an existing test and its answer decides whether every coherence assertion in the
repository is sound. Then **T2**, which unblocks two failures and needs the delivery chain because it changes
`Sources/Main`. **T4 and T5** are minutes each and can go with anything. **T7 last**, because T1 supplies the
mechanism it needs.
