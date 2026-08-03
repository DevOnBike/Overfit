# Overfit — completed roadmap

Work that is finished, closed or superseded, moved out of [ROADMAP.md](ROADMAP.md) on 2026-08-03 so
that file shows only what is still open.

**Kept rather than deleted, and the reason is this project's own rule.** Most of what follows is a
measurement — including the negative ones: register blocking, K-blocking, Winograd F(2,3), the AVX-512
decode port, bias in the tiled prefill GEMM and `OverfitPool<T>` all regressed or tied and were
reverted. A deleted negative result is an experiment somebody repeats.

Sections appear in the order they held in the original file.

## ✅ FIXED — decode runtime (found 2026-08-01 by `overfit-find-bugs-game`, fixed 2026-08-02)

**All three fixed 2026-08-02.** (1) `Cls` pooling now takes the FIRST token in `CachedLlamaSession.Embed`;
the branch was `pooling != Mean && i == last`, which handed `Cls` the last-token vector. (2)
`GemmTiled512` gained the `groupStart`/`groupCount` parameters `GemmTiled` already had and
`TiledBandChunk` now branches on `c.Avx512` like its three siblings — **verified**: the banded parity
theory in `Avx512PrefillParityTests` is bit-identical across every band and against the unbanded result,
and a four-arm benchmark (untouched `WeightStationary` and `Tiled` as canaries, both flat at ratio
1.00–1.01 and unchanged respectively) puts the banded 512 path at **1.15–1.17× on `ffn_gate_up`, 1.05–1.06×
on `attn_qo`, and a tie on `ffn_down`/`llama_ref`** against the same call with AVX-512 forced off. Note the
banded path is `UseOutputBlocking`, which is **off by default**, so this speeds an opt-in path rather than
the shipped one — and banding remains slower than plain `Tiled` on this box. (3) `Q4KWeight.EnsureRepacked`
publishes through a `byte[]` reference under a lock instead of writing a `ReadOnlyMemory<byte>?`, which
cannot be published atomically; a weight set is shared across concurrently created sessions by design.
E2E coherence re-checked on the real Qwen2.5-0.5B Q4_K_M after the kernel change.

Two bug hunts over `Sources/Main/LanguageModels/Runtime`. The first read a third of the directory in five
minutes and returned **nothing**; its value was the list of files it had *not* opened, which pointed the
second hunt straight at the dense kernels instead of spending its budget on reconnaissance. Reports in
`docs/bug-hunts/`.

Two of the three are confirmed by reading; the third is reported unconfirmed on purpose, because settling it
needs a concurrency stress test and a measurement was running on the box.

| # | Defect | Why it matters |
|---|---|---|
| **1** | **`EmbeddingPooling.Cls` silently returns the wrong vector.** `CachedLlamaSession.Embed` branches only on `Mean` versus everything-else, so `Cls` takes the last-token path instead of the first. `BertEncoder` implements the same contract correctly, so this is one path breaking a live contract rather than an unimplemented option. No test covers `Cls` in either direction. | **Cheapest to fix, ugliest symptom.** The result has the right dimension and is correctly normalised; nothing throws. Similarities are simply worse and nobody can say why. It is in the public API. |
| **2** | **AVX-512 is silently inert in the banded Q4_K prefill.** `BatchedQuantProjection.TiledBandChunk` has no `if (c.Avx512)` branch and always calls the 256-bit `GemmTiled`. Its three siblings — `TiledChunk`, `TiledQ6KChunk`, `TiledQ6KBandChunk` — all have it, so the omission is asymmetric rather than a design choice. | A feature that is configured, parity-tested (`Avx512PrefillParityTests`) and documented as active does not run on one path — precisely the short-prompt case banding exists for. The test gap is the lesson: the two kernels are pinned against each other, and **nothing checks that the dispatcher selects the 512-bit one**. |
| **3** | **`Q4KWeight.EnsureRepacked()` builds its lazy cache without synchronisation** while the architecture deliberately shares one weight set across concurrently created sessions — cheap session creation is the whole point of the design. Concurrent first decode is therefore a reachable race on a `ReadOnlyMemory<byte>?`. | **Unconfirmed by design.** Reading establishes the missing lock and the reachability; observing a torn read needs a stress test. Reported anyway, because an unverified finding costs a follow-up and a verified one would have cost a day of measurement. |

Fix order: 1, then 2, then 3 — cheapest and worst-symptom first, then the silent flag, then the race that
needs a test harness before it can be confirmed or dismissed.

**Not reached by either hunt**, so no claim is made about them: `BatchedProjectionKernel.cs`,
`Q8DotKernel.cs`, `Q6KRepack.cs`, `Gpt1SlmModelAdapter.cs`, `CachedGpt1ModelAdapter.cs`,
`SingleTokenLayerNormKernel.cs`, `SingleTokenProjectionKernel.cs`.

## ✅ FIXED — evolutionary (found 2026-08-01 by `overfit-find-bugs-game`, fixed 2026-08-02)

**All four fixed 2026-08-02**, with one correction to the hunt's own prescription worth recording.

The report said `GenerationalGeneticAlgorithm.Tell()` should carry its siblings' `if (!_hasPendingPopulation)
throw`. Applying that broke **seven existing tests**, and the tests were right: for `OpenAiEsStrategy` and
`SeparableCmaEsStrategy`, `Ask` *draws* the population being scored, so fitness without a draw is
meaningless; here the population is durable state that `Initialize` creates and `Ask` merely copies out, so
`Initialize` → `Tell` is a legitimate loop. The real hole was the missing **initialization** guard, which is
what shipped — the reachable garbage (a `PooledBuffer<float>` taken with `clearMemory: false`) was the
finding, and the prescribed fix was the wrong shape for it. The other three went in as written: NaN
descriptors are rejected by `!(value >= min && value <= max)`, `PartialSort.SortIndices` takes a
`Span<int>` so a shrinking population no longer throws, and `IEvolutionCheckpoint`'s documentation now says
what all three implementations actually do.

All 29 files read, and the hunt **ended by scope rather than by the clock** — which is what makes the score
meaningful. Four defects against a target of eleven says the module is in better shape than the game
assumed; the same number on the decode runtime would have said only that two thirds of it went unopened.
Report in `docs/bug-hunts/evolutionary-2026-08-01-2226-bugs-game-findings.md`.

| # | Defect | Why it matters |
|---|---|---|
| **1** | **`GenerationalGeneticAlgorithm.Tell()` has no Ask/Tell guard.** Its two siblings, `OpenAiEsStrategy` and `SeparableCmaEsStrategy`, carry an identical `if (!_hasPendingPopulation) throw` — this one goes straight to `fitness.CopyTo`. Verified by reading all three side by side. | Calling `Tell()` cold reads `_workspace.Population`, a `PooledBuffer<float>` taken with `clearMemory: false` — **leftover pool contents, not zeros** — and stores them as `_bestParameters`. Nothing throws, the result has the right shape, and the search starts from somebody else's freed buffer. The asymmetry rules out a deliberate choice: it is one of three parallel implementations missing a check the other two have. |
| **2** | **`GridEliteArchive` silently files a NaN descriptor into cell 0.** The bounds test `value < min \|\| value > max` is **false for NaN in both directions**, so a NaN falls through and the conversion lands it in the first cell. | The contrast is the tell: the *fitness* argument has an explicit `EliteInsertStatus.InvalidFitness` path for exactly this case, and the descriptor has none. A MAP-Elites grid whose first cell quietly collects every degenerate candidate is not a map of the behaviour space any more. |
| **3** | **`CenteredRankFitnessShaper.Shape()` throws when the population shrinks.** Its own documentation promises a ranking buffer that "grows monotonically… for zero-alloc reuse", and then passes the full, larger array unsliced to `PartialSort.SortIndices`, which requires equal lengths. | A false claim and a crash from one cause. The doc is what a caller reads before deciding a shrinking population is safe. |
| **4** | **`IEvolutionCheckpoint`'s documentation contradicts every implementation.** It states that checkpoints do **not** capture RNG state and that resumes are not bit-identical; all three strategies persist RNG state, and one carries an inline comment saying a resumed run *is* bit-identical rather than merely statistically equivalent. | A one-line fix, and the one most likely to cost somebody real time: reproducibility of a resumed search is exactly the property a person checks the interface documentation for, and here it tells them the opposite of the truth. |

Fix order: 1, 2, 3, 4 — the silent-garbage path first, then the silent-NaN path, then the crash-plus-false-claim,
then the doc. Nothing here is large; 4 is a single paragraph.

**Checked and clean**, so nobody repeats the work: the `PrecomputedNoiseTable` index round-trip between
`Ask` and `Tell` — the trick the whole OpenAI-ES memory profile rests on — and the parallel population
evaluator's order-independence, which is what a reproducible-from-seed claim needs in order to hold.

## ✅ FIXED — deep learning layers (found 2026-08-01 by `overfit-find-bugs-game`, fixed 2026-08-02)

**All three fixed 2026-08-02.** `LstmCell.Save`/`Load` and `LstmLayer.Save(string)`/`Load(string)` were
empty bodies and now write and read the three parameter tensors in the same length-then-floats wire format
`Parameter.Save` uses. `DepthwiseConv2DLayer.Load` reads the bias flag as a statement about the FILE and
constructs `Bias` on demand — the old `flag == 1 && Bias is not null` consumed the flag and skipped the
floats, misaligning every later read in a composite `Load`. `CheckpointedModule` refuses a segment
containing dropout unless `allowNonDeterministic: true`, since a checkpointed segment is run twice and the
recomputation would use a different mask than the forward pass did.

Breadth-first over the high-yield areas; the hunt **stopped voluntarily with time left but without full
coverage**, so the score says nothing about the parts it did not open. Report in
`docs/bug-hunts/deeplearning-2026-08-02-2237-bugs-game-findings.md`.

| # | Defect | Why it matters |
|---|---|---|
| **1** | **LSTM weights never persist.** `LSTMCell.Save` and `LSTMCell.Load` are literally empty method bodies — not incomplete, `{ }`. `LSTMLayer` delegates through unchanged, and `LSTMAutoencoder` and `Crnn` inherit it. Verified by reading. | **This breaks a shipped capability in the way that is hardest to notice.** Save a trained CRNN, load it, and its convolutions, norms and classifier all come back correctly while the recurrent core sits at random initialisation. Nothing throws, the model looks loaded, and it produces nonsense. The OCR demo that reads digits at loss 0.006 cannot survive a round-trip through disk. |
| **2** | **`DepthwiseConv2DLayer.Load` desynchronises the stream.** `if (br.ReadInt32() == 1 && Bias is not null)` consumes the flag but skips the trailing floats when the file has a bias section and the layer was built with `useBias: false`, misaligning every subsequent read in a composite `Load`. `ConvLayer.Load` handles the same case correctly by constructing `Bias` lazily — so, as in the evolutionary strategies, the asymmetry between two implementations of one pattern is the evidence. | A misaligned stream does not fail where it went wrong. It fails later, in a different layer, as dimensions that do not match — or worse, does not fail at all and loads plausible garbage. |
| **3** | **`CheckpointedModule` does not enforce the determinism it requires.** `ComputationGraph.Checkpoint` documents that the segment must be deterministic and nothing checks it; `CheckpointedModule` accepts any `IModule`, and `TensorMath.Dropout` draws from an unseeded `Random.Shared`. Recomputation during backward then uses a different mask than the forward pass did. | **Reported unconfirmed in the strict sense**: `GPT1Model`'s own use is safe because its transformer block has no dropout, so no shipped path is currently wrong. The composition is unguarded, and the failure mode is wrong gradients with no error — a model that trains, converges to something, and is quietly optimising a different objective. |

Fix order: 1, 2, 3. The first is a shipped model that cannot be reloaded; the second corrupts loading
silently; the third is a trap that nothing currently steps in.

## ✅ FIXED — data preparation (found 2026-08-01 by `overfit-find-bugs-game`, fixed 2026-08-02)

**All four fixed 2026-08-02.** `FastRandomForest` now carries rows as an index array partitioned in
place, so a node sees its own subset, both stopping conditions became reachable, and a split that separates
nothing becomes a leaf instead of two identical children — the defect made every tree a constant, every
importance score a tally of random draws, and every tree a complete binary tree regardless of dataset size.
`BorutaSelectionLayer` selects once and reapplies, with `Reset()` to select again deliberately.
`ConstantColumnFilterLayer.IdentifyByUniqueRatio` compares with `>=` and floors the bar at one, so
`minUniqueRatio = 1.0` is the strictest setting rather than a no-op. `TabularToTensorConverter` gained
`WriteCategories`/`ReadCategories` so the fitted one-hot ordering can be stored beside the weights, and
`Convert` refuses to run without a fitted schema rather than inventing a column order.

Stopped voluntarily at ~5 minutes with coverage incomplete. Report in
`docs/bug-hunts/data-2026-08-01-2316-bugs-game-findings.md`.

**The first entry is the most severe defect found anywhere in the solution today.**

| # | Defect | Why it matters |
|---|---|---|
| **1** | **`FastRandomForest` never splits its data.** `BuildRecursive` computes a feature and a threshold, stores them on the node, and then passes the **identical, unpartitioned** `x` and `y` to both children. Verified by reading. | Two consequences, and the second was not obvious. **The forest cannot predict anything**: every leaf averages the whole target column, so every tree returns a constant, and the "importance" scores are just a tally of which features were picked at random. `BorutaSelectionLayer` and `ShapSelectionLayer` both select features with it, so **feature selection is noise wearing the clothes of statistics**. And because `rows` never shrinks, the `rows < 2` stop can never fire — the only exit is `depth >= _maxDepth`, so it always builds a **complete** binary tree. The constructor permits `maxDepth` up to 64 on the reasonable assumption that data size bounds tree size; it does not. At the default 10 that is ~2 000 nodes per tree and merely wrong; at 30, which the constructor accepts, it is over two billion nodes per tree regardless of how small the dataset is. |
| **2** | **`BorutaSelectionLayer` has no `_fitted` guard**, unlike every sibling in `Prepare/`. It reruns full selection — retraining forests, requiring `context.Targets` — on every `Process()`. | Shares the learned-selection root cause with #1. A second call on the same pipeline instance re-selects, so **the column indices every later layer holds no longer refer to the same columns**. This is precisely the failure the Data README already records as having happened once, in a new place. |
| **3** | **`ConstantColumnFilterLayer.IdentifyByUniqueRatio` is off by one.** With `minUniqueRatio = 1.0` — a value the layer validates and accepts — `count > minUnique` cannot hold for any column, and the "kept nothing, so keep everything" fallback then turns the filter off entirely. | **The strictest available setting silently becomes a no-op.** Worse than an error, because the constant columns this layer exists to remove are exactly what makes a later scaler divide by a zero range. |
| **4** | **`TabularToTensorConverter._categoryMaps` is never persisted**, and `ModelSerializer` validates tensor shape but not schema. | A re-`Fit` at inference with a different category set produces one-hot columns of matching width in a different order. Shapes agree, nothing throws, and the model reads scrambled inputs. |

Fix order: 1 first and alone — it invalidates every consumer of the forest, and until it is fixed any
measurement of Boruta or SHAP selection is meaningless. Then 2 (same family), then 3, then 4.

## ✅ FIXED — audio (found 2026-08-01 by `overfit-find-bugs-game`, fixed 2026-08-02)

**All four fixed 2026-08-02.** The synthetic-speech marker is now what happens when a caller says
nothing: `WavAudioSink` marks the file when `metadata` is omitted, and writing genuinely unmarked audio
requires naming `SyntheticSpeechMetadata.Unmarked` at the call site where a reviewer can see it. The
server's `response_format=pcm` path has no container to carry the marker, so it travels on the media type
(`audio/pcm; synthetic=true; generated-by=Overfit`) and reaches the client in the Content-Type header.
`Mp3Decoder` clamps `big_values * 2` to the 576-entry granule; `WavReader` validates every chunk size
against the file and **fails loudly on a short data chunk** rather than silently transcribing truncated
audio; `EnglishNumberToWords` covers the whole `long` range.

Stopped voluntarily at ~4.5 minutes with coverage incomplete. Report in
`docs/bug-hunts/audio-2026-08-01-2245-bugs-game-findings.md`.

**Read the first entry before the others.** It is the only defect found today that concerns a safety
property rather than correctness, and it was concealed by a document — one written the same day, from
memory, without checking the code.

| # | Defect | Why it matters |
|---|---|---|
| **1** | **The synthetic-speech marker is not enforced anywhere in the synthesis path**, and `Sources/Main/Audio/README.md` asserted that it was. Verified: `OrpheusVoiceEngine` does not reference `SyntheticSpeechMetadata` at all; it is applied by three call sites (two in the CLI, one in the server); `WavAudioSink` takes it as `SyntheticSpeechMetadata? metadata = null`, so omitting it writes an unmarked file; and the server's `response_format=pcm` path emits raw samples with no container to carry it. | **A voice-cloning path that can produce unmarked audio of a real person is the one thing in this repository that must not be merely intended.** Any new caller gets unmarked output by default, which is the wrong direction for a default to fail in. The README has been corrected already — a document asserting a safety property that does not exist is worse than no document, because it stops the next person checking. Enforcement belongs at the engine or the sink, where omission is impossible rather than merely discouraged. |
| **2** | **`Mp3Decoder` indexes past a fixed buffer on a malformed frame.** `_bigValues[g]` is a raw 9-bit side-info field (0–511) used unclamped as `bigValues * 2` — up to 1022 — against a 576-entry array. Unhandled in `Mp3Reader` and `AudioFile`. | A crafted or merely corrupt MP3 crashes the process. This is user-supplied input on a path with no `try`. |
| **3** | **`WavReader` trusts file-supplied chunk sizes.** A negative size throws a raw `ArgumentOutOfRangeException` instead of this project's `OverfitFormatException`, and an oversized-but-positive one **silently truncates the audio** with no signal to the caller. | The truncation is the worse half: shorter audio is not obviously wrong, and everything downstream — transcription, similarity scoring — accepts it. Shares a root cause with #2: a length taken from inside an already-validated envelope and used in buffer arithmetic without a second check. |
| **4** | **`EnglishNumberToWords.Convert` throws on any number ≥ 10¹⁸** — a `Scales` array of six entries indexed from a value that is still inside `long` range. Reachable unhandled from `POST /v1/audio/speech` and from the CLI. | A nineteen-digit number in ordinary text takes down the request. Cheapest fix on this list. |

Fix order: 1 first and separately — it is a policy gap, not a bug, and deserves its own change with a test
that a sink cannot emit unmarked audio. Then 2 and 3 together, since they are one root cause on two parsers.
Then 4.

**Note for the next hunt in `DeepLearning`:** that run predated the agent's README-first step, and
`Sources/Main/DeepLearning/README.md` states the contract *"layers own their parameters;
`TrainableParameters()` is the canonical way to enumerate them"* — a testable assertion that was supplied to
the agent by hand rather than found. A re-run should check every layer's allocations against what it
enumerates and what it disposes, which this one did not systematically do.

---

## ✅ DONE — the `else` sweep (OVERFIT021), 322 → 0

**Status: COMPLETE (2026-07-21).** `else` / `else if` is banned by the in-repo Roslyn analyzer **OVERFIT021**
(`Sources/Analyzers/ElseClauseAnalyzer.cs`). It is *not* an MSBuild task and *not* a `BannedSymbols.txt` entry —
that file bans **API symbols**, and `else` is a language keyword, so it cannot be expressed there. An
MSBuild-task variant with an `ElseDebt.txt` ledger was built and then deleted in favour of the analyzer (real
syntax tree, IDE squiggles, per-directory severity).

The ratchet is finished and retired: the rule is now **`error` across every project except `Tests`**, wired
centrally in `Directory.Build.props` rather than per-csproj, so the per-directory allow-list in `.editorconfig`
is gone. `Tests` stays at `suggestion` (test code is local and disposable). The **only** remaining `else` sites
in the repo are 6 in `Sources/Benchmark/ElseRefactorBenchmark.cs` — intentional, since the `else` forms are that
benchmark's measurement subject, and the project is excluded from the analyzer.

Verified semantically rather than by grep: the solution builds clean with the rule at `error` globally.

### Cost is measured, not assumed — `Sources/Benchmark/ElseRefactorBenchmark.cs`

| rewrite | ratio |
|---|---|
| `if/else` → ternary | 1.01 |
| `else if` chain → `continue` guards | 1.00 |
| invert to rare-branch-first + `continue` | 1.01 |
| extract method, **JIT inlines it** | 1.01 |
| extract method, **JIT does NOT inline it** | **2.25×** |

So in-place rewrites are free and **the only real risk is extracting a method**. `try/finally` bodies are
*never* inlined; large bodies and lambdas usually block it too. In hot paths (`LanguageModels`, `Ops`,
`Kernels`, `Intrinsics`) extract only after confirming the JIT inlines it — otherwise leave the `else` as debt.

### Two traps this sweep already hit — do not rediscover them

1. **A guard-clause `return` can silently skip trailing code.** In `OverfitLicense` the first rewrite moved the
   Android case to `return`, which skipped the `Debug.WriteLine` *after* the branch. These sites cannot be
   scripted: `else` may only be removed once you have read the whole method and confirmed the branch really
   does always exit.
2. **`.editorconfig` scoping.** A `[section]` header scopes everything below it (so directory sections belong at
   the end of the file), the glob must be `<dir>/**.cs` not `<dir>/**/*.cs`, and an ID listed in
   `<WarningsNotAsErrors>` reverts `error` back to warning even where `.editorconfig` promotes it.

---

## ✅ CLOSED — CPU PREFILL + DECODE PERF TRACK (2026-07-22 → 2026-07-23)

**Both compute-side performance tracks are closed, by measurement rather than assertion.** The detailed,
chronological record is preserved below — every win, every reverted negative, and the measurement discipline
that produced them. The headline:

| | start of track | end of track |
|---|---:|---:|
| **prefill** | 143 tok/s (3.76× behind llama.cpp AVX-512) | **~299 tok/s (~1.81×)** — 1.14× to their AVX2 build |
| **decode** | — | **memory-bound, 1.13× behind**, GEMV kernel at 82% of the DRAM ceiling |

**What shipped this track** (all bit-identical or coherence-safe, pinned by parity tests): Q6_K tiled GEMM,
shared activation quantization, whole-matrix O / Q / K/V projections, register-tiled prefill kernels, the
F16-scale hoist (decode once per projection, not once per column tile), **AVX-512 Q4_K and Q6_K prefill
kernels** (pair what is already adjacent in memory — the choice of what shares a register turned a −20%
port into +12%), register-resident attention value accumulation, and vectorized softmax exp.

**Why it is closed:**
- **Prefill.** Our Q4_K matmul measured *faster* than llama.cpp's at equal ISA and thread count (1.70 vs
  1.56 TFLOP/s), so the remaining gap is AVX-512 coverage (now largely done) and kernel structure (done).
  What is left — a flash-attention GEMM for the `attn_scores` dot (~2% e2e), the scalar `Unpack` (~3.5% of
  one kernel) — is high-effort, low-return.
- **Decode.** `DecodeGemvRooflineBenchmark` showed the kernel's compute runs at 132.5 GB/s hot, *above* the
  90 GB/s DRAM ceiling, and 82% of it when streaming from DRAM — so decode is memory-bound and an AVX-512
  decode kernel cannot help (the direct measurement behind the reverted decode-port negative). The whole-model
  shortfall is per-token overhead and layer→layer serial latency, where we are already 1.13× of llama.cpp.

**The most valuable output was the measurement discipline** — roughly eleven mechanism hypotheses refuted,
the rules that survived recorded in the `feedback-measurement-discipline` memory, and permanent infrastructure
left behind: `MachineRooflineBenchmark`, `DecodeGemvRooflineBenchmark`, `Diagnostics/Throughput.cs`, and the
BenchmarkDotNet throughput columns.

**Next move is a business decision about product direction, not another *LLM* kernel.** Any further
perf work should measure the ceiling before writing code — the discipline that made this track pay.

#### ⚠ BUT: the largest untouched perf reserve in the project is CNN inference, not LLM — 13.2× behind ORT

Measured 2026-07-23, `LargeCnnComparisonBenchmark`, VGG-16 (~15.5 GFLOPs/inference), same box:

| | time | GFLOP/s | % of this box's float ceiling (2.19 TFLOP/s) |
|---|---:|---:|---:|
| ONNX Runtime (native MLAS) | **9.96 ms** | 1557 | **71%** |
| Overfit (im2col + GEMM, DAG importer) | **131.7 ms** | 118 | **5.4%** |

Parity is exact (maxAbsDiff 6.7e-8, cosine 1.000000, same argmax) — this is purely speed. For scale: the
whole prefill sprint above chased a **1.8×** gap on a path already near half of its instruction mix's
ceiling. This is a **13×** gap on a path at 5% of the machine ceiling.

**A framing correction this exposes.** The README headline "~8× faster than ONNX Runtime" is measured on
`Linear(784 → 10)`, where ORT's *per-call overhead* dominates — it is a real result for small-model,
in-process serving, but it says nothing about kernel quality. VGG-16 is compute-dominated and is the honest
kernel-vs-kernel test. Both statements are true; only the second describes the kernels.

**Where the 13× splits — worker sweep, ORT stable at 11.8–12.0 ms throughout as the canary:**

| workers | Overfit | GFLOP/s | speedup |
|---|---:|---:|---:|
| 1 | 663.2 ms | 23 | 1.00× |
| 4 | 256.4 ms | 60 | 2.59× |
| 16 | 138.4 ms | 112 | 4.79× |
| default (32) | 141.3 ms | 110 | 4.69× |

Two independent problems, both large:
1. **Per-core kernel: 23 GFLOP/s against a ~137 GFLOP/s single-core ceiling — 17% efficiency.**
2. **Parallel scaling: 16 workers buy 4.79×, not ~14× — 30% efficiency.**

**The most useful comparison is internal.** Our Q4_K prefill GEMM runs at 2.15 TFLOP/s over 32 threads
≈ **134 GFLOP/s per core** — the same project, the same machine, ~6× the per-core efficiency of the conv
GEMM. We demonstrably know how to write a competitive GEMM (it measured faster than llama.cpp's at equal
ISA); the convolution path simply does not use that class of kernel. The techniques that paid there —
register tiling, weight-stationary reuse, hoisting fixed per-block work, counting how often work repeats —
have not been applied here at all.

*Caveat before targeting a number:* VGG-16 is entirely 3×3 convs, where ORT's MLAS may use Winograd (a
2.25× FLOP reduction), so its 1557 GFLOP/s is not necessarily 71% of the hardware ceiling in executed FLOPs.
This repo measured Winograd as a **negative** (+79% on deepcnn) with its current infrastructure. Wall-clock
is what matters, and wall-clock says 13×.

#### ★★ PARALLEL im2col — VGG-16 141 → 73 ms, gap to ORT 13.2× → 6.3×

The GEMM was already parallel; **the im2col patch gather never was**. On VGG-16 that gather is enormous —
`conv1_2` alone materialises a `[576 × 50176]` matrix (115 MB) one scalar element at a time — and an Amdahl
fit over the worker sweep put the serial fraction at ~15.5%, i.e. **~103 of the 138 ms at 16 workers**.
`Im2Col` now fans out over `krow`: each row owns a disjoint `n`-element slice of `cols` and only reads the
input, so no synchronisation is needed and the result is bit-identical. Gated below `K·N < 65536` so small
convs keep the serial path; `OVERFIT_PARALLEL_IM2COL=0` restores it.

| workers | before | after | |
|---|---:|---:|---|
| 1 | 663 ms | 668 ms | unchanged — the control: no work was added, only spread |
| 4 | 256 | 226 | |
| 16 | 138 | **87.7** | |
| default (32) | 141 | **73.0** | 110 → **212 GFLOP/s** |

Speedup over one worker **4.69× → 9.16×**; serial fraction **15.5% → 7.3%**. Parity unchanged
(maxAbsDiff 6.7e-8, cosine 1.000000, same argmax), conv tests 56/0, suite 1499/0/232. The paired A/B run had
ORT flat at 11.66 vs 11.75 ms as the canary.

#### ▶ RETRACTED — there is no "~49 ms of serial work". Conv is simply 91% of the time

The Amdahl fit above predicted ~49 ms of serial residue dominating at 32 workers. `OnnxGraphModel.ProfileNodes`
(new, opt-in, the CNN counterpart of `PrefillProfiler`) measured it per operator instead:

| operator | 32 workers | share | 1 worker |
|---|---:|---:|---:|
| **ConvLayer** (13 nodes) | **72.10 ms** | **90.7%** | 1557 ms |
| MaxPool2DLayer (5) | 5.22 | 6.6% | 5.46 |
| ReluActivation (13) | 1.92 | 2.4% | 1.70 |
| LinearLayer (1) | 0.21 | 0.3% | 0.23 |
| GlobalAveragePool2DLayer | 0.01 | 0.0% | 0.01 |

MaxPool and ReLU *are* serial — they do not move between 1 and 32 workers — but together they are **7.1 ms,
9% of the total, not 49 ms**. The Amdahl model overestimated the serial share by more than 6×, because a
clean serial/parallel split does not describe this system: Conv's scaling is imperfect (21.6×), not absent,
and imperfect scaling reads as "serial fraction" to that fit. **Treat Amdahl fits as a pointer, not a
measurement — it was right that something was wrong, and wrong about what and how much.**

**So the next lever is kernel quality, not parallelism.** Parallelising MaxPool + ReLU is real but capped at
~1.1× overall.

#### ★ im2col vs GEMM, and a FLOP-counting correction that changes the target

`Conv2DGemmKernels.ProfileParts` (opt-in) + `ConvGemmPartProfileTests` split conv time on VGG-16:

| part | per run | share of conv |
|---|---:|---:|
| im2col gather | 9.33 ms | 14.7% |
| **GEMM** | **54.22 ms** | **85.3%** |

So after parallelising the gather, **the micro-kernel is the target** — confirmed rather than assumed.

**Correction to every VGG GFLOP/s figure above.** They used "15.5 GFLOPs/inference", which is the commonly
quoted VGG-16 **MAC** count. Under this project's convention (MAC = 2 ops, matching `Throughput` and
llama.cpp's `test-backend-ops`) VGG-16's conv layers are **30.7 GFLOP**. The earlier rates were understated 2×:

| | time | GFLOP/s | % of the 2190 GFLOP/s ceiling |
|---|---:|---:|---:|
| our GEMM alone | 54.2 ms | **566** | **26%** |
| whole model, Overfit | 73 ms | 420 | — |
| whole model, ORT | 11.6 ms | **2647** | **121%** ⚠ |

**ORT "achieves" 121% of this machine's measured float ceiling, which is impossible** — so it is executing
fewer operations than the formula counts. That is direct evidence for the Winograd hypothesis flagged
earlier: MLAS uses a FLOP-reducing transform on 3×3 convs (F(2,3) cuts FLOPs 2.25×). Corrected, ORT runs at
roughly **1176 GFLOP/s ≈ 54% of ceiling**.

**This reframes the gap: part of ORT's lead is algorithmic, not kernel craft.** Our GEMM at 26% against
their ~54% is about **2×** of kernel-quality difference, with the rest coming from doing less work.

#### ▶ REFUTED — the micro-kernel tile shape is NOT the problem; it is already at hardware peak

The hypothesis was that `Mr=8 × Nr=8` is load-port bound (9 loads per 8 FMAs) and that a `6×16` tile would
pay. `GemmMicroKernelShapeBenchmark` measured the candidate shapes single-threaded, accumulators in named
locals, panels L1-resident:

| shape (1 thread) | GFLOP/s | vs today |
|---|---:|---:|
| **AVX2 8×8 (today)** | **148** | 1.00× |
| AVX2 6×16 | 182 | 1.23× |
| AVX2 4×24 | 182 | 1.23× |
| AVX-512 8×16 | 276 | 1.86× |
| **AVX-512 8×32** | **337** | **2.27×** |
| AVX-512 6×48 | 337 | 2.27× |

**AVX2's single-core FMA peak is ≈138 GFLOP/s** (8 lanes × 2 ops × 2 FMA units × ~4.3 GHz), and the current
shape measures **148** — it is already at the hardware ceiling, boost clock and all. The load-port argument
was wrong: Zen 5 sustains those loads. Reshaping the AVX2 tile is worth ~1.2×, not the 4× the production
deficit implies.

**What this reveals instead.** The micro-kernel can do 148 GFLOP/s per core → ~2370 GFLOP/s across 16 cores.
Production conv GEMM does **566 — 24% of what its own micro-kernel achieves when fed properly.** The kernel
is fine; **everything around it is not**: B-panel packing, the memory traffic of a `[K, N]` im2col matrix
(conv1_2's is 115 MB, far past any cache), and panel scheduling. That is the target, not the tile.

AVX-512 is separately worth **2.27×** on the micro-kernel — but only to the extent production is
compute-bound, and at 24% efficiency it plainly is not. Expect far less than 2.27× end to end.

#### ▶ REFUTED AGAIN — packing is not it either. The micro-kernel is starved by the cache hierarchy

Ablation inside `GemmNPanelWorker` (`AblatePackB` / `AblateMicroKernel`, measurement-only), VGG-16:

| arm | ms/run | share |
|---|---:|---:|
| baseline (pack + micro) | 76.04 | 100% |
| pack only | 27.61 | 36.3% |
| micro only | 67.66 | 89.0% |

Netting out the rest of the model (im2col 9.3, MaxPool+ReLU 7.1): **packing ≈ 11 ms, micro-kernel ≈ 51 ms.**
The negative "unattributed" (−19 ms) is expected overlap — removing either side frees cache for the other —
so both figures are upper bounds. Either way the micro-kernel dominates the GEMM, and the strided scalar
pack, plausible as it looked, is the minority cost.

**The one number that matters.** The same micro-kernel measures **148 GFLOP/s per core in isolation** and
**~38 GFLOP/s per core in production** (51 ms for 30.7 GFLOP over 16 cores) — **4× slower running the same
instructions.** The difference is the memory feed: `packB` is `K × Nr` floats, which at K=2304 is **73 KB
against a 32–48 KB L1**, so every row-block re-streams the panel from L2, and the A rows (8 × K floats,
another 73 KB) do the same. The isolation benchmark had both in L1, which is exactly why it hit peak.

**So the target is K-blocking** — split the contraction so `packB` and the A slice fit L1. One caveat that is
mine to state: the kernel's own comment records that a BLIS-style **K-blocked + A-packed** variant was tried
and regressed (vgg 140 → 189 ms). That measurement predates the parallel im2col and bundled A-packing, whose
one-time `O(M·K)` cost may have dominated it. It is not proof that K-blocking alone fails — and equally, not
licence to repeat it blind. Measure the L1-residency effect on a single VGG layer shape first.

#### ▶ REFUTED — K-blocking does not help either, and the prototype exposes where the loss really is

`GemmKBlockingBenchmark` runs the full conv5_1 GEMM (M=512, K=4608, N=196) single-threaded at several
contraction blocks. `Kc=4608` is today's unblocked kernel (144 KB packed panel); the rest bring it inside L1:

| Kc | packed panel | ms | TFLOP/s |
|---|---:|---:|---:|
| **4608 (today)** | 144 KB | **6.99** | **0.13** |
| 1152 | 36 KB | 7.24 | 0.13 |
| 512 | 16 KB | 7.25 | 0.13 |
| 256 | 8 KB | 8.00 | 0.12 |
| 128 | 4 KB | 7.96 | 0.12 |

**Unblocked wins.** Making the panel L1-resident is flat to slightly worse, so the L1-residency hypothesis is
refuted and the earlier K-blocking negative is independently confirmed — this time without A-packing to
confound it.

**But the prototype answers a better question than the one asked.** It does everything production does —
pack, micro-kernel, real memory — single-threaded at **132 GFLOP/s**, while production conv at one worker
runs at **19.7 GFLOP/s** (30.7 GFLOP in 1557 ms). Same structure, **6.7× apart**. So the deficit is neither
the tile, nor the pack, nor cache blocking: it is **shape-dependent**, and this prototype picked a shape
where everything is fine (small N, large K).

The suspicion now points at the early layers, where the arithmetic-to-overhead ratio inverts. `conv1_2` is
M=64, K=576, **N=50176**: A (147 KB) is re-read for each of **6272 panels**, and the pack does one scalar
branchy copy per FMA issued. **Next measurement: per-layer conv timing, not per-operator** — then each
layer's achieved GFLOP/s against its own shape.

**Four hypotheses refuted in a row on this path** — tile shape, packing, L1 blocking, and the "49 ms serial"
model. Each cost minutes to measure; the rewrites they prevented would have cost days.

#### ★★★ FOUND IT — A is re-read once per N-panel: 7.5 GB of traffic for 30.7 GFLOP of work

The per-layer profile (`PerNodeProfileReport`) shows the conv layers are **uniform**, 347–581 GFLOP/s, with
no outlier — so "the early layers are the problem" is refuted too. The shape table is where it shows:
`GemmNPanelWorker` sweeps M *inside* the panel loop, so the whole A matrix is re-read **for every N-panel**.

| layer | panels | A | A traffic |
|---|---:|---:|---:|
| conv2 | 6272 | 144 KB | 903 MB |
| conv4 | 1568 | 576 KB | 903 MB |
| conv6 / conv7 | 392 | 2304 KB | 903 MB each |
| conv9 / conv10 | 98 | 9216 KB | 903 MB each |
| others | | | ~2.1 GB |
| **total** | | | **≈7.5 GB per inference** |

**7.5 GB moved for 30.7 GFLOP computed = 0.24 bytes/FLOP**, where a well-blocked GEMM runs at ~0.01 — **24×
more traffic than the arithmetic requires**. And it matches the clock: 7.5 GB in 73 ms is **103 GB/s**,
against a measured 90 GB/s DRAM read ceiling (L3 is faster, but finite and shared by 16 cores).

**The conv GEMM is bandwidth-bound on re-reading A** — which is why a single-threaded prototype hit
132 GFLOP/s while production gets ~30 per core: one thread has the cache to itself.

**Fix: block over N-panels.** Process a group of panels (e.g. 8 = 64 columns) and sweep M once per group,
cutting A traffic by the group size — 7.5 GB → ~0.94 GB at 8 panels. This is the outer half of the standard
Goto/BLIS structure, and it is the piece this kernel has never had.

**Why every earlier hypothesis missed it:** tile shape, packing and K-blocking are all *within* one panel.
The waste is *between* panels, which no measurement scoped to a single panel could see.

#### ▶ REFUTED — N-panel grouping does nothing, and the traffic argument was wrong about *where*

`Conv2DGemmKernels.NPanelGroup` (default **1**, `OVERFIT_CONV_PANEL_GROUP`) packs and sweeps several panels
together so A is read once per group. Interleaved against an ORT canary (4% spread over the whole sweep):

| group | 1 | 2 | 4 |
|---|---:|---:|---:|
| VGG-16 | 72.7 ms | 73.0 | 73.3 |

0.8% apart — inside the noise. **The 7.5 GB figure was right; the conclusion drawn from it was not.** That
traffic never reaches DRAM: this CPU has **128 MB of L3 (V-cache)**, so every layer's A (≤9 MB) is re-read
from L3. Counting bytes without asking *which cache level serves them* is worthless.

#### ★★ `Sources/MachineProbe` — a standalone hardware probe, and it explains both blocking failures

A console app with **no reference to Overfit, no BenchmarkDotNet, no packages** — one file, `Stopwatch` only,
so it can be run on a customer box, a CI runner or a cloud VM before anyone reads meaning into an Overfit
number. `dotnet run -c Release --project Sources/MachineProbe`.

On the 9950X3D:

| peak FMA | 1 core | all cores | scaling |
|---|---:|---:|---:|
| 128-bit | 84 GF/s | 1336 | 15.8× |
| 256-bit | **179** | **2310** | 12.9× |
| 512-bit | 351 | 4200 | 12.0× |

Memory: read **89.0 GB/s**, copy 74.0, triad 49.5.

**Correction it forces:** the AVX2 single-core peak was *estimated* at 138 GF/s, which made the 148 GF/s
micro-kernel look like it exceeded the hardware. Measured, the peak is **179** — the micro-kernel is at
**83% of it**, still high, but the earlier claim was arithmetic, not measurement.

**The working-set sweep is the real payload** (one core, sequential read):

| 8 KB | 48 KB | 512 KB | 8 MB | 32 MB | 128 MB |
|---:|---:|---:|---:|---:|---:|
| 72.9 | 74.6 | 75.3 | 76.0 GB/s | 68.9 | 58.5 |

**Flat from L1 to 8 MB.** One core reads ~75 GB/s *wherever the data lives* — for streaming access this
machine has **no L1/L2/L3 cliff at all**, because the prefetcher keeps up.

*The first version of that sweep was wrong and the conclusion drawn from it is withdrawn.* It used a single
`Vector<float>` accumulator, so it measured the chain's **latency** (~75 GB/s, below every cache level's
bandwidth) and produced a perfectly flat curve that appeared to prove "this machine has no cache cliff". With
eight independent streams the structure appears — see below. The probe now lives in
`Tests/Diagnostics/MachineProbeTests.cs` (xUnit, `ValueStopwatch`, asserts its own loops allocate 0 B).

#### ★★★ WHY PARALLEL SCALING IS POOR — bandwidth stops scaling past ~2 MB per core

| working set | 1 core | all cores | scaling |
|---|---:|---:|---:|
| 16 KB – 2 MB | ~76 GB/s | 700–900 GB/s | **9–12×** |
| 8 MB | 67.3 | 112.8 | **1.7×** |
| 32 MB | 64.4 | 67.6 | 1.1× |
| 128 MB (DRAM) | 59.6 | 63.6 | 1.1× |

**Compute scales 12–15×; bandwidth scales 10× only while the per-core working set fits private cache, then
collapses to ~1×.** One core already draws 60% of total DRAM bandwidth; the other fifteen add 60%.

The cliff lands exactly where **16 cores × 8 MB = 128 MB = this chip's L3 including V-cache** — the
measurement validates itself against a number it was never given.

**So any kernel that outruns private cache cannot be fixed by more cores or by blocking — only by needing
fewer bytes per FLOP.** That reframes conv: the fix is arithmetic intensity, not scheduling.

#### ★★ AVX-512 8×32 conv micro-kernel — VGG-16 72.7 → 63.7 ms (1.14×)

An `Mr×Nr` tile loads `Mr+Nr` floats per k-step and performs `2·Mr·Nr` FLOPs, so intensity is
`Mr·Nr / (2(Mr+Nr))`: **2.0 FLOP/byte at 8×8, 3.2 at 8×32**. AVX-512's 32 registers make 16 accumulators
plus 2 B vectors and a broadcast fit. Interleaved A/B, three rounds, ORT canary within 2%:

| | median | GFLOP/s |
|---|---:|---:|
| AVX2 8×8 | 72.7 ms | 422 |
| **AVX-512 8×32** | **63.7 ms** | **482** |

Parity exact in every round, conv tests 56/0, `OVERFIT_CONV_AVX512=0` falls back. **Gap to ORT 6.3× → 5.35×.**

*Honest note on the model:* intensity predicted up to 1.6× and delivered 1.14×, so intensity is a real but
not dominant term — do not extrapolate a further tile widening from it without measuring.

**Machine identity, measured rather than reported** (`MachineProbeTests`): AMD Ryzen 9 9950X3D, **5.59 GHz**
from a dependent-add chain — cross-checked against 5.53 GHz derived independently from the AVX2 FMA peak,
agreeing to 1%.

*Unexplained and therefore not built on:* the standalone driver measures 1564 ms at one worker where the
BenchmarkDotNet sweep measured 668 ms — same variable, same box. The per-operator conclusion rests on the
default-worker numbers, where the two agree (73 vs 79.5 ms); the single-worker column is indicative only.

---

<details>
<summary>▼ Full chronological record of the perf track (preserved)</summary>

### PREFILL — starting point: measured 3.76× behind llama.cpp, compute-bound

**Measured 2026-07-22, same file (`qwen.q4km.gguf`), same 672-token prompt, best configuration on both sides:**

| | prefill (pp672) | notes |
|---|---:|---|
| llama.cpp b10088 (built from `D:\llamacpp-tmp`, `/arch:AVX512`, 16 threads) | **541.7 ± 2.6 tok/s** | `llama-bench -p 672 -n 0 -r 3` |
| Overfit (sidecar `.repack` present, 32 workers) | **144 tok/s** | `PrefillProfileTests` |
| | **3.76×** | |

**Not a thread-configuration artefact.** Worker sweep: 8 → 92, 16 → 122, 24 → 130, 32 (default) → 144 tok/s —
monotonic, default is best. Prefill *scales* with cores, unlike decode (which has a cliff at
`workers == procCount`). The gap is algorithmic.

**This corrects the previous heading here, which read "the cheap CPU-perf levers are exhausted".** That was
true of **decode** and was wrongly generalised to performance as a whole. The two paths are not alike:

| path | gap to llama.cpp | why |
|---|---|---|
| decode | **1.13×**, uniform across context | memory-bound, sitting on the DRAM floor |
| **prefill** | **3.76×** | compute-bound — there is no floor here |

**The reference kernel is NOT tinyBLAS.** `ggml/src/ggml-cpu/llamafile/sgemm.cpp` contains no
`GGML_TYPE_Q4_K` case at all. The Q4_K prefill path is `ggml_gemm_q4_K_8x8_q8_K` in
`ggml/src/ggml-cpu/arch/x86/repack.cpp` (~1450 lines) — the same `block_q4_Kx8` repacked layout Overfit
already uses. So this is not a missing algorithm; it is the same algorithm implemented far better.

**Per-projection micro-bench (2026-07-22, `Q4KPrefillProjectionBenchmark`, 672 rows, real Qwen-3B shapes) —
this REFUTED the first hypothesis written here, which claimed the tiled kernel "wins nothing":**

| shape | Tiled | WeightStationary | ReDecodePerRow | Tiled 1-thread |
|---|---:|---:|---:|---:|
| `ffn_gate_up` (2048→11008) | **15.44 ms** | 53.62 ms | 86.55 ms | 169.3 ms |
| `ffn_down` (11008→2048) | **17.27 ms** | 55.55 ms | 85.61 ms | 169.7 ms |
| `attn_qo` (2048→2048) | **4.41 ms** | 11.85 ms | 20.64 ms | 32.2 ms |

`GemmTiled` is **~3.2–3.4× faster than weight-stationary**, exactly as its own docs claim. It is a real GEMM
and it already carries the FFN in production (a `.repack` sidecar sets `IsPrepacked`, which routes every
bias-free projection through it).

**So why did the 2026-07-21 end-to-end A/B tie at 0.999×?** Because that A/B only moved the *biased*
projections — attention Q/K/V. Those are **88% of the dispatch count but only ~6% of the FLOPs**: Q is
dispatched per head at 2048→128, while one FFN layer is 3 × 30.3 GFLOP. The tie was real and correctly
measured; it simply measured the small projections. **Dispatch count is not work — always weight a path
census by FLOPs before drawing a conclusion from it.**

**The real gap is kernel throughput.** At 672 rows a projection is 30.3 GFLOP, so our best kernel runs at
**≈1.9 TFLOP/s** (15.4 ms) against llama.cpp's **≈3.7 TFLOP/s** whole-model rate — a **~1.9× kernel gap**,
not a missing algorithm. The residual beyond that is dispatch overhead in the per-head attention path, where
the same activation matrix is re-quantized once per head.
**The gap decomposes — measured, not assumed.** llama.cpp was rebuilt AVX2-only
(`-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF`, `D:\llamacpp-tmp\build-avx2`) and re-benched on the
same file:

| build | pp672 |
|---|---:|
| llama.cpp, AVX-512 | 539.9 tok/s |
| llama.cpp, AVX2 only | 336.7 tok/s |
| Overfit, AVX2 | 144 tok/s |

**3.76× = 2.34× (kernel quality at equal ISA) × 1.60× (AVX-512).**

This **refutes the ranking first written here**, which called AVX-512 "the most likely source of ~2×". It is
the *smaller* factor. Porting the kernel to AVX-512 caps out at 1.60×; the larger 2.34× is available without
touching the instruction set. Note also that the old "AVX-512 ≈ 0" result stands for **decode** (memory-bound,
where wider SIMD cannot help by construction) — here it is worth 1.60×, so that negative genuinely does not
transfer to compute-bound prefill.

### Prefill component breakdown — measured 2026-07-22 (`PrefillProfiler`, first time in the project)

Qwen-3B Q4_K_M, 672-token prompt, median of 3 (`PrefillProfileTests.Prefill_ComponentBreakdown`):

```
total/request : 4697.5 ms (143 tok/s)
  attention   : 1595.7 ms  34.0%   (36 calls)
  ffn         : 3000.8 ms  63.9%   (36 calls)
    attn_kv   :  179.7 ms   3.8%   ( 72)
    attn_q    :  624.1 ms  13.3%   (576)   <- per head
    attn_scores: 263.9 ms   5.6%   (576)
    attn_out  :  427.9 ms   9.1%   (576)   <- per head
    ffn_gateup: 1222.5 ms  26.0%   ( 36)
    ffn_down  : 1778.1 ms  37.9%   ( 36)   <- biggest single item
  other       :  101.0 ms   2.1%
```

### ▶▶ THE NEXT LEVER: Q6_K has no batched prefill kernel

`ffn_down` costs **more** than `ffn_gateup` while doing **half** the work (one 30.3 GFLOP projection vs two).
Per layer that is 0.61 TFLOP/s against gate_up's 1.78 — a 2.9× efficiency gap that the micro-bench did *not*
show (Tiled: 17.27 vs 15.44 ms). So production is not taking the same path. Cause, confirmed by dumping the
GGUF tensor types:

- **`ffn_down` is Q4_K ×18 + Q6_K ×18** (and `attn_v` likewise) — half the layers are Q6_K.
- In `BatchedQuantProjection`, the Q6_K branch has **only `Q6KDotKernel.ProjectBatched`** (re-decode per row).
  There is **no `ProjectBatchedWeightStationary` and no `GemmTiled` for Q6_K**, while Q4_K has both.
- Arithmetic checks out: 18 layers × 17.3 ms (tiled) + 18 × X = 1778 ms ⇒ X ≈ 81.5 ms, and the micro-bench
  measured `ReDecodePerRow` at 85.6 ms for that shape.

**The repack layout for Q6_K already exists** (`Q6KRepack`, `RowsInterleaved = 8`, `Q6KGemvKernel.GemvParallel`)
— it is wired for *decode* only. So this is filling a gap in an existing kernel family, not inventing one.

**Estimated payoff: `ffn_down` 1778 → ~670 ms ≈ 1.1 s of 4.7 s (~23%), i.e. 143 → ~187 tok/s (1.31×).**
An estimate, not a promise — Q6_K does more work per weight (6-bit vs 4-bit) than the Q4_K kernel it is
modelled on.

**Attack order, by value/risk rather than by ceiling:**

| lever | ceiling | risk |
|---|---|---|
| **Q6_K batched prefill kernel** | ~1.31× | **low** — layout exists, structure copied from Q4_K |
| AVX-512 port | 1.60× | high — intrinsics rewritten from scratch |
| per-head attention (`attn_q` + `attn_out` = 22.4%, 576 dispatches each) | unknown | medium — dispatch restructuring |

#### ✗ Q6_K weight-stationary — BUILT, MEASURED +13.5% SLOWER, REVERTED (2026-07-22)

`Q6KDotKernel.ProjectBatchedWeightStationary` was written on the Q4_K model: unpack each super-block once
into scratch, contract against a 64-row tile. Bit-identical (12/12 parity tests, including tile-boundary and
no-bias cases). Measured on the real model:

| component | before | after | Δ |
|---|---:|---:|---:|
| `ffn_down` | 1778.1 ms | **2018.0 ms** | **+13.5%** |
| `ffn_gateup` *(canary)* | 1222.5 ms | 1247.5 ms | +2.0% |
| `attention` *(canary)* | 1595.7 ms | 1616.7 ms | +1.3% |

Canaries drifted 1–2%, `ffn_down` moved 13.5% — a real regression, reverted.

**Two mistakes in the analogy, both worth remembering.** (1) Q4_K's weight-stationary hoists only the
*scale/min* decode; the 4-bit nibble unpack still happens **in registers, per row**. I hoisted the entire
6-bit unpack into a 256-byte stack buffer, so every row now stores and reloads it through L1 instead of
consuming it from registers. (2) Inverting the loop order made activation reads strided (one 256-byte slice
per row, 11 008 bytes apart) instead of streaming a row contiguously.

**So the Q6_K gap is not closed by the obvious transform.** The right analogue to Q4_K's 3.3× is the *tiled*
kernel over the repacked `block_q6_Kx8` layout — and `Q6KRepack` already produces that layout for decode.

#### ✅ Q6_K tiled GEMM — SHIPPED, prefill 143 → 185 tok/s (1.29×)

`Q6KGemvKernel.GemmTiled` unpacks each weight super-block once and holds it **in registers** across a tile of
up to 16 activation columns — the opposite of the reverted weight-stationary attempt, which pushed the unpack
through a stack buffer. Wired into `BatchedQuantProjection` via `DispatchTiledQ6K` (gate:
`UseTiledPrefillQ6K && bias.IsEmpty && CanRepack && AVX2 && FMA`).

| component | before | after | Δ |
|---|---:|---:|---:|
| `ffn_down` | 1778.1 ms | **713.6 ms** | **−59.9%** (2.49×) |
| `ffn_gateup` *(canary)* | 1222.5 ms | 1242.3 ms | +1.6% |
| `attention` *(canary)* | 1595.7 ms | 1571.5 ms | −1.5% |
| **prefill total** | **4697.5 ms · 143 tok/s** | **3632.6 ms · 185 tok/s** | **−22.7% · 1.29×** |

Canaries within ±1.6%, and an independent run of `PrefillPathAbTests` measured 186 tok/s. The estimate that
motivated the work (1778 → ~670 ms, 143 → ~187 tok/s) landed almost exactly.

**Correctness.** `Q6KTiledGemmParityTests` pins `GemmTiled` bit-identical to `GemvAvx2` per column (6 cases).
End-to-end the first generated token is **576, unchanged** from before the kernel. Note this is *coherence*
evidence, not byte-parity: the old path (`ProjectBatched`, non-repacked) associates the reduction differently
from the repacked kernels, so outputs differ in the low bits — the same standard `OVERFIT_REPACK_ATTN` is held
to.

**Cost:** `Q6KWeight` has no prepacked-sidecar path, so `EnsureRepacked()` allocates a heap copy of the Q6_K
tensors on first use. Worth revisiting if RAM matters more than TTFT.

**Gap to llama.cpp: 3.76× → 2.93×.** Remaining, by measured share: `ffn_gateup` 34.2%, `attn_q` + `attn_out`
28.9% (the per-head dispatches, 576 calls each), `attn_scores` 6.8%. AVX-512 (ceiling 1.60×) still last.

#### ▶▶ NEXT LEVER (sized 2026-07-22): hoist activation quantization out of the per-head loop — ~18.8%

`Q4KPrefillProjectionBenchmark.QuantizeActivationsOnly` measures Q8_K quantization of `672 × 2048`
activations at **~1.0 ms**. Against the profile:

| | dispatches over `hidden` | quantization cost | actually needed |
|---|---:|---:|---:|
| `attn_q` (621.7 ms / 576 calls = 1.079 ms) | 576 | ~576 ms | — |
| `attn_kv` (181.4 ms) | 144 | ~144 ms | — |
| **total** | **720** | **~720 ms** | **36** (once per layer) |

So **~93% of a Q-head dispatch is activation quantization** — the projection itself is 2048→128, roughly
0.08 ms. `hidden` is loop-invariant across heads, so the same matrix is quantized 16× per layer. `attn_out`
is NOT affected: its input is the per-head `attn` band.

**Recoverable ≈ 684 ms of 3632.6 ms ≈ 18.8% → prefill 185 → ~228 tok/s.**

Decode already fixed exactly this in 2026-05 (`ProjectPreQuantized`, "hidden was re-quantized per head, now
quantized once per layer"); the batched prefill path never got the equivalent.

#### ✅ SHIPPED — shared activation quantization: 185 → 194 tok/s (1.05×), but 3.3× short of the estimate

`BatchedQuantProjection.Dispatch` takes optional pre-quantized Q8_K scratch;
`CachedMultiHeadAttention.DecodeBatchedQuant` quantizes `hidden` once per layer and passes it to every Q/K/V
dispatch. Q4_K and Q6_K share the Q8_K format bit-for-bit, so one buffer serves all three.

| component | before | after | Δ |
|---|---:|---:|---:|
| `attn_q` | 621.7 ms | **456.5 ms** | −26.6% |
| `attn_kv` | 181.4 ms | **139.3 ms** | −23.2% |
| `ffn_gateup` *(canary)* | 1242.3 ms | 1211.0 ms | −2.5% |
| `attn_out` *(canary)* | 427.4 ms | 427.7 ms | +0.1% |
| **prefill total** | **3632.6 ms · 185 tok/s** | **3462.3 ms · 194 tok/s** | **−4.7% · 1.05×** |

**The estimate said ~684 ms; the measurement says ~207 ms — 3.3× optimistic.** Cause: the sizing benchmark
timed quantization of a 672×2048 block **in isolation** (~1.0 ms), i.e. reading 5.5 MB cold. In production
the 16 repeats run back-to-back on a cache-resident `hidden`, so the redundant passes were far cheaper than
the isolated measurement implied. **Lesson: an operation benchmarked alone over-states its cost when the
thing you are removing is a repeat on hot data — size the repeat, not the first call.**

#### ✅ SHIPPED — whole-matrix O projection: 194 → 219 tok/s (1.13×)

Per head the O projection is `[headDim → dModel]`, and **headDim (128) is not a multiple of the 256-element
Q4_K super-block**, so `CanRepack` is false and all 16 dispatches per layer were stuck on the
weight-stationary kernel. The whole matrix is `[nHeads·headDim → dModel]` = 2048 wide, which *does* repack.
`BlockWeights.WoWhole` was already loaded **zero-copy from the mmap** (and prepacked when a sidecar exists),
so this costs no extra RAM — it only needed the per-head bands concatenated before one dispatch.

| component | before | after | Δ |
|---|---:|---:|---:|
| `attn_out` | 427.7 ms / 576 calls | **109.6 ms / 36 calls** | **−74.4%** (3.9×) |
| `attn_q` *(canary)* | 456.5 ms | 459.1 ms | +0.6% |
| `ffn_gateup` *(canary)* | 1211.0 ms | 1222.7 ms | +1.0% |
| **prefill total** | **3462.3 ms · 194 tok/s** | **3068.5 ms · 219 tok/s** | **−11.4% · 1.13×** |

**Gate on `WoWhole.IsQ4K`, NOT `HasWholeAttnQ4K`.** The latter also demands Q/K/V, and under Q4_K_M `attn_v`
is Q6_K in half the layers — so the four-way gate enabled this in only 18 of 36. The measurement caught it:
`attn_out` reported **306 calls** (18 layers × 16 heads + 18 × 1) instead of 36, and fixing the gate roughly
doubled the win.

Contracting all heads inside one matmul reassociates a sum the per-head path does in head order, so
`useWholeO` also honours `DisableRepackedKernelsForParity` — without that the batched-vs-single-token parity
test can never reach its 1e-2 bound.

#### ✅ MEASURED — biased projections on the tiled kernel: 220 → 249 tok/s (1.13×)

`GemmTiled` gained the optional bias again (it folds into the final store; the no-bias path keeps two
separate store loops so its bit-identity is untouched), and `bias.IsEmpty` came out of the Q4_K tiled gate.

**On its own that changed nothing — `attn_q` moved 459.1 → 463.9 ms, a tie for the second time.** The reason
was not the shape and not the bias: per-head Q/K/V weights are *slices* of the tensor the `*.gguf.repack`
sidecar covers, so `IsPrepacked` is false for them, and `OVERFIT_TILED_PREFILL` was unset — the gate
`(IsPrepacked || UseTiledPrefillQ4K)` failed before `bias.IsEmpty` ever mattered. **The same dead-flag trap
as 2026-07-21. Check that the path is taken before concluding the kernel does not help.**

With `OVERFIT_TILED_PREFILL=1`:

| component | before | after | Δ |
|---|---:|---:|---:|
| `attn_q` | 463.9 ms | **171.3 ms** | **−63%** (2.7×) |
| `attn_kv` | 140.5 ms | **80.3 ms** | −43% |
| `ffn_gateup` *(canary)* | 1208.5 ms | 1214.2 ms | +0.5% |
| **prefill total** | **3056.4 ms · 220 tok/s** | **2699.6 ms · 249 tok/s** | **1.13×** |

Parity green in BOTH configurations: reference path `maxAbsLogitDiff = 0`, fast path agrees on the token.

**Not enabled by default — it costs RAM.** The flag makes `EnsureRepacked()` allocate a heap copy for every
repackable Q4_K weight that the sidecar does not cover, i.e. all ~600 per-head Q/K/V slices (~100 MB on
Qwen-3B). **The zero-RAM version is whole-matrix Q/K/V**: `WqWhole` / `WkWhole` / `WvWhole` are already loaded
zero-copy from the mmap and prepacked by the sidecar, exactly like `WoWhole` — so the same gather/scatter
refactor that landed for O would buy this win without the allocation. That is the next build.

#### ✅ SHIPPED — whole-matrix Q: 249 tok/s at ZERO extra RAM

One `[dModel → nHeads·headDim]` projection replaces 16 per-head ones, then each head gathers its columns
(and adds its own bias, since `BlockWeights` keeps the Q bias per head and there is no concatenated form).

| component | per-head | whole-matrix | Δ |
|---|---:|---:|---:|
| `attn_q` | 463.9 ms / 576 calls | **103.8 ms / 36 calls** | **−78%** (4.5×) |
| **prefill total** | **3056.4 ms · 220 tok/s** | **2695.8 ms · 249 tok/s** | **1.13×** |

**This is the same 249 tok/s the `OVERFIT_TILED_PREFILL=1` experiment produced, without its ~100 MB** —
`WqWhole` is mmap'd zero-copy and covered by the sidecar, so it is prepacked without allocating anything.
It also beats the flag on the component itself (103.8 vs 171.3 ms): one large matmul wins over sixteen small
ones even on the same kernel.

**No parity gate needed here, unlike whole-matrix O.** O contracts over `nHeads·headDim` and therefore
reassociates a sum the per-head path performs in head order; Q's contraction is over `dModel` in both
shapes, so every output element is the same dot product either way.

#### 📖 READ — how llama.cpp's `ggml_gemm_q4_K_8x8_q8_K` differs from ours

`D:\llamacpp-tmp\ggml\src\ggml-cpu\arch\x86\repack.cpp:2042`. Four variants:

| ISA | tile (act rows × out cols) | accumulators |
|---|---|---|
| AVX-512 main | 16 × 16 | `__m512 acc_rows[16]` + `acc_min_rows[16]` = 32 ZMM |
| AVX-512 tail | 4 × 16 | 8 ZMM |
| AVX2 main | 16 × 8 | `__m256 acc_rows[16]` + `[16]` |
| AVX2 tail | 4 × 8 | 8 YMM |
| **ours (`GemmTiled`)** | **`cols` × 8** | **5 `stackalloc` spans of length `cols`** |

Differences, in order of likely cost:

1. **Constant vs runtime accumulator index.** Theirs are `acc_rows[0]`…`[15]` with fully unrolled updates
   (lines 2789-2792, 3464-3467 are four explicit FMAs, not a loop), so the compiler register-allocates and
   spills selectively. Ours are indexed by a runtime `c`, so every access is a stack read/write **and** a
   bounds check — register allocation is impossible, not merely unlucky. Note their AVX2 path declares 32
   `__m256` against 16 YMM, so it spills too and is still fast: the win is *selective* spilling.
2. **Five accumulator arrays to their two.** `accRow`, `accMin`, `iaccB`, `iaccMinB`, `q8s` — 40 vectors of
   stack traffic per iteration at `cols=8`.
3. **Activations are repacked too** (`block_q8_Kx4`, four rows interleaved), so one load feeds four rows.
   That is why their row tile is always a multiple of 4. Ours loads each column separately.
4. AVX-512 is a consequence of (1), not an independent lever: 32 ZMM is what makes the 16×16 tile fit.

#### ✗ ATTEMPTED — unrolled fixed-tile specialisation: INCONCLUSIVE, reverted

A `cols == 4` specialisation with named accumulators was written and passed parity — **but was never
executed**: the dispatcher picks `nr = rows/8 >= cores ? 8 : 4`, which is 8 at 672 rows on 32 cores. That is
the **third** unreached-path mistake in one day (after the dead `OVERFIT_TILED_PREFILL` flag and the
`IsPrepacked` gate hiding the bias change).

Retargeting it to 8 columns by regex-rewriting the existing kernel text produced **incorrect code** —
duplicate unrolled bodies (the generator reported 11 where 8 were expected, and I proceeded anyway), parity
failed at `cols: 8`, and the kernel ran 7-9× slower (70-83× single-threaded). Reverted.

#### ✗ TESTED AND REFUTED — register pressure is not the bottleneck

Register accounting first, since it reframes the task: **AVX2 has 16 YMM registers**, and the kernel keeps
**16 decoded weight vectors** live across the column loop plus 3 hot accumulators per column — 40 vectors
wanted at `cols=8`. Naming the accumulators cannot help, because they have nowhere to go. (This also explains
why llama.cpp's own AVX2 path spills: it declares 32 `__m256`.)

That analysis produced a concrete, small change instead: the low-nibble weight vectors feed only `iacc0` and
the high-nibble ones only `iacc1`, so they are never needed simultaneously. **Splitting the sub-block into
two half-passes over the columns halves peak weight pressure from 16 vectors to 8** — bit-identical (parity
5/5), and the only thing it changes is register lifetime.

**Measured: a tie.** Single-thread is the low-noise signal (StdDev ~1%) and it did not move —
`ffn_gate_up` 167.8 → 168.9 ms, `attn_qo` 31.7 → 32.1 ms, i.e. marginally *worse*. The parallel column showed
`attn_qo` −12.7%, but that sits inside the run-to-run spread of that measurement (5050 / 5217 / 4408 µs
across runs) and the FFN shapes — 71% of prefill — did not move at all. Reverted.

**So spilling is not what costs us.** The remaining structural difference to llama.cpp is the one that
reduces *loads*, not register pressure: `block_q8_Kx4` interleaves four activation rows so one load feeds
four of them, where we issue four `BroadcastLo` per column. That is the next thing to size — and it is a
change to the activation-quantization output layout, not to the kernel's register allocation.

#### ★ LIKE-FOR-LIKE KERNEL COMPARISON — our Q4_K matmul is FASTER than llama.cpp's

Everything above compared whole-model tok/s and *inferred* the kernel difference. That inference was wrong.
llama.cpp's own `test-backend-ops perf -o MUL_MAT` (AVX2 build, 32 threads — it uses
`std::thread::hardware_concurrency`) reports for `q4_K m=4096 k=14336 n=512`, 60.13 GFLOP/run:

| | time | TFLOP/s |
|---|---:|---:|
| llama.cpp | 38 559 µs | **1.56** |
| **Overfit `GemmTiled`** (same shape, 32 workers) | **35 308 µs** | **1.70** |

**Ours is 1.09× faster**, and ~1.91 TFLOP/s with the activation quantization (3 750 µs) excluded.
**So the Q4_K matmul is not where we lose.** Two earlier conclusions are retracted: the "2.34× kernel craft
at equal ISA" attribution, and the register/interleaving hypotheses built on top of it.

**The unexplained part, restated honestly.** Prefill FLOPs are ≈3.72 TFLOP (36 layers; the LM head runs on
the last position only). Ours: 2.695 s = 1.38 TFLOP/s. Theirs (AVX2): 1.996 s = 1.86 TFLOP/s. Our own split:

| | time | FLOPs | TFLOP/s |
|---|---:|---:|---:|
| FFN | 1923 ms | 3.27 T | **1.70** |
| attention projections | 662 ms | 0.45 T | **0.68** |
| other | 110 ms | — | — |

Our FFN already matches the isolated kernel rate. **Attention runs at 0.4× the FFN's efficiency** — that is
where the FLOP throughput collapses, and it is 25% of prefill.

Also unresolved: their production run (`llama-bench`) chose **16 threads** and beat a 32-thread
`test-backend-ops`, so thread count is worth re-sweeping on our side too. The last worker sweep
(8→92, 16→122, 24→130, 32→144 tok/s) predates every optimisation since and may no longer hold at 249 tok/s.

#### ★ MACHINE ROOFLINE — measured, in-repo (`MachineRooflineBenchmark`)

Every kernel figure above was a bare number. These are the ceilings that make them readable
(32 workers, AVX2). Rates are derived by `Helpers/WorkAmount.cs` + `Helpers/ThroughputColumn.cs`,
declared next to each benchmark — *not* in a script, after an out-of-repo script credited a
quantization-only benchmark with the matmul's FLOP count and reported a fictitious 29.6 TFLOP/s.

| ceiling | measured |
|---|---:|
| peak float FMA | **2.19 TFLOP/s** |
| peak int8 dot (`vpmaddubsw`+`vpmaddwd`) | **11.21 TOPS** |
| DRAM read | **89.9 GB/s** |
| copy | 73.0 GB/s |
| STREAM triad | 47.3 GB/s |

**Where our Q4_K GEMM (1.70 TFLOP/s) actually sits:** 15% of the integer ceiling, **78% of the float
ceiling**, and 1% of DRAM bandwidth (33 MB of weights in 35.3 ms = 0.94 GB/s). So the kernel is neither
memory-bound nor integer-issue-bound — **it is bound by the float side of dequantization** (scale
multiplication and int32→float conversion of the accumulators). That is also why llama.cpp's AVX-512
build wins 1.60×: AVX-512 doubles both ceilings.

**Decode, for the first time with a number under it:** Qwen-3B Q4_K (~1.9 GB) at 24.4 tok/s consumes
≈46 GB/s against an 89.9 GB/s read ceiling. The long-standing "decode is at the DRAM floor" conclusion
was previously reasoning only; it now has a measurement.

*Benchmark trap paid for here:* the first version put the accumulator chains in a `stackalloc` span and
measured a float peak of **0.79 TFLOP/s** — below the 1.70 our real matmul achieves, which is impossible
for a loop that touches no memory. The span forced an L1 round-trip per accumulator per iteration.
Constant-index named locals are what keep a value in a register.

#### ▶ NEGATIVE — prefill worker sweep: llama.cpp's 16-thread choice does not transfer

llama.cpp's `llama-bench` picks 16 threads over the machine's 32 and beats a 32-thread
`test-backend-ops`, so our worker count was re-swept at 249 tok/s (the previous sweep predated every
optimisation in this section). More workers still wins for us; there is nothing to take here.

| workers | 8 | 12 | 16 | 24 | 31 | 32 (default) |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 151 | 197 | 218 | 209 | 238 | **246** |

#### ▶ attn_scores — load-balanced query order: real but 6× smaller than predicted

`OverfitParallel.For` splits its range into **contiguous** chunks, but under the causal mask query `i`
attends over `basePos+i+1` keys, so work grows linearly with the index. On a 672-token prefill across
32 workers, worker 0 got rows 0-20 (≈231 dot products) and worker 31 rows 651-671 (≈13 902).
`BatchedAttentionKernel.BalancedQueryIndex` now pairs slot `2k`→query `k` with slot `2k+1`→query
`rows-1-k`, so every consecutive pair costs `rows+1` wherever it lands. Bit-identical (queries are
independent; nothing is reduced across them), zero cost, `OVERFIT_BALANCED_ATTN=0` disables it.

ABAB-interleaved, 3 rounds, best-of-N, untouched FFN as the canary:

| component | baseline | balanced | ratio |
|---|---:|---:|---:|
| **attn_scores** | 275.5 ms | **244.9 ms** | **1.12×** |
| attn_q (canary) | 105.4 | 105.6 | 1.00× |
| ffn_down (canary) | 725.4 | 722.2 | 1.00× |
| total/request | 2793.4 | 2768.5 | 1.01× |

Kept — clean separation across all three rounds, canaries flat. But **the prediction was 1.97× and the
measurement was 1.12×**, so the model behind it was wrong: chunk imbalance is a real cost but not what
dominates this kernel. Worth recording as the correction, because the same "longest chunk sets the
duration" reasoning would misprice the next scheduling change too.

**What the profile actually says about attn_scores.** Per head-layer the causal QK plus softmax·V is
≈115.7 MFLOP; across 16 heads × 36 layers that is **66.6 GFLOP in 244.9 ms = 0.27 TFLOP/s** — **12% of
this machine's 2.19 TFLOP/s float ceiling**, and 6× below our own Q4_K GEMM. Keys per head are 344 KB,
so this fits L2 and is not bandwidth-bound. The kernel itself (`CachedAttentionKernel.ComputeSingleHead`,
reached one query at a time) is the open question — that, and the float side of Q4_K dequantization, are
the two measured candidates left.

#### ★ AVX-512 GO/NO-GO GATE — PASSED, the port is worth writing

Our Q4_K GEMM sits at 78% of the 256-bit float ceiling and FFN is 69% of prefill, so the only way to move
the dominant cost is to raise the ceiling. Before writing any kernel, `MachineRooflineBenchmark` was
extended with `Vector512` variants to check whether this silicon actually delivers the wider ceiling.
This was a real risk: Zen 4 double-pumps 512-bit ops through a 256-bit datapath (~1.1×) and many Intel
parts drop clocks under 512-bit load, either of which would have killed the plan.

Box: **AMD Ryzen 9 9950X3D** (Zen 5), 16 physical / 32 logical, AVX-512 F+BW+CD+DQ+VL + VNNI + VBMI + IFMA.

| ceiling | 256-bit | 512-bit | gain |
|---|---:|---:|---:|
| float FMA | 2.20 TFLOP/s | **4.15** | **1.89×** |
| int8 dot | 11.16 TOPS | **22.87** | **2.05×** |

Zen 5 has the full 512-bit datapath and the measurement shows it — essentially the theoretical 2×, with no
visible clock penalty. **Our 1.70 TFLOP/s GEMM is 78% of the 256-bit ceiling but only 41% of the 512-bit
one.** If the port preserves utilisation, FFN 1923 ms → ~1000 ms and prefill 2769 → ~1850 ms ≈ **373 tok/s**,
which would be *above* llama.cpp's AVX2 build (336.7) and 1.45× off their AVX-512 (541.7).

Two notes. **AVX-512 VNNI is present**: `vpdpbusd` collapses the `vpmaddubsw`+`vpmaddwd`+`add` triple our
kernel issues into one instruction — a second-order lever here since the kernel is float-bound, but it is
what llama.cpp uses. And the existing **"AVX-512 decode port" negative does not transfer**: decode is
memory-bound (measured today at ~46 GB/s against an 89.9 GB/s ceiling), where wider vectors buy nothing;
prefill is compute-bound and pinned against the float ceiling.

Next: port `Q4KGemvKernel.GemmTiled` (ffn_gate_up, attn_q/o), then `Q6KGemvKernel.GemmTiled` (ffn_down) —
parity test first, then measure. Dispatch at run time through `CpuFeatures.HasAvx512`, never at compile
time: `Cli.csproj` pins `IlcInstructionSet=avx2` and the AOT build must keep running on machines without it.

#### ★ RETRACTION + the kernel's real ceiling

**Retracted: "our Q4_K GEMM runs at 78% of the float ceiling."** That divided *logical* MACs by the
*floating-point instruction* ceiling, but the kernel performs one `vpmaddubsw` per **32** MACs and issues
only ~6 float ops per column per block against ~160 integer/shuffle ops. Against the ceiling that actually
applies it sits at **15% of 11.2 TOPS**, not 78%. The AVX-512 recommendation survives the correction, but
its stated reason ("the ceiling is too low") was wrong — the real problem is instructions issued per MAC.

`MachineRooflineBenchmark` now measures the ceiling **for this kernel's instruction mix** — the eight-
statement `iacc0` block verbatim, `Blend` + two lane shuffles + `vpmaddubsw` + `Add` — rather than an
idealised dot chain:

| | TFLOP/s |
|---|---:|
| idealised int8 chain (3 instructions / 32 MACs) | 11.2 |
| **kernel's instruction mix, 4 live accumulators** | **4.63** |
| same mix, 512-bit | 7.71 |
| **our real GEMM** | **1.70** |

So the shuffles cost 2.4× against the idealised chain, and we then reach only **37% of our own mix's
ceiling**. That residual 2.7× is not arithmetic — it is loads, weight decode, the float tail, and spills.

**NEGATIVE — register pressure is not the explanation.** The suspicion was that `GemmTiled`'s `stackalloc`
accumulator spans spill: at four columns it holds `accRow`+`accMin` (8 vectors, live across the block loop),
`iaccB`+`iaccMinB` (8, across the sub-block loop) and `iacc0`+`iacc1` (8) — 24 vectors before a single
weight, against 16 ymm registers. Probing it with `IntegerDotChains`' body at 12 vs 16 chains (12+3
constants fit ymm, 16+3 do not; both fit 512-bit's 32 zmm):

| live accumulators | 256-bit | 512-bit |
|---|---:|---:|
| 12 (fits ymm) | 11.50 | 23.52 |
| 16 (exceeds ymm) | **14.99** | 24.19 |

Sixteen chains are **30% faster** at 256-bit, not slower — more independent chains cover latency better and
any spill hides behind the surrounding work. **There is no register cliff, and the "de-spill first" plan is
dropped.**

An earlier version of this probe reported the opposite (−41% at 256-bit, and 512-bit falling *harder* than
256-bit, which cannot be true if the larger file helps at all). It routed each step through a helper taking
five vector parameters; once the statements were written inline the effect vanished entirely. The tell was
the impossible 512-bit ordering — treat that shape of result as a broken benchmark, not a discovery.

**Still unexplained: 1.70 actual vs 4.64 for its own instruction mix, a 2.7× residual.** The mix benchmark
models only the eight `iacc0` statements. Ablating the three pieces it omits, inside the real kernel
(`Q4KGemvKernel.Ablate*` — measurement-only toggles, default off), on `ffn_gate_up` at 672 rows:

| ablation | mean | vs `Tiled` |
|---|---:|---:|
| `Tiled` (baseline) | 15.567 ms | — |
| no F16 scale/min decode | 13.701 ms | **−12.0%** |
| no scalar `Unpack` of 6-bit scales | 15.016 ms | −3.5% |
| no nibble `And`/shift | 15.679 ms | +0.7% (tie) |

Error bars are ±1.0–1.1 ms on ~15 ms (≈7%), so only the F16 result clears the noise, and barely; the other
two sit inside it. **Together they bound at ~15% and do not explain a 2.7× residual (≈63% of runtime).**

The one thing the mix benchmark did not model at all is **memory traffic** — it ran on register constants.
The real kernel issues 8×32 B weight loads per sub-block (12.7 MB streamed per projection), per-column
activation loads, and span-backed accumulator accesses. That is the remaining suspect, and it is untested.

**NEGATIVE — the F16 decode's 12% is the scalar conversions, not the memory round-trip.**
`LoadF16x8Rearrange` used to store its shuffled vector to `stackalloc` and immediately re-read it as eight
`ushort`s — textbook store-to-load forwarding stall. Extracting the lanes from the register with `pextrw`
instead measured **15,269 µs vs 15,567 µs, i.e. −1.9% against ±7% noise: a tie**, with the ablation floor
unchanged at 11.9%. The round-trip was free; the eight scalar `Half`→`float` conversions are the cost.

Capturing it therefore means *eliminating* the conversions, not speeding them up: store the scales as **f32 at
repack time**. `block_q4_Kx8` is our own layout, so this is available — +32 B on a 1152 B block (**+2.8%
weight RAM for 12%**), at the price of a `.gguf.repack` sidecar format change. Note x86 could do this in one
`vcvtph2ps`, but .NET exposes neither an `F16C` intrinsic class nor a `Half` overload of `Vector128.Widen`.

#### ★★ THE WEIGHT STREAM IS READ 84 TIMES PER PROJECTION — no cache blocking exists

Applying the standard model (Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication* — the
GotoBLAS/BLIS scheme, where block sizes are derived from cache sizes: an `mr×nr` tile of C in registers, a
`kc×nr` panel of B in L1, an `mc×kc` block of A in L2, a `kc×n` panel of B in L3) exposes what the profiling
missed all day:

`GemmTiled` receives **8 columns** and walks the **entire** weight matrix. The dispatcher splits 672 rows
into tiles of 8, so **84 tiles each stream all 12.68 MB** of `ffn_gate_up`'s weights:

    84 × 12.68 MB = 1.07 GB per projection, in 15.27 ms = ~70 GB/s
    measured DRAM read ceiling = 90 GB/s

Our tiling is a *register* tile (`MaxTileCols`) only — there is **no L2/L3 blocking level at all**. This is a
candidate for the whole remaining 1.56× residual, and unlike everything else on the list it is a structural
fix with a textbook algorithm behind it.

**Measured — the traffic argument holds, then breaks on parallel granularity.** `ffn_gate_up`, two runs each,
agreeing on ordering:

| tile | passes | 672 rows | 1024 rows |
|---|---:|---:|---:|
| NR=4 | rows/4 | 17.8 ms · 1.70 | 27–29 ms · 1.66 |
| NR=8 | rows/8 | **15.2–15.9 ms · 1.95** | 23.7–24.5 ms · 1.92 |
| NR=16 | rows/16 | 16.5–17.0 ms · 1.80 | **22.7–23.0 ms · 2.02** |

Halving the traffic 4→8 buys **+17%** exactly as predicted. Halving it again 8→16 *loses* **8%** at 672 rows,
because 42 tiles over 32 cores leaves ten workers with two and twenty-two with one — a 1.52× imbalance
against 1.14× at NR=8. At 1024 rows there are enough tiles again and the wider tile wins by 4–7%.

So `ResolveTileCols` now takes the widest tile that still gives **~2 tiles per core**, not one. That
reproduces NR=8 at 672 (no change to the profiled prompt, prefill stays 249 tok/s) and switches to NR=16 from
~1024 rows — a real gain for long prompts. The ≥1024 branch was measured rather than reasoned, because six
mechanism hypotheses were refuted the same day.

#### ▶ NEGATIVE — output-row banding (the "missing L2 blocking level") is 20% SLOWER

Implemented as `BatchedQuantProjection.UseOutputBlocking` (default **off**, kept as the record): parallelise
over bands of output groups instead of column tiles, so each worker owns an L2-sized slice of the weight
matrix and re-reads it from its own cache for every column tile. `Q4KGemvKernel.GemmTiled` gained
`groupStart`/`groupCount` for this. Two runs, agreeing:

| arm | run 1 | run 2 | TFLOP/s |
|---|---:|---:|---:|
| **Cols8 (today's production)** | **14.91 ms** | **15.06** | **2.02** |
| Banded16 | 16.72 | 16.04 | 1.85 |
| Cols16 | 17.10 | 16.89 | 1.78 |
| **Banded8** | **18.21** | **17.81** | **1.68** |

**And this refutes the traffic story that motivated it.** Banding removes 84× of the weight re-reads; if that
traffic were the constraint it had to show. It did not — this chip's 128 MB L3 (V-cache) holds the whole
12.7 MB matrix, so those re-reads were never going to DRAM in the first place.

**Unified explanation that fits every measurement taken today.** The per-block fixed work — F16 scale decode
(ablated at **12%**), scalar `Unpack` (**3.5%**), nibble unpack (~0%) — is amortised across the columns in a
tile. At NR=8 that is ~15% of runtime; at NR=4, ~30%; at NR=16, ~7.5%. Predicted 4→8 gain
`1.30/1.15 = 1.13×` against **1.17× measured**; predicted 8→16 gain `1.15/1.075 = 1.07×`, overwhelmed by the
1.52×/1.14× imbalance shift, against **−8% measured**. No bandwidth term is needed anywhere.

**So the lever is to delete the fixed work, not to move the data.**

#### ★ WIN — hoisting the F16 scale decode: prefill 249 → 256 tok/s

`GemmTiled` decodes each block's F16 scale/min pair inline, which reads as amortised — but the kernel runs
**once per column tile**, 84 times at 672 rows and NR=8, so every pair is widened 84 times over.
`Q4KGemvKernel.DecodeBlockScales` now widens them once per projection into a pooled scratch
(`BatchedQuantProjection.UsePrecomputedScales`), which the tiles share. Bit-identical — same conversions,
fewer of them — and it needs **no format change and no extra weight RAM**, unlike storing f32 in
`block_q4_Kx8` (+2.8% permanently, and every `.gguf.repack` sidecar invalidated).

| arm (672 rows, `ffn_gate_up`, two runs) | run 1 | run 2 | TFLOP/s |
|---|---:|---:|---:|
| **hoisted, NR=8** | **13.88 ms** | **13.60** | **2.21** |
| ablation floor (decode removed entirely) | 14.45 | 14.31 | 2.11 |
| NR=8 baseline | 15.91 | 15.21 | 1.95 |
| hoisted, NR=16 | 16.66 | 17.05 | 1.80 |
| NR=16 baseline | 18.01 | 17.87 | 1.69 |

**+13% at NR=8** — faster than the ablation floor, because ablation still built a constant vector and took the
branch, so the full 12% was recovered and a little more.

**The amortisation theory predicted this before it was measured, twice over.** Fixed per-block work is ~15% of
runtime at NR=8 and ~7.5% at NR=16, so the gain should roughly halve with the wider tile: predicted 2.0×,
measured 13%/6% = 2.2×. End to end it predicted `0.45 × 0.13 = 5.9%` off prefill → 2606 ms; measured
**2622–2632 ms, 255–256 tok/s**, within 0.6%. Gap to llama.cpp's AVX-512 build: 2.18× → **2.12×**.

**The same hoist for Q6_K pays 5× less than predicted.** `Q6KGemvKernel.DecodeBlockScales` mirrors the Q4_K
one and carries `ffn_down` (27% of prefill). Predicted ~13% and another ~9 tok/s; measured **ffn_down 708.6 →
690.4 ms, −2.6%**, worth 3 tok/s. The reason was checkable in advance and was not checked: Q6_K widens
**eight** values per block against Q4_K's sixteen (it has no `dmin`), and its block is larger — 1680 B vs
1152 B — with more compute in the 6-bit unpack. The fixed decode is therefore a much smaller fraction of a
bigger block: 12% / ~4.6 ≈ 2.6%, which is what came out. The amortisation model predicts well *within* a
kernel — it called the tile-width scaling and the end-to-end figure correctly — but extrapolating it *across*
kernels without re-reading their inputs was a guess.

**Both hoists together: prefill 249 → 258–259 tok/s, gap to llama.cpp's AVX-512 build 2.18× → 2.10×.**
Suite 1486/0/229.

#### ★★ AVX-512 Q4_K PREFILL KERNEL — prefill 259 → 280 tok/s, gap under 2× for the first time

`Q4KGemvKernel.GemmTiled512` processes **two activation columns per instruction**: column `2p` in the low 256
bits of every vector, `2p+1` in the high. Weights are identical for both, so they are broadcast into both
halves; only activations, their scales and their block sums differ. Every shuffle in this kernel is
per-128-bit-lane, so it widens without changing meaning — no new repack layout, `block_q4_Kx8` untouched,
sidecars still valid.

Two decisions worth keeping: pairing **columns** rather than widening the output-row group avoids a
`block_q4_Kx16` layout and the sidecar invalidation that implies; and the pair loop stays **innermost**,
because hoisting it would re-decode the sixteen weight vectors per pair and throw away the amortisation the
tile-width sweep showed to dominate this kernel.

| component | before | after | |
|---|---:|---:|---:|
| `ffn_gateup` | 1180 ms | **1017.6** | **−13.8%** |
| `attn_out` | 106.0 | **89.4** | −15.7% |
| `attn_q` | 100.6 | **85.6** | −14.9% |
| `ffn_down` (Q6_K, not ported) | 691 | 658 | −4.8% |
| `attn_scores` (different kernel) | 207.6 | 210.8 | flat |
| **prefill** | **2597 ms / 259 tok/s** | **2398 / 280 tok/s** | **+8.2%** |

`attn_scores` staying flat while every Q4_K path moves 14–16% is the internal control: this is the change,
not box drift. **Gap to llama.cpp AVX-512 2.10× → 1.93×; to their AVX2 build 1.20×.**

The kernel itself gained ~1.16×, not the 1.67× its instruction mix promised, because that mix is roughly half
the kernel — loads, the scalar `Unpack` and the stores did not widen. The prior estimate was 1.24×.

**Bit-identical**, pinned by `Avx512PrefillParityTests` (8 cases: odd and even column counts, bias, and the
pre-decoded scale path) asserting exact equality rather than a tolerance. Gated through
`CpuFeatures.HasAvx512`/`HasAvx512Bw` — the repo's own OVERFIT015 analyzer rejected a direct `IsSupported`
check, which is what that rule is for. Suite 1494/0/229.

#### ★★ Q6_K AVX-512, SECOND ATTEMPT — pair what is already adjacent: +12% on `ffn_down`

The failure below was diagnosed as broadcast traffic, not vector width, and that diagnosis held. `ql03`/`ql47`
are stored adjacently (`k*64` and `k*64+32`), as are `qhL03`/`qhL47` — so **one 512-bit load carries real data
in both halves and no weight broadcast is needed at all**. Activations come from a single `vpbroadcastq`
(`Vector512.Create(long)` replicates the 8-byte pattern), which is exactly the tiling the 256-bit path built
by hand from two Create calls. The only cross-half move left is one `GetUpper` per reduction, unavoidable
since AVX-512 has no `vphaddd` for zmm. Accumulators stay 256-bit, so register pressure is unchanged.

ABAB-interleaved, three rounds:

| component | 256-bit | 512-bit | |
|---|---:|---:|---:|
| **ffn_down** | 659.7 ms | **589.2** | **1.12×** |
| ffn_gateup / attn_scores / attn_kv / attn_q / attn_out (canaries) | 1017.1 / 187.7 / 138.0 / 84.9 / 89.2 | 1011.3 / 189.0 / 139.0 / 84.7 / 89.0 | 0.99–1.01× |
| total | 2373.6 | 2298.6 | **1.03×** |

**Prefill 283 → 292 tok/s, gap 1.91× → 1.85×.** Bit-identical, `Avx512Q6KPrefillParityTests` 7/7, suite
1501/0/229. Same kernel, same instruction set, same shape — **only the choice of what shares a register**
turned −20% into +12%.

#### ▶ NEGATIVE (superseded above) — column-pairing the Q6_K port is SLOWER

`Q6KGemvKernel.GemmTiled512` exists and is bit-identical (`Avx512Q6KPrefillParityTests`, 7 cases), but
`BatchedQuantProjection.UseAvx512PrefillQ6K` is **off**: on the same machine and prompt where the Q4_K port
took `ffn_gateup` down 13.8%, this took `ffn_down` from 658 ms to **794–900 ms** and prefill from 280 back to
256–265 tok/s. Reverting restored 2398.5/2399.6 ms and `ffn_down` 656/659 ms exactly.

Two things marked it as real rather than drift: the Q4_K components held steady across the same runs
(`ffn_gateup` 1023/1009, `attn_q` 86/84), and the run-to-run spread was concentrated entirely on `ffn_down`.

**Why the identical technique inverts between the two kernels.** Column pairing pays for the
`vinserti64x4` that builds each broadcast with the arithmetic subsequently done on it. Q4_K broadcasts eight
weight vectors per sub-block and then issues sixteen paired statements against them. Q6_K broadcasts six per
`k`, sixteen times per block, for far less arithmetic each — and its `ReduceRows` cannot widen at all, since
AVX-512 has no `vphaddd` for zmm, adding three more cross-half moves per call across 32 calls per block. The
lane-crossing traffic outruns the arithmetic saved. **A wider vector is not a property of the ISA alone; it
is a ratio between broadcast cost and work done per broadcast, and that ratio is per-kernel.**

#### ★ attn_scores — register-resident value accumulation: −13% on the component

The softmax-weighted value sum walked every `d` for each `t`, so it loaded **and stored** the whole output
accumulator once per `t`: 512 B of value read against 512 B of accumulator read plus 512 B written — two
thirds of the traffic was the accumulator round-tripping through L1, ~231 MB of ~347 MB per head-layer.
`AccumulateValuesBlocked` blocks `d` into 64 dimensions so eight accumulators stay in registers across the
whole `t` loop; the value stream is unchanged in volume, just read in two passes. Bit-identical — ascending
`t` order per `d` preserved, and the deliberate no-FMA property kept.

ABAB-interleaved, three rounds, best-of-N:

| component | baseline | blocked | |
|---|---:|---:|---:|
| **attn_scores** | 213.8 ms | **186.3** | **1.15×** |
| attn_kv / attn_q / attn_out (canaries) | 137.4 / 84.6 / 88.5 | 137.9 / 85.5 / 89.2 | 1.00 / 0.99 / 0.99× |
| ffn_gateup / ffn_down (canaries) | 1006.8 / 659.2 | 1012.8 / 662.6 | 0.99× |
| total | 2385.7 | 2366.6 | 1.01× |

**End to end this is only +0.8%**, because attn_scores is 8% of prefill. A first single-arm run appeared to
show 280 → 292 tok/s, but `attn_kv` and `ffn_gateup` — neither touched by the change — moved with it, so that
reading was box drift and is withdrawn. Interleaving the arms with those components as canaries is what
separated the two. **Prefill stands at ~283 tok/s.**

**On the .NET-vs-C++ gaps this work exposed.** Three are real: no `F16C` intrinsic class (nor a `Half`
overload of `Vector128.Widen`), no first-class AVX-512 mask registers, and no `restrict`. All are
dotnet/runtime JIT work, not something a library can supply — F16C in particular is a well-scoped ask with an
existing pattern to follow. `TensorPrimitives` is the right home for the subset expressible as *bulk*
buffer-to-buffer work, and does carry hardware paths not otherwise reachable; it did not fit here because the
values are eight at a time, interleaved every 1152 bytes inside a hot loop. The deeper difference is
optimisation budget — RyuJIT is a fast JIT, and Native AOT uses the same backend, so there is no LLVM-class
scheduling to reach for. **None of this explains the remaining gap**: the Q4_K matmul measured faster than
llama.cpp's at equal ISA and thread count. What is left is AVX-512 coverage and our own kernel structure.

#### ▶ NEGATIVE — the unaccounted time holds no surprise; it is spread thin

A claimed "~192 ms unaccounted" was an arithmetic error: it conflated time outside both blocks with time
inside attention that no sub-slice covers. The profiler's own top-level rows split it properly:

| | time | share |
|---|---:|---:|
| **attention** (top level) | 585.2 ms | 24.6% |
| — sub-slices (`kv`+`q`+`scores`+`out`) | 504.0 | |
| — **unattributed inside attention** (RoPE, QK-norm, the whole-matrix Q/O gather+scatter) | **81.2** | **3.4%** |
| **ffn** (top level) | 1683.3 ms | 70.8% |
| — sub-slices (`gateup`+`down`) | 1683.1 | |
| — unattributed | **0.2** | **0%** |
| **other** (norms / residual / embed / final norm) | 109.4 | 4.6% |

**FFN is 100% accounted**, which kills the hypothesis that SwiGLU's 266M `silu` calls were a hidden cost —
the activation lives inside `ffn_gateup` and is not separable at this granularity. The residual is genuinely
thin: halving *both* remaining pieces would buy ~4%. The work is in the large kernels, not hiding beside them.

**What F16C would be worth now, if .NET exposed it: ~0.1%.** Ablation priced the F16 decode at 12% of the
Q4_K kernel, but hoisting already removed 83/84 of that work by decoding once per projection instead of once
per column tile. The missing instruction would speed up what remains; the restructuring deleted it. Worth
recording as the general shape: **a workaround that removes work beats an instruction that accelerates it**,
and having the instruction available would likely have stopped the search at 12%. Where F16C would still pay
is *model loading* — `GgufReader`, `GgmlDequant`, `SafetensorsReader` and the Whisper loader all widen halves
in scalar loops, hundreds of millions of values per 3B model — but that is startup, not inference.

#### ★ whole-matrix K/V — one dispatch per projection: prefill 291 → 297 tok/s

Per-group K/V projects `[dModel → headDim] = [2048 → 128]`, which a micro-bench put at **0.37 TFLOP/s**:
352 MFLOP is too little work to amortise the dispatch's fixed cost, and single-thread was only 1.9× slower
than the pool, so the 84-tile launch — not the matmul — dominates. Ceiling measured before building: two
narrow dispatches 3206 µs vs one wide `[2048 → 256]` 1768 µs = **1.81×** on the projection. Built it:
project all KV heads through `WkWhole`/`WvWhole` once, gather each group's band (adding the per-KV-head bias
in the copy). Gated on both whole handles being Q4_K, so Q6_K `attn_v` layers fall back to per-group.

ABAB, canaries flat: **attn_kv 139.1 → 96.4 ms (1.44×)**. End-to-end, six paired rounds:

| | median | min |
|---|---:|---:|
| per-group | 2311 ms / 291 tok/s | 2298 / 292 |
| **whole** | **2262 / 297** | **2251 / 299** |

**+2.2% e2e**, matching the 1.81×-on-projection ceiling. Bit-identical (each output row's dot product is
unchanged; no reassociation), no parity gate, suite 1501/0/229, `OVERFIT_WHOLE_KV=0` disables.

*Methodology note kept as a warning:* the first component table read total as 1.00× while attn_kv clearly
dropped — an artifact of taking each component's min from a different run, so total-min and attn_kv-min came
from different rounds. A paired total-only measurement resolved it. **Best-of-N per component does not give a
consistent end-to-end number; measure total paired.**

#### ▶ attn_scores, split by ablation — and why flash-attention is the WRONG lever

Before writing a blocked kernel, ablation inside the real kernel split the 189.4 ms three ways:

| removed | attn_scores | share |
|---|---:|---:|
| Q·Kᵀ dot | 137.7 ms | **27%** |
| softmax exp | 128.0 ms | **32%** |
| both | 60.7 ms | rest **32%** |

**Neither dominates, and exp is the larger of the two.** A flash-attention rewrite only attacks the dot
(27% of the component = 2.3% of prefill) — the most expensive, highest-risk change aimed at the smaller
piece. Dropped. The exp is the better target and is a *bulk contiguous buffer*, exactly the shape
`TensorPrimitives` serves — the same lever SwiGLU already took (`ApplySiLU` → `TensorPrimitives.Sigmoid`).

Replacing the scalar `MathF.Exp` loop with `TensorPrimitives.Subtract`/`Exp`/`Sum`: **attn_scores 189.4 →
173.2 ms (1.09×)**, e2e 297 → 299 tok/s (vector won all six paired rounds). Smaller than the 32% ablation
because `TensorPrimitives.Exp` is not free and the fused scalar loop became three passes over a short buffer;
exp itself went ~61 → ~45 ms. Not byte-parity vs the F32 reference (few-ULP, coherence-safe like SwiGLU), but
prefill and decode both reach this method so they stay bit-identical to each other — parity suite green,
1501/0/229. `OVERFIT_ATTN_VEXP=0` disables.

**attn_scores is now optimised across all three parts** (value sum register-resident, exp vectorised, the dot
is what remains). Further gains need the flash-GEMM for the dot — ~2% e2e at high risk, not worth it now.

#### ★ DECODE IS MEMORY-BOUND — measured directly, AVX-512 cannot help

`DecodeGemvRooflineBenchmark` runs the production decode GEMV (`GemvParallel`, AVX2) on a Q4_K FFN weight at
two sizes — one that fits this box's 128 MB L3, one that does not — to separate the kernel's compute rate
from the memory rate it is fed:

| weight | source | GB/s |
|---|---|---:|
| 12.7 MB (fits L3) | hot cache | **132.5** |
| 203 MB (exceeds L3) | DRAM | **73.5** |
| DRAM read ceiling | — | ~90 |

**The kernel's compute (132.5 GB/s hot) is well above the DRAM ceiling (90)**, so the dequant consumes bytes
faster than DRAM delivers them: decode is not compute-bound, and an AVX-512 / VNNI decode kernel cannot help.
This is the direct measurement behind the earlier reverted "AVX-512 decode port" negative. Streaming from
DRAM the GEMV hits **73.5 GB/s = 82% of the ceiling** — the kernel itself is near-optimal.

The whole-model decode figure (~46 GB/s) is well below the isolated GEMV's 73.5, so that shortfall is **not**
the weight kernel — it is per-token overhead (attention over the growing KV cache, RoPE, norms, sampling) and
the serial layer→layer dependency that leaves memory idle between GEMVs. That is an overlap/latency problem,
not a compute one, and we are already at 1.13× of llama.cpp there. **Decode's compute levers are exhausted,
by measurement.**

**Remaining measured item:** the scalar `Unpack` at ~3.5% of the Q4_K kernel.

*Invalidated run, kept as a warning:* the first tile sweep ran inside an 11-benchmark class and reported
`Tiled` and `Tiled_Cols8` — **the same configuration** — 21% apart, far outside their ±9% bars. Two identical
arms in one table is the cheapest canary there is; narrowing the filter so the arms sit adjacent in time made
the result reproducible.

#### ▶ WHAT IS LEFT — profile at 249 tok/s, gap 2.18×

```
ffn_gateup  1222.7 ms  39.8%   (36)   <- Q4_K tiled already
ffn_down     720.1 ms  23.5%   (36)   <- Q6_K tiled already
attn_q       459.1 ms  15.0%  (576)   <- weight-stationary: blocked by `bias.IsEmpty`
attn_scores  251.1 ms   8.2%  (576)
attn_kv      140.3 ms   4.6%   (72)
attn_out     109.6 ms   3.6%   (36)   <- done
other        105.1 ms   3.4%
```

**The structural waste is spent.** Every remaining component is already on the best kernel Overfit has, with
two exceptions:

1. **`attn_q` — 15.0%, and it is blocked by one gate, not by shape.** Per-head Q is `[2048 → 128]`:
   `inputSize % 256 == 0` ✓ and `outputSize % 8 == 0` ✓, so **`CanRepack` is TRUE** — the only thing keeping
   it off the tiled kernel is `bias.IsEmpty` (Qwen puts a bias on Q/K/V). Micro-bench for that shape class:
   tiled 4.41 ms vs weight-stationary 12.95 ms; measured `attn_q` is 12.75 ms/layer. **Ceiling ≈ 300 ms of
   3068 ≈ 9.8% → ~243 tok/s.**
   Bias support in `GemmTiled` was built once and reverted on a measured **0.999× tie** — but that tie was
   taken when the biased projections were ~6% of FLOPs and the FFN dwarfed them. The composition has changed;
   **re-measure before rebuilding, and re-measure with the FLOP-weighted census, not the dispatch count.**

2. **`attn_scores` — 8.2%, never examined.** `BatchedAttentionKernel.ComputeParallel` has had no profiling
   pass at all.

**Everything else is kernel quality, i.e. writing better SIMD.** The measured headline: llama.cpp built
AVX2-only does 336.7 tok/s against our 219 — so **1.54× of the remaining 2.47× is pure kernel craft at equal
instruction set**, and AVX-512 accounts for the other 1.60×. Both are intrinsics work on `GemmTiled`
(register-blocking the accumulators, 512-bit lanes), not structural fixes. Expect weeks, not evenings, and
size each step against its share before building.

#### ✅ RESOLVED — `BatchedPrefillParityTests` (was failing since before this work)

`BatchedPrefill_MatchesSingleToken_OnRealQwen` asserts `maxAbsLogitDiff == 0` between batched prefill and the
single-token path. It now reports `argmax batched=11 single=13, maxAbsLogitDiff ≈ 0.44`.

**Not caused by the changes above.** Disabling *both* repacked paths (Q6_K tiled off AND the `IsPrepacked`
short-circuit removed from the Q4_K gate) makes it pass 5/5 — with the shared quantization still enabled,
which also proves that change is bit-identical. The trigger is the `*.gguf.repack` sidecar created
2026-07-20: it sets `IsPrepacked`, routing bias-free Q4_K projections through the repacked `GemmTiled`, whose
reduction is associated differently. The Q6_K tiled kernel is the same class of change and breaks it
independently.

**Nobody noticed because the test is `[LongFact]`** — skipped by default, so a numerics regression sat
unobserved for two days. Decision needed: either make the test explicitly disable the repacked paths (so it
keeps testing the batched-vs-single-token *math* it claims to), or replace the exact-equality gate with a
coherence check, as `OVERFIT_REPACK_ATTN` already is. Do not silently relax it.

**Plan.** Q4_K and Q6_K share the Q8_K scratch format bit-for-bit (`SuperBlockElements 256`, `GroupSize 16`,
and `Q6KDotKernel.QuantizeActivationQ8K` delegates to Q4_K's), so ONE pre-quantized buffer serves Q, K and V
regardless of whether V is Q4_K or Q6_K. Steps: (1) add `bool preQuantized = false` to the three batched
kernel entry points — `Q4KDotKernel.ProjectBatched` / `ProjectBatchedWeightStationary`,
`Q6KDotKernel.ProjectBatched` — guarding their internal quantize loop (anchor: the
`"Activation quantization scratch is too small for rows."` validation, which occurs exactly at those three);
(2) give `BatchedQuantProjection.Dispatch` optional pre-quantized scratch spans, defaulting to today's
pooled-and-quantize behaviour; (3) quantize `hidden` once at the top of
`CachedMultiHeadAttention.DecodeBatchedQuant` and pass it to the Q/K/V dispatches. Output must stay
bit-identical — quantization is deterministic, so this is a pure de-duplication.

At 3.4 B params × 672 tokens the gap is ≈3.7 TFLOP/s-equivalent for them against ≈1.0 for us.

**Why this lever is different from the five that were refuted:** it has a measured ceiling, a named cause, and
a working reference implementation to read. The earlier register-/cache-blocking negatives were on
*memory-bound* paths, where blocking cannot help by construction. Prefill is compute-bound.
**Honest expectation: 3.76× is the ceiling, not a promise — 2× would be a good outcome.** Size a single
projection with a micro-bench against `sgemm.cpp` before writing any kernel.

---

### Refuted levers — do not re-open without new evidence

**Three candidates were sized and all three died on measurement (2026-07-21).**

1. **`SearchValues` / tokenizer-level work — CLOSED.** A prefill profile (Qwen-3B Q4_K_M, 672-token prompt,
   median of 5) puts tokenization at **0.04% of time-to-first-token** — 1.8 ms against 4731 ms of prefill
   forward (366 000 tok/s vs 142 tok/s). Infinite tokenizer speedup buys 0.04%.
   `Tests/LanguageModels/Diagnostics/PrefillProfileTests.cs`.
2. **Struct-operator (static-abstract interface) dispatch — CLOSED without building.** The premise does not
   hold here: `ElementwiseKernels` contains **no delegates** (14 hand-written span loops), the hot parallel
   paths already use `delegate*<int,int,void*,void>`, and the whole elementwise slice is **0.5% of decode**.
   The scalar operator shape also cannot express the `TensorPrimitives` fast path, which is itself built on
   this pattern inside the BCL.
3. **Bias support in the Q4_K tiled prefill GEMM — BUILT, MEASURED 0.999×, REVERTED.** A path census showed
   `bias.IsEmpty` barred **88% of Q4_K prefill dispatches** (all attention Q/K/V) from `GemmTiled`. Lifting
   it was an exact tie, because `ProjectBatchedWeightStationary` already amortises weight decode across the
   row tile — the same thing the tiling does. The "~3×" in the kernel docs is measured against
   `ProjectBatched` (re-decode per row), **not** against weight-stationary. Recorded in `CLAUDE.md`.

Decode is ~88% quantized GEMV sitting at the DRAM floor (`ffn 69.3% · attention 19.3% · lm_head 10.3%`), so
there is no cheap **decode** kernel win left — this was later confirmed directly by `DecodeGemvRooflineBenchmark`
(kernel compute above the DRAM ceiling; see the closing summary at the top of this track).
The product direction (perf course vs. the on-prem commercial track) remains deferred and is a separate,
non-technical decision.

</details>

---

## (Historical) Session resume point 2026-05-22 → 23 — SUPERSEDED by "Nearest plan (2026-05-25)" above

**Big session — zero-Python loading completed, chat turnkey, and the anomaly+LoRA product
track closed with an empirically-corrected verdict.** Strategic frame (still holds): NOT chasing
llama.cpp on decode; embeddability / training / product moat. See [[project-loading-story]],
[[project-loading-direction]], [[project-chat-runtime]], [[project-anomaly-lora]], [[project-perf-sprint]].

**Zero-Python loading — DONE (all inbound formats native; one-directional, NO exporters — see
[[project-loading-direction]]):**
- **Native Llama/Qwen safetensors loader** `SafetensorsLlamaLoader.Load(dir, quantize)` →
  `CachedLlamaInferenceEngine` + `LlamaConfigReader` (config.json via `Utf8JsonReader`). **VALIDATED
  coherent on real Qwen2.5-0.5B** ("The capital of France is" → " Paris..."). Found+fixed a **RoPE
  row-permute** bug (HF rotate-half → GGUF adjacent-pair on q/k weights+biases; `RopeKernel` is NEOX/
  adjacent-pair so HF weights need the llama.cpp permute). NOTE: `Scripts/convert_llama.py` has the SAME
  bug (unpermuted) — its `.bin` for RoPE models is suspect; not fixed (no Python here).
- **GPT-2 loader peak-RAM 2×→~1×** — `SequentialChunkReadStream` streams one param block at a time into
  `GPT1Model.Load` (bounded backward seek for the MHA legacy-peek); no full in-RAM `.bin` copy.
- Parity test (`SafetensorsLlamaLoaderTests`) bit-identical vs `.bin` (GQA + permute); the loader-vs-.bin
  test CANNOT catch RoPE-permute (cancels) — only the real-model run does.

**Chat runtime — `ChatSession` now actually drives Llama/Qwen + turnkey:**
- `CachedLlamaSession` now implements `ISlmSession` (was IDisposable-only — the GGUF/safetensors path
  couldn't feed `ChatSession`). `HuggingFaceChatTemplate` reads `tokenizer_config.json` chat_template.
- **`QwenChatModel.LoadFromDirectory(dir)`** — turnkey zero-Python HF dir → `ChatSession`
  (`QwenChatTokenizer` adapts `QwenTokenizer`→`ITokenizer`). **VALIDATED on real Qwen2.5-0.5B**:
  `Send("What is the capital of France?")` → "France's capital is Paris." (Qwen-only; cl100k pre-tokenizer.)

**Anomaly + LoRA product track — closed with measured verdicts:**
- **LoRA target A/B** on the anomaly task (Stage 1/2/3/All): tiny-RANDOM base → single-stage unstable,
  union wins. **TRAINED 256d production base (retrained this session, val 7.70→0.86 in 4m23s) → LM-head
  ALONE is best AND cheapest** (benign 6.45 false-positive → 0.0000, 31694× sep, 1 adapter / 8 KB; union
  needs 205 adapters for worse sep). Cross-pod residual lives in OUTPUT calibration = LM head. **Demo
  reverted to LM-head** (I'd wrongly switched to AllLinear on the misleading tiny-base result).
- **EWMA classical baseline** (`EwmaAnomalyDetector`) + head-to-head: the un-adapted GPT base does NOT
  beat a trivial EWMA floor (base normal 6.45 vs EWMA 0.00) — **the per-pod LoRA adaptation is the edge**,
  not the raw transformer. Demo shows it three-way (EWMA / GPT base / GPT+LoRA). Verdict in
  `docs/gp-anomaly-baseline.md`; GP escalation NOT warranted.
- Production base regenerated at `D:\k8s_anomaly_production.bin` (20.8 MB, out of repo).

**Continued 2026-05-23 — family-generic tokenizer/chat, llama3 RoPE scaling, GPT-2 bit-parity:**
- **Generic HF BPE tokenizer** `HuggingFaceBpeTokenizer : ITokenizer` — reads the pre-tokenizer Split
  regex + merges (both `"a b"` and `["a","b"]` shapes) + EOS from `tokenizer.json`/`tokenizer_config.json`,
  no per-model hard-coding. **VALIDATED bit-exact vs `QwenTokenizer`** (incl. digits) AND **round-trips on
  real Llama-3.2-1B** (vocab 128256). `HuggingFaceChatModel.LoadFromDirectory(dir)` = family-generic turnkey
  (stops per `ChatTemplateFormat`). **Llama-3.2-1B VALIDATED end-to-end**: raw completion "The capital of
  France is" → " Paris. The capital of Germany" (loader correct on Llama dims: 2048d/32h/8kv GQA/16L/tied;
  base model so chat echoes — completion is the right loader check for a base).
- **llama3 RoPE scaling DONE** — `RopeScaling` (NTK-by-parts, port of HF `_compute_llama3_parameters`)
  applied in `RopeTable`, parsed from `config.json rope_scaling` (only `rope_type:"llama3"`). Validated
  (`RopeScalingTests` + real Llama still " Paris" with scaling active). Closes the long-context limitation.
- **GPT-2 safetensors bit-parity VALIDATED** — downloaded `openai-community/gpt2` `model.safetensors`
  (548 MB → `C:\gpt2\`), `Load_RealGpt2Safetensors_BitParity_WithBinFixture` PASSES: loader output is
  **byte-for-byte identical** to `gpt2_small.bin`. **ALL loaders now validated on real models.**

**Out-of-repo dev artifacts (re-runs):** `C:\gpt2\model.safetensors`, `C:\llama3\{config,tokenizer,tokenizer_config}.json + model.safetensors`,
`C:\qwen3b\model.safetensors`, `D:\k8s_anomaly_production.bin`. Real-model tests are `[LongFact]` (flip to `[Fact]` to run).

**Full suite: 765 / 0 / 90 green. The 2026-05-22 batch is COMMITTED (`llama` commits); the 2026-05-23
work (generic tokenizer + chat + rope_scaling + tests) — CHECK `git status`, commit if not yet done.
Working tree was clean at session end except pre-existing staged marketing files (not mine).**

**Resume — pick one (loading + chat tracks are now fully closed & validated):**
1. **Anomaly operator workflow / productization** — real Prometheus metrics → per-pod LM-head adapter
   lifecycle (deployment story; ML verdicts settled — LM-head LoRA on a trained base is the recommendation).
2. **Mistral / non-ByteLevel tokenizers** — extend the generic tokenizer if a SentencePiece/Unigram model
   matters (currently BPE-only; Mistral pre-tokenizer untested — drop a Mistral dir to validate).
3. **Fix `Scripts/convert_llama.py` RoPE permute** (legacy; needs Python to test, low ROI) or the
   **decode lever** (LM-head alloc-free parallel matmul; low strategic ROI per the pivot).

---

**Earlier (2026-05-20): the decode-track sprint (`docs/llamacpp-cpu-analysis.md` §5 steps 1+2+3).**

Three-stage cumulative result on Qwen2.5-3B-Instruct (dev box, best-of-3, single-stream CPU decode):

| Stage | Decode | Steady RAM | Load |
|-------|-------:|-----------:|-----:|
| start (F32-upcast)             | 2.58 tok/s  | ~14 GB   | ~28 s |
| +parallel (step 1)             | 4.01 tok/s  | ~14 GB   | —    |
| +Q8_0 in-RAM (step 2)          | 13.28 tok/s | 5.85 GB  | 1.7 s |
| **+Q4_K_M in-RAM (step 3)**    | **14.56 tok/s** | **4.40 GB** | **1.4 s** |

End-to-end vs Overfit's own starting point: **5.6× decode, −69 % RAM, 20× faster load**, zero allocations per token preserved. Parity verified at both Q8 (32/32) and Q4_K_M (29/32, worst swing 2.16). 680 / 0 / 68 `-c Release`.

**Same-file A/B vs LLamaSharp (2026-05-20, option A done — corrected).** Re-benchmarked LLamaSharp 0.27.0 on the *same* `qwen.q4km.gguf`: **27.5 tok/s @ 3.2 GB**. The earlier "1.51× faster than LLamaSharp" line was wrong — it compared Overfit-Q4_K_M against LLamaSharp's *FP16* number (9.67). Diagnostic that came out of it: FP16→Q4_K_M sped llama.cpp 2.85× but Overfit only ~1.0× → **Overfit decode was overhead-bound, not bandwidth-bound.** First lever acted on (option D below): **GQA K/V-once took Overfit 13.85 → 17.2 tok/s (+24 %)**, narrowing the same-file gap from ~2.0× to **~1.6×** (still llama.cpp's favour, still 27 % more RAM committed). Defensible edge stays allocation (1 B vs 21 KB/token), pure-managed, AOT-clean, no native dep. Full numbers in `overfit-bench/RESULTS.md`.

**Git state at session end:**
- **Committed** (in the `llama` commits on `next`): all Q4_K_M code + tests + `docs/llamacpp-cpu-analysis.md` — `Q6KDotKernel.cs`, `Q6KWeight.cs`, `Q6KDotKernelTests.cs`, `Q4KMDecodeParityTests.cs`, `DecodeWeight.cs` (4-way tagged union `{F32|Q8|Q4_K|Q6_K}`), the per-weight-dispatch decode blocks (`CachedFeedForwardBlock` / `CachedSingleHeadAttention` / `CachedMultiHeadAttention` / `CachedTransformerBlock` / `CachedGptStack`), and the native Q4_K + Q6_K loader reads (`GgufLlamaLoader.cs` / `GgufReader.cs`).
- **Uncommitted at handoff:** this `ROADMAP.md` (the resume-point + Slot 2b update). Pre-existing staged files unrelated to this work: `index.html`, `launch-copy.md`, `linkedin-*.md`, `docs/parallel_opts.txt`.

### (Historical) Order that session: C → B — both delivered

- **(A) LLamaSharp re-bench on Q4_K_M — ✅ DONE 2026-05-20.** Restored the `llama` mode in `D:\overfit-bench` (LLamaSharp 0.27.0, `dotnet run -- llama qwen.q4km.gguf`). Result above: llama.cpp ~2× faster + 27 % less RAM on the same file; "1.51× faster" claim retracted everywhere. Surfaced a new lever (D).
- **(C) — NEXT NOW — Active track: anomaly + LoRA.** Pick one of the four options in the Active-track "NEXT" section below: end-to-end integration test / Stage-2 LoRA on FFN / Production base training / deployment-architecture decision. The live product track.
- **(B) Prefill GEMM (B>1 batched matmul) — ✅ DONE 2026-05-21, 3.48× TTFT.** Phase 1 `BatchedProjectionKernel` → Phase 2 `BatchedAttentionKernel` → Phase 3 `CachedGptStack.PrefillBatched`, wired into `CachedSlmSession.Prefill` (≥16-token GPT-2 prompts). The win came from head-coarse parallelism (the per-head wiring was 3.8× slower). Quant (Q4_K_M) batched prefill is the follow-on — Phase 1 is F32 only. Details in the Prefill section below.
- **(D) — surfaced by (A), 2 levers DONE — make decode bandwidth-bound.** llama.cpp got 2.85× from FP16→Q4 (bandwidth-bound); Overfit got ~1.0× (overhead-bound). **Done #1: GQA K/V-once** — K/V projection was recomputed once per Q head (8× for Qwen 16Q/2KV) instead of once per KV group; +24 % (13.85 → 17.2 tok/s), cut wasted K/V weight-read bandwidth 8×. **Done #2: fuse-quantize** — `hidden` was re-quantized to Q8_K per head per projection (~20×/layer); now quantized once per layer in `CachedMultiHeadAttention`, shared read-only across heads via `Q4K/Q6KDotKernel.ProjectPreQuantized`; **+~1.5 % (17.2 → 17.5 tok/s)** — small, as predicted (quantize is ~0.04 % of matmul arithmetic), but consistent. Both bit-identical (same 24-token greedy sequence before/after) + 680/0/68. Gap to llama.cpp now ~1.6× (was ~2.0×). **Tried #3: VNNI `vpdpbusd` — REVERTED 2026-05-21.** Replaced the AVX2 `vpmaddubsw`+`vpmaddwd` dot with one `vpdpbusd` (`AvxVnni.MultiplyWideningAndAdd`) in both Q4_K/Q6_K kernels + deferred-horizontal-sum; box has `AvxVnni`/`AVX512BW` (verified via bench `caps` mode). Parity 8/8 green. **Same-state A/B: AVX2 ≈ VNNI ≈ 19.1 tok/s — ~0 gain, reverted.** Lesson: after #1+#2 the decode is **memory-bandwidth-bound, not ALU-bound** — fusing the dot saves cycles already hidden behind weight-read latency. (The earlier "17.5 baseline" was a slower thermal state; same-state both ~19.) **The real remaining lever is MEMORY, not ALU:** llama.cpp reads 3.2 GB (mmap), Overfit 4.4 GB committed — the ~1.2 GB extra read per token *is* the speed gap. Levers: mmap-style weight loading instead of committed heap, drop F32 duplication (embeddings/norms), tighter weight layout. GEMV-unroll / core-util profiling are secondary now (ALU isn't the bottleneck). This redirects D from kernel-ALU to memory-bytes-per-token.

  **Profiled 2026-05-21 (thread-scaling + effective-BW).** Decode tok/s vs workers
  (`OVERFIT_PARALLEL_WORKERS`): 1→4.14, 2→8.04, 4→13.86, 8→**19.52**, 16→19.53,
  32→18.90. **Plateaus hard at 8 threads** → memory-bandwidth-bound confirmed
  (more cores starve on RAM). Effective BW: ~2.0 GB streamed/token × 19.5 ≈
  **~39 GB/s (~55 % of DDR5 peak); llama.cpp ~55 GB/s (~80 %)**. Same bytes, same
  RAM ceiling — llama.cpp just extracts more effective bandwidth. **Root cause
  (confirmed by reading llama.cpp `llama-model`): attention weight layout.**
  llama.cpp keeps `attn_q/k/v/o.weight` as full per-projection matrices and does
  ONE big contiguous `mul_mat` per projection (all heads), reshaping to heads only
  inside SDPA. Overfit splits per head (`WqHeads[h]` etc.) → nHeads small
  fragmented GEMVs → prefetcher starved → ~55 % BW. **THE decode lever now:
  fuse per-head attention into full-matrix contiguous matmuls** (Wq/Wk/Wv/Wo as
  full tensors, one streaming K-quant GEMV each, reshape to heads only for SDPA).
  Bonus: full Wo becomes Q6_K-able (contraction dModel, vs per-head headDim<256
  forcing Q8). Est. 55 %→~75 % BW ≈ ~26 tok/s (near llama.cpp's 27.5). **Big
  refactor** — the per-head structure is baked into K/V-once AND Stage-3 per-head
  LoRA, so fusion ripples into the LoRA target design. Deliberate sprint, not a
  tail-end change. (token_embd-as-F32 = the 1.2 GB RAM gap, but it's a per-token
  lookup not streamed → RAM-axis only, does NOT help decode speed.)

  **RAM AXIS CLOSED 2026-05-21.** Investigating token_embd found the embedding was
  held **twice** — engine-owned `TensorStorage` + a per-session `ToArray()` F32
  copy in `CachedLlamaSession`. Dropped the copy (session now references the
  engine storage, sliced per-token; consistent with the zero-copy weight design).
  **RAM 4.39 → 3.21 GB (−1.18 GB) = parity with LLamaSharp's 3.20 GB.** Decode
  unchanged (17.6, as expected — embedding is a lookup, not streamed); bit-identical
  (same 24-token greedy seq); 683/0/68. token_embd→Q6_K could shave another ~1 GB
  (3.2→~2.2) but parity is the milestone and it needs a dequant-row lookup path —
  deferred. **Only the decode-SPEED axis remains** (the attention-fusion lever above).

### Quantized batched prefill (TTFT lever — 2026-05-26, Phase 1 DONE)

The catchable llama.cpp gap that ISN'T the off-moat decode race: for Qwen/Llama/Mixtral, **prefill is
single-token** (`CachedLlamaSession.Prefill` loops `DecodeToken` per prompt token) — batched prefill
(3.48× TTFT) only exists for the F32/GPT-2 path. So a long prompt's time-to-first-token = N × single
decode, re-reading ALL weights N times; llama.cpp batches it. Closing this is on-moat (chat
responsiveness) and — unlike attention-fusion — does NOT touch the per-head LoRA design.

- **Phase 1 DONE — batched quant projection kernels.** `Q8DotKernel.ProjectBatched` +
  `Q4KDotKernel.ProjectBatched` + `Q6KDotKernel.ProjectBatched`: quantize all N activation rows once,
  then split the output loop over `OverfitParallelFor` with the **rows loop innermost** so each weight
  output row is read from DRAM once and reused (cache-hot) across all N dots — cuts weight byte-traffic
  ~N× (prefill is weight-bandwidth-bound). Bit-identical to N× single-token `Project` (parity tests:
  `Q8/Q4K/Q6K DotKernelTests.ProjectBatched_IsBitIdenticalToPerRowProject`, rows ∈ {1,2,3,7,16}). 888/0.
- **Phase 2a DONE — batched SwiGLU FFN dispatch.** `CachedFeedForwardBlock.DecodeSwiGluBatchedDispatched`
  + `ProjectBatchedDispatched` (Q6_K/Q4_K/Q8_0/F32) run gate/up/down as batched projections over N rows
  — the FFN is the dominant weight read (~7× the attention weights for Qwen-3B), so this captures most of
  the prefill weight-byte amortisation. Bit-identical to N× `DecodeSwiGluDispatched`
  (`CachedFeedForwardBlockBatchedTests.DecodeSwiGluBatchedDispatched_IsBitIdentical_To_PerRowSwiGlu`).
- **Phase 2b DONE — batched attention (quant).** `CachedMultiHeadAttention.DecodeBatchedQuant`: KV
  groups processed sequentially (each projection parallelises internally — no nested parallelism), per
  group batched K/V projection (`BatchedQuantProjection.Dispatch` — Q6_K/Q4_K/Q8_0/F32) + per-row RoPE +
  cache writes (K/V-once for GQA); per Q head batched Q projection + per-row RoPE + the proven causal
  `BatchedAttentionKernel` (RoPE-agnostic) + batched O projection. `BatchedQuantProjection` extracted as
  the shared batched-projection dispatch (FFN reuses it).
- **Phase 2c + 3 DONE — wired + measured.** `CachedTransformerBlock.DecodeBatchedQuant` (per-row RMSNorm
  + 2b + 2a, dense SwiGLU only) → `CachedGptStack.PrefillBatchedQuant` → `CachedLlamaSession.Prefill`
  takes the batched path for prompts ≥16 tokens (dense, non-sliding, fits cache; MoE/sliding fall back).
  **Parity: BIT-IDENTICAL to the single-token loop on real Qwen2.5-3B Q4_K_M** (maxAbsLogitDiff = 0,
  same argmax — `BatchedPrefillParityTests`, RMSNorm+RoPE+GQA 16:2+SwiGLU+mixed-K-quant). **TTFT win:
  256-token prompt 12 725 ms → 5 309 ms = 2.40×** (best-of-3, same model). This closes the prefill/TTFT
  gap to llama.cpp (which batches prefill) on the chat-responsiveness axis — the on-moat catch.
- **MoE batched prefill — DONE 2026-05-26.** `MoeFeedForwardBlock.DecodeBatched` (group rows by expert →
  gather → batched expert SwiGLU → scatter) + `Qwen2MoeFeedForwardBlock.DecodeBatched` (batched shared
  expert + per-row sigmoid gate), dispatched from `CachedTransformerBlock.DecodeBatchedQuant` on
  `weights.IsMoe`; session eligibility no longer excludes MoE. **Investigation note (empirical rigor):**
  the first cut accumulated each row's experts in *expert-index* order → real-Qwen-MoE Q8_0 end-to-end
  diverged 0.60 (argmax still matched) — a routing-flip cascade (the reorder perturbs the residual
  stream → a borderline top-k decision flips in a later layer). An isolated block-level F32 parity test
  localised it as NOT a block bug (<1e-4). Fix: stash each expert's output at the row's top-k SLOT and
  sum per-row in top-k order (= single-token order) → now **BIT-IDENTICAL** end-to-end (maxAbsLogitDiff
  = 0 on real Qwen1.5-MoE Q8_0), validated by `BatchedPrefillParityTests` + block-level
  `Qwen2MoeFeedForwardBlockTests.DecodeBatched_MatchesPerRowDecode` (exact). Mixtral (routed-only) reuses
  the same path.
  Remaining (optional): F32/RoPE batched attention could now also route here (the old `DecodeBatched`
  F32-only restriction is superseded for the Llama path); Phase-1-style attention fusion for decode
  (separate, off-moat).

### Speculative decode (prompt-lookup) — DONE 2026-05-26 (decode-throughput lever)

The only axis where llama.cpp still led was single-stream decode (~1.6×). Closed it on repetitive text
WITHOUT chasing SIMD: greedy prompt-lookup speculative decoding, reusing the new batched kernels for the
verify. `PromptLookupDrafter` (n-gram match of the suffix against earlier context → propose the
continuation, no draft model, zero extra RAM) + `CachedLlamaSession.GenerateSpeculative` (embed
[t0, draft…] → ONE batched verify forward → greedy-accept the longest matching prefix; `KeyValueCache.
TruncateTo` rolls back rejected drafts). Output is **BIT-IDENTICAL to greedy single-token** (greedy
verification only accepts what greedy would emit) — validated on real Qwen2.5-3B Q4_K_M
(`SpeculativeDecodeParityTests`, identical 40-token sequence + multi-commit confirmed).
**Investigation (empirical rigor):** first cut measured **1.01×** — the batched stack amortised but I
LM-headed each draft row separately, re-reading the huge LM-head weights N× and cancelling the win.
Fix: `CachedGptStack.ProjectLogitsBatched` (head read once for all rows). Then on a genuinely-echoing
tokenised prompt: **avg 3.45 tokens/step accepted, decode 18.1 → 23.1 tok/s = 1.27×** (vs llama.cpp
27.5 → gap ~1.6×→~1.2× on such text). Honest scope: the win is workload-dependent — high on repetitive /
structured / context-quoting output (code, JSON, agentic — the moat), ~1× on novel text (drafts miss).

**Sampling-correct speculative — DONE 2026-05-26.** Generalised greedy → any sampler (temp / top-k /
top-p / min-p). The prompt-lookup draft is a point-mass proposal, so speculative-sampling reduces to:
accept draft `d` w.p. `p(d)` under the sampler's target distribution, else resample the renormalised
residual `norm(max(0, p − e_d))` — output is distributed EXACTLY as a direct draw from `p` (greedy is
the T→0 special case ⇒ still bit-identical). `TokenSampler.ComputeProbabilities` exposes the sampler's
exact post-transform distribution (shared `SelectSurvivors` core with `Sample`); `SpeculativeSampler.
AcceptOrResample`/`Sample` is the rejection core; `CachedLlamaSession.GenerateSpeculative(in
SamplingOptions, …)` threads it (the correction/bonus token is forwarded so cache + `_logits` stay
consistent — committed = t0 + accepted + 1, ≤ maxDraft+2). Validated: `SpeculativeSamplerTests`
(statistical — empirical output distribution matches the target for ANY draft, incl. the greedy
point-mass) + the greedy-reduction parity (`Speculative_ProducesIdenticalSequence`, bit-identical on
real Qwen-3B). The greedy overload `GenerateSpeculative(history, committed, maxDraft…)` delegates with
`SamplingOptions.Greedy`.

### Prefix / system-prompt KV reuse — DONE 2026-05-26 (agentic multi-turn / multi-request TTFT)

Prefill a fixed prefix (system prompt, few-shot examples, a quoted document) ONCE, snapshot its KV, then
restore it into a session per request — skipping the prefix forward pass entirely (a memcpy, not a
re-encode). `KeyValueCache.Snapshot()` captures the live region <c>[0, CurrentLength)</c> compactly
(per-(layer,head) run, prefix-sized not full-cache); `RestoreFrom` copies it back + restores
`CurrentLength`/`BasePosition`; surfaced as `CachedLlamaSession.SavePrefix()` / `RestorePrefix(snapshot)`.
Restoring a prefix + appending a turn is **bit-identical** to prefilling prefix+turn together (causal —
the prefix never attends to the turn), validated on real Qwen2.5-3B across two different requests reusing
one snapshot (`PrefixKvCacheParityTests`). On-moat: the heavy cost in agentic / chat is re-encoding a long
fixed system prompt on every turn — now a O(prefix) memcpy. (From the "adopt from GPT-chat stacks" scan;
RoPE-scaling NTK/YaRN for >trained-context and flash-attention online-softmax for long-context memory
remain on that list.)

### Gradient checkpointing (TRAINING memory) — DONE 2026-05-26

Measured first (`CnnTrainingMemoryDiagnostics`): the tape arena's **live activations** dominate training RAM
(165 MB vs 14 MB im2col on a wide conv; arena cap 191 MB) — NOT im2col (corrected an earlier wrong guess).
Lever = gradient checkpointing. `ComputationGraph.Checkpoint(segment, input, subArenaElements)` runs a
segment without keeping its activations (forward in a throwaway sub-graph, keeps only input+output), then
recomputes them in backward (`OpCode.Checkpoint` + `CheckpointBackward`, segment stored in a graph-side list
indexed by `op.I1`). Wired two ways: **`CheckpointedModule : IModule`** decorator (drop into a `Sequential`
to checkpoint heavy segments) and **`GPT1Model(config, checkpointBlocks: true)`** (each transformer block
checkpointed — the high-value target: activations × batch × seq). Bit-close to non-checkpointed + lower
arena high-water, validated on MLP, a deep MLP, GPT-1 (6 blocks: logits + block-0 grad match), and a
Sequential (`CheckpointParityTests`, 4 fast). **Measured on a larger GPT-1 (12L, d=512, dFF=2048, b=2,
t=128): live-activation arena 438.5 MB → 18.3 MB = 24× less, 420 MB saved** (`Checkpoint_Gpt1_MemorySavings`,
[LongFact]). Pure "train deeper / longer-seq / bigger-batch on the same
RAM" = moat. Follow-ons: pool the sub-graph arena (currently fresh per call); CNN im2col is ~10× smaller
(low priority).

---

## Recently completed (chronological, newest first)

- **OCR track — CRNN + CTC, end to end (2026-05-27).** New reusable capability: train image→text models in pure C#.
  - `Ops/CtcLoss.cs` — CTC negative-log-likelihood (Graves 2006) forward-backward in log-space + `softmax−posterior` gradient (finite-difference-verified). `Ops/CtcDecoder.cs` — greedy best-path + **CTC prefix beam search** (Hannun 2014) + **LM-rescored beam** via `Ops/ICtcLanguageModel.cs`. `Ops/NGramCtcLanguageModel.cs` — ready-made trainable label n-gram (add-k smoothing + back-off).
  - `Autograd` op **`TransposeLastTwo`** (new `OpCode`, fwd/bwd + facade) — the conv→sequence "map-to-sequence" primitive.
  - `DeepLearning/Crnn.cs` — **facade** `new Crnn(height, width, classCount)`: `ConvLayer → map-to-sequence → LstmLayer → Linear`; `CreateInput`/`Forward`→[T,C]/`ComputeCtcLoss`/`Recognize` (greedy / beam / LM-beam). `IModule`.
  - Demos ([LongFact]): digits (loss→0.006, 8/8) and **lexicon words + n-gram LM** (greedy 17/24 → **LM-beam 24/24**; the LM fixes CREEN→GREEN, BRD→BIRD, …). Unit tests: CtcLoss (FD gradient), CtcDecoder (beam>greedy textbook case + LM flip), Crnn geometry, TransposeLastTwo (exact), n-gram LM.
- **Data-parallel training — public API (2026-05-27).** `Training/DataParallelTrainer` (model-agnostic all-reduce / grad-norm clip / optimizer step / weight broadcast over N `Parameter` replicas, gradients averaged) + `DataParallelSession`/`DataParallelReplica` turnkey builder + `DataParallelLearningRate` (√N / linear scaling). Key fix: `OverfitParallelFor.SuppressParallelismOnCurrentThread` (per-thread inline opt-out) kills the N×cores oversubscription + pool-lock contention — measured **2.3× → 6×** throughput (24 workers, TinyShakespeare GPT-1). TinyShakespeare demo refactored onto it.
- **Gradient checkpointing (2026-05-26).** `ComputationGraph.Checkpoint` runs a segment in a throwaway sub-graph (keeps only input+output, recomputes in backward) + `CheckpointedModule : IModule` + `GPT1Model(checkpointBlocks:true)`. **Measured 24× live-activation arena cut** (438.5 → 18.3 MB on a 12L GPT-1) — "train deeper on the same RAM". Measurement showed the activation arena (not im2col) dominates training RAM.
- **MoE inference verified coherent (2026-05-27).** Re-ran full 24L Qwen1.5-MoE Q8_0 end-to-end → "capital of France?" → "Paris". The `norm_topk_prob` arch-aware fix holds; Qwen-MoE + Mixtral both coherent in pure C#. (Earlier "incoherent" resume note was stale.)
- **`PooledArray<T>` + `Array.Copy` ban (2026-05-26).** `Runtime/PooledArray.cs` — zero-cost `using` wrapper over `ArrayPool` (benchmarked); swept 12 loader try/finally trios onto it. `Array.Copy` added to `BannedSymbols.txt` (4 overloads) → 10 sites converted to `Span.CopyTo` (benchmarked ≥, memmove-safe). _Note 2026-05-29: `PooledArray<T>` was a duplicate of `PooledBuffer<T>` and has been deleted — its callsites migrated to `PooledBuffer<T>(n, clearMemory: false)`. See [[feedback-overfit-pool]]._
- **Q4_K_M parity "bug" resolved — was a test bug (2026-05-26).** Native Q4_K_M decode is correct; `GgufQ4KMParityTests` wrongly compared 4-bit-native logits vs the 16-bit FP16 file demanding exact top-1. Fixed to compare vs F32-dequant of the same file (near-tie tolerant). Isolation ladder (mmap≡copy, kernels 0.36%, head-splits 0.3–1.3%) ruled out every code path.
- **Fuse-quantize decode optimization** (2026-05-20). The attention `hidden` row was re-quantized to Q8_K once per head per Q/K/V projection (~20×/layer for Qwen). Now `CachedMultiHeadAttention` quantizes it once per layer (when attention is K-quant) into a shared read-only buffer; heads' Q/K/V projections consume it via new `Q4KDotKernel`/`Q6KDotKernel.ProjectPreQuantized` (the `Project` core minus the quantize pass). Wo keeps self-quantizing (its input is the per-head attention output, not `hidden`). **Qwen2.5-3B Q4_K_M 17.2 → 17.5 tok/s (+~1.5 %)** — small as predicted (quantize is ~0.04 % of matmul arithmetic) but consistent across runs. Bit-identical (same 24-token greedy sequence) + 680/0/68 `-c Release`.
- **GQA K/V-once decode optimization** (2026-05-20). Under grouped-query attention every Q head in a KV group shares one K/V weight set and one cache slot, but `CachedMultiHeadAttention.DecodeKvGroup` was recomputing the K and V projection (+ RoPE + cache write) once per Q head — 8× redundant for Qwen2.5-3B (16 Q / 2 KV). Added a `projectKv` flag to `CachedSingleHeadAttention.DecodeDispatched` so only each group's first head projects K/V; the rest read what it wrote. **Qwen2.5-3B Q4_K_M decode 13.85 → 17.2 tok/s (+24 %)** (also cuts wasted K/V weight-read bandwidth 8×). Bit-identical: same 24-token greedy sequence before/after on the real model (git before/after). 680 / 0 / 68 `-c Release`. Narrows the same-file gap to LLamaSharp from ~2.0× to ~1.6×.
- **Q4_K_M in-RAM decode path** (`docs/llamacpp-cpu-analysis.md` §5 step 3, sub-steps 3.1 → 3.4). `Q4KDotKernel` + `Q6KDotKernel` (AVX2 `vpmaddubsw` on the 4-bit nibbles / reassembled 6-bit quants + scalar fallbacks); `Q4KWeight` + `Q6KWeight` (output-major super-blocks); `DecodeWeight` widened to a 4-way tagged union `{F32 | Q8 | Q4_K | Q6_K}`; **per-weight dispatch** in `CachedFeedForwardBlock` / `CachedSingleHeadAttention` / `CachedGptStack.ProjectLogits` so a heterogeneous Q4_K_M file (Q4_K attn-Q/K/O + FFN gate/up, Q6_K FFN-down + attn-V + token-embd + output, Q8 per-head Wo) picks the right kernel per projection; native Q4_K + Q6_K loader reads. **Qwen2.5-3B-Instruct: load 1.4 s, decode 14.56 tok/s, steady RAM 4.40 GB** (vs Q8: 1.7 s / 13.28 tok/s / 5.85 GB; vs FP16-src: 7.1 s / 13.29 tok/s / 5.90 GB). 0 B/token preserved. Parity vs same-file F32 baseline: 29/32 top-1 match teacher-forced, worst swing 2.16 (every mismatch a near-tie). 680 / 0 / 68 `-c Release`.
- **Q8_0 in-RAM decode path** (`docs/llamacpp-cpu-analysis.md` §5 step 2). `Q8DotKernel` INT8 `vpmaddubsw` SIMD GEMV + `Q8Weight` output-major storage + `DecodeWeight` tagged handle. LM-head + FFN + per-head Q/K/V/O all Q8-resident. Native Q8_0 GGUF load (no dequant/re-quantize). Qwen-3B decode 4.01 → 13.38 tok/s (3.3×), steady RAM ~14.4 → 5.90 GB (−59 %), 32/32 greedy parity vs F32.
- **GGUF Q4_K + Q6_K dequantization** — `GgmlDequant` pure decoders + `GgufReader` streaming wrappers (stackalloc, zero managed allocations per call). 13 unit tests on synthetic blocks. Loader can now consume any `*.Q4_K_M.gguf` from Ollama/HuggingFace.
- **`[LongFact]` test convention** — 53 integration/diagnostic/training tests gated behind a custom xUnit attribute that skips by default. `dotnet test -c Release` now runs ~15 s instead of multi-minute.
- **`CachedTransformerBlock.Decode` argument validation** — explicit guards for input/output/FfnW1/FfnW2 lengths; surfaces caller bugs as `ArgumentException` instead of `IndexOutOfRangeException` mid-block.
- **Binary loader RAM optimization** — `Unpooled` `TensorStorage` for model weights + direct `ReadExactly` into destination span. Removes pool pow2-rounding overhead and intermediate scratch `byte[]`. **3B FP32: ~30 GB → ~14 GB peak load; matches file size exactly.**
- **Token-by-token streaming** — `CachedLlamaSession.StreamGenerate(StreamingOptions, CancellationToken)` returns `IAsyncEnumerable<int>` with stop-token / cache-full / cancellation termination.
- **Chat runtime: multi-turn template + string stops (2026-05-21)** — `ChatTemplate` (`LanguageModels/Chat/`) renders multi-turn `ChatMessage[]` to a prompt for ChatML / Llama-3 / Mistral, with `Detect(jinja)` fingerprinting a GGUF `tokenizer.chat_template` (no Jinja engine — AOT-hostile) and a ChatML fallback. `StopSequenceDetector` adds correct *string* stop sequences on top of the existing token stops — streaming-safe (holds back a trailing partial that could grow into a stop, never emits the stop marker). 16 unit tests (template detect/render for all 3 formats, stop split-across-pieces / partial / multi-stop / flush). **Validated end-to-end on real Qwen Q4_K_M** (`ChatIntegrationTests`, [LongFact]): detects the GGUF's 2509-char `chat_template` → ChatML, renders a system+user chat, tokenizes, generates, and assembles the stream through the stop detector (markers suppressed).
- **Turnkey `ChatSession` (2026-05-22)** — `ChatSession` (`LanguageModels/Chat/`) wraps any `ISlmSession` + `ITokenizer` + `ChatTemplate`: owns conversation history, and each `Send(userMessage, options, onText)` renders the whole history, prefills, generates, and assembles the reply applying both the EOS token stop and string stops (`StopSequenceDetector`) — incremental detokenize holds back a trailing partial codepoint rather than emit garbage. 3 fake-backed unit tests (history accumulation + template-rendered prefill, EOS stop, string-stop truncation + streaming).
- **Sliding-window KV eviction (2026-05-22, RoPE models)** — `KeyValueCache.Evict(count)` drops the oldest tokens by shifting every (layer,head) K/V block down one contiguous memmove, shrinking `CurrentLength` and growing `BasePosition`; retained K/V are NOT re-rotated — RoPE scores depend only on the relative offset, which the climbing `BasePosition` preserves (the new token rotates at `slot + BasePosition`, the true absolute position). `CachedLlamaSession.EnableSlidingWindow(evictBlock)` (opt-in, RoPE-only — learned-absolute-position GPT-2 can't slide) evicts instead of throwing once the cache fills, so chats run unbounded over a rolling context. Tests: `KeyValueCacheEvictTests` (5 — slot shift, BasePosition, reset, invalid count) + `SlidingWindowTests` ([LongFact], real Qwen — enabling sliding is bit-identical before the cache fills; generates 40 tokens over a 16-slot cache with eviction triggered while non-sliding throws). NOTE: sliding window is intentionally NOT equivalent to recomputing on a truncated context (retained hidden states encode since-evicted tokens — standard StreamingLLM behaviour). **Surfaced through the ergonomic API 2026-05-25:** `ISlmSession.SupportsSlidingWindow` + `EnableSlidingWindow(...)` (default-throw for non-RoPE sessions; real on `CachedLlamaSession`), and `new ChatSession(..., slidingWindow: true)` enables it and drops the "stop at MaxContextLength" loop guard so multi-turn chats roll instead of hitting the wall. Wiring test: `ChatSessionSlidingWindowTests` (3 fast — slides past context, stops without it, throws on unsupported sessions). **Chat-runtime gaps now closed (template, stops, turnkey session, eviction — end to end).**
- **Codebase hygiene (2026-05-22): one top-level type per file across the whole solution.** Audited + split 12 multi-type files (incl. the just-added `ChatTemplate`/`SafetensorsReader`/`SafetensorsSource`); `OfflineTrainingConfig.cs` (held `GptTrainingConfig` + `OfflineTrainingResult`, named after neither) renamed into per-type files. Solution builds clean, 740/0/72.
- **GGUF native loader** — `GgufLlamaLoader` reads GGUF files end-to-end without Python tooling. Supports F32/F16/BF16/Q8_0/Q4_K/Q6_K tensors, hand-rolled protobuf-free parser.
- **Native safetensors reader (2026-05-21)** — `SafetensorsReader` reads the HuggingFace `safetensors` format directly: 8-byte header length + `Utf8JsonReader`-parsed JSON header (reflection-free, AOT-clean — no `JsonSerializer`) + F32/F16/BF16 → F32 streaming dequant via `LoadF32`. Closes the last Python-dependent path (a raw HF repo with no GGUF variant). 7 unit tests over synthetic files (header parse, shape/dtype, F32/F16/BF16 round-trip, error paths).
- **Native GPT-2 safetensors loader (2026-05-21)** — `SafetensorsGpt2Loader.Load(path, Gpt2Config.Small)` builds a `GPT1Model` straight from a HF GPT-2 `model.safetensors`, no Python / no `convert_gpt2.py` / no intermediate `.bin`. C# port of the script's mapping (Conv1D `[in,out]` as-is, `c_attn [d,3d]` → per-head Q/K/V `[d,dHead]`, `c_proj [d,d]` → per-head `[dHead,d]`, `c_attn.bias` per-head split, LM head = `wte.T`); serialises into the exact byte stream `GPT1Model.Load` reads (order ≡ `GPT1Model.Save`), so ordering/shapes are guaranteed by the validated load path. Names resolve with/without `transformer.` prefix. Tests: a synthetic tiny GPT-2 (d=4/2-head/1-layer) with ramp-filled tensors proves Q/K/V/O split + bias split + `wte.T` placement against hand-computed expectations (independent of loader logic) + finite decode through `CachedGpt1ModelAdapter`; a `[LongFact]` asserts **bit-parity of `loader.Save()` vs `gpt2_small.bin`** when a real `model.safetensors` is present (resolves `$OVERFIT_MODEL_DIR` / `C:\gpt2\`, no-ops otherwise — real end-to-end parity needs that file, absent on the dev box for now).
- **Sharded safetensors repos (2026-05-21)** — `ISafetensorsSource` abstraction (single-file `SafetensorsReader` + multi-file `ShardedSafetensorsReader`); `SafetensorsSource.Open(pathOrDir)` factory auto-detects `model.safetensors.index.json` (sharded) vs `model.safetensors` (single). `ShardedSafetensorsReader` parses the index `weight_map` (`Utf8JsonReader`, reflection-free), opens each shard once, reads each tensor on demand from its owning shard (no shard fully materialised — low-RAM). `SafetensorsGpt2Loader` now takes `ISafetensorsSource` so it loads single or sharded transparently. Tests: shard merge + correct-shard read + missing-tensor error, and **a sharded GPT-2 loads byte-identically to the single-file model** (`ShardedSafetensorsReaderTests`). **Next:** Llama/Qwen safetensors loader (port `convert_llama.py` — RoPE/GQA/SwiGLU). (Note: the GPT-2 loader still round-trips weights through a full in-memory `.bin` stream — ~2× peak RAM at load; fine for GPT-2, worth streaming-into-params for larger models.)
- **LoRA adapter** — `LlamaLoRAAdapter` with Enable/Disable in-place weight injection. Zero-copy `TensorStorage` references — adapter updates visible to inference without re-binding.
- **GPT-2 Small inference + parity** — 124M params, KV-cache decode 0 B/token, 6.4× faster than naive O(N²). Top-10 logit overlap 10/10 vs PyTorch, maxAbsDiff 0.000107.
- **KV-cache runtime** — `CachedSlmInferenceEngine` + `CachedSlmSession`. `SingleHeadWeights` / `BlockWeights` / `StackWeights` hold zero-copy `TensorStorage` refs.
- **ONNX import** — 14 operators (Conv, Gemm, ReLU/Tanh/Sigmoid/Softmax, MaxPool, GlobalAveragePool, BatchNorm, Add, Reshape, Flatten, AveragePool, ReduceMean). Linear topology via `OnnxImporter`, DAG/skip connections via `OnnxGraphImporter`.
- **Autograd ownership (PR5)** — `Parameter` first-class type, `AutogradNodeOwnership` enum, `graph.Reset()` by ownership, optimizers on `IEnumerable<Parameter>`.
- **Sampling** — top-P with heap sort, repetition penalty (on `CachedSlmSession`), greedy.
- **PERF kernels** — `LinearKernels.ForwardBatched` weight-stationary outer product, hybrid threshold backward, `MaxPool` pool=2 SIMD fast path.

---

## (Historical) GPT-2 Small showcase — SUPERSEDED by the in-process agentic stack

**Superseded (2026-05-25):** the primary showcase is now the **in-process agentic stack** — RAG +
tool calling + structured output on a Qwen GGUF (see "Nearest plan (2026-05-25)" and `Demo/AgentDemo`).
Qwen / Llama / LoRA / quantization are **no longer deferred — they shipped**. Kept below for history.

**Why (at the time):** the parity claim ("top-10 overlap 10/10 vs PyTorch, maxAbsDiff 0.000107, 0 B / generated token, KV-cache decode") is already implemented and validated. Productizing this single story end-to-end (defended in CI, demoable in one command, documented honestly) is higher-value than chasing more model families. Qwen / Llama / LoRA / quantization work continues to live in the codebase but is **explicitly deferred** out of the current week's focus.

### This week

- [x] **GPT-2 parity diagnostics run on every `dotnet test`** — `Gpt2ImportParityDiagnostics`, `Gpt2ImportStageParityDiagnostics`, `Gpt2ImportAttentionParityDiagnostics` flipped from `[LongFact]` back to `[Fact]`. Sweep cost: +2 s. Headline claim now defended on every push.
- [x] **`Demo/Gpt2ConsoleDemo` project** — exists (`Demo/Gpt2ConsoleDemo`). (The launch showcase is now `Demo/AgentDemo`.)
- [x] **Fixture / model path resolver** — done: `OVERFIT_MODEL_DIR` env var is the resolution path across demos/tests.
- [ ] **GPT-2 generation benchmark** — BenchmarkDotNet: cold-start, prefill cost, per-token decode time, allocations. Confirm 0 B/token quantitatively for the README.
- [x] **README cleanup** — done: README/TECHNICAL/scenarios consolidated and updated through 2026-05-25 (agentic stack, mmap, benchmarks).

### Next (after the GPT-2 week)

- [x] **`Prefill()` vs `GenerateNextToken()` API split** — `CachedLlamaSession` and `CachedSlmSession` both expose `Reset()` (clear-only) + `Prefill(prompt)` + the legacy `Reset(prompt)` facade. Backwards compatible: every existing caller keeps working. Enables chat-history-style incremental context (system → user → assistant turns) without re-prefilling the prefix.
- **Prefill: multi-token batched matmul** — phased work toward 5-10× TTFT speedup. Same FLOPs as today, but weights loaded once for N tokens instead of N times; memory-bound for small models, so the speedup is large for short prompts and grows with prompt length.
  - [x] **Phase 0 — skip LM-head for non-final prompt tokens.** Split `CachedGptStack.Decode` into `DecodeWithoutLogits` (transformer blocks + final norm) + `ProjectLogits` (LM head). `Prefill` in both `CachedSlmSession` and `CachedLlamaSession` now calls `DecodeWithoutLogits` for tokens `0..N-2` and full `Decode` only for the last token. Saves ~27 % per-token cost on GPT-2 Small for N-1 of N tokens → ~25 % overall prefill speedup. Parity preserved (greedy output bit-identical to pre-split, demo test still 0 B / generated token).
  - [x] **Phase 1 — `BatchedProjectionKernel` (2026-05-21).** `[N×I] × [I×O] → [N×O]` allocation-free F32 GEMM, `Project` (sequential) + `ProjectParallel` (over output columns). Loop order output-tile → input → row: each weight tile loaded once and reused across all N rows → weight-read bandwidth amortised N×. Per-output-element accumulation order (input-ascending, `TensorPrimitives.MultiplyAdd`, `x==0` skip) identical to the single-token kernel, so **bit-identical to N× `SingleTokenProjectionKernel.Project`** — verified by `BatchedProjectionKernelTests` (6 cases: N=1/N>1, ±bias, outputSize>tile, GPT-2-scale, input zeros; both seq + parallel exact). Foundation for Q/K/V/O + FFN-W1/W2 batched prefill.
  - [x] **Phase 2 — `BatchedAttentionKernel` with causal mask (2026-05-21).** Scores N query positions against a shared K/V cache in one call; query `i` (absolute pos `basePos+i`, `basePos = cacheLength-rows`) attends `[0..basePos+i]` under the causal mask → reduces to one `ComputeSingleHead` with `sequenceLength = basePos+i+1`, so **bit-identical to per-query single-head**. `Compute` (sequential) + `ComputeParallel` (queries fan out over `OverfitParallelFor` — independent, disjoint output + per-query score-scratch rows, read-only shared K/V; not weight-bound so the win is core-parallelism, not bandwidth). `BatchedAttentionKernelTests`: 5 cases (fresh prefill basePos=0, prefix basePos>0, Qwen-scale head; seq + parallel exact).
  - [x] **Phase 3 — top-level batched stack pass** *(DONE 2026-05-21, ~3.5× TTFT)*. New `CachedGptStack.PrefillBatched(promptTokens, weights, cache, ...)` wires Phase 1+2 through the block batched paths. Scoped first to the F32 / GPT-2 path (standard LayerNorm, GeLU FFN, MHA, no RoPE); SwiGLU/RoPE/GQA/quantized are the follow-on (the quant path also needs a batched K-quant projection — Phase 1 is F32 only).
    - [x] **FFN slice — `CachedFeedForwardBlock.DecodeBatched` (2026-05-21).** Both projections via `BatchedProjectionKernel`, the SAME element-wise `ApplyActivation` across the whole `[N×dFF]` intermediate → bit-identical to N× `Decode`. `CachedFeedForwardBlockBatchedTests` (4 cases: GeLU + ReLU, N=1/N>1). FFN is ~⅔ of layer FLOPs — the biggest single batched win.
    - [x] **Attention slice — `CachedMultiHeadAttention.DecodeBatched` (2026-05-21).** Per head: batched Q/K/V projection (`BatchedProjectionKernel`), per-position cache writes, batched causal attention (`BatchedAttentionKernel` — query n attends `[0..basePos+n]`), batched O projection accumulated into output in head order. Bit-identical to N× `Decode` — `CachedMultiHeadAttentionBatchedTests` (4 cases, with attention bias, N=1/N>1). Scoped F32/MHA/no-RoPE (throws for quant/GQA/RoPE).
    - [x] **Stack orchestration (2026-05-21).** `CachedTransformerBlock.DecodeBatched` (per-row LN → batched MHA → residual → LN → batched FFN → residual, reusing the two verified batched blocks + `SingleTokenLayerNormKernel`) + `CachedGptStack.PrefillBatched` (layer loop + last-token final norm, mirroring `DecodeWithoutLogits`). **Stack-level parity: `PrefillBatched` last-token final-hidden bit-identical to the single-token `DecodeWithoutLogits` loop** — `CachedGptStackTests.PrefillBatched_LastToken_IsBitIdentical_To_SingleTokenLoop` (3 cases, 2 layers, random weights). Scoped F32/GPT-2.
    - [x] **Session delegation — SHIPPED (2026-05-21, after head-parallel fix).** Wired
      `CachedGpt1ModelAdapter.PrefillBatched` (embed N + advance cache + `PrefillBatched`
      + project last-token logits) and delegated from `CachedSlmSession.Prefill` for
      prompts ≥ `BatchedPrefillThreshold` (16; falls back to the single-token loop below
      that and for non-GPT-2 stacks via `SupportsBatchedPrefill`). Correctness: batched
      last-token logits bit-identical to the single-token loop (`CachedSlmPrefillBatchedTests`,
      3 cases). **TTFT on GPT-2-Small dims, 64-token prompt: 567 ms single-token vs 163 ms
      batched = 3.48× faster** (`Ttft_BatchedVsSingleToken_Gpt2SmallDims`, [LongFact]).
    - **The fix that flipped it (the key result).** First wiring measured **0.26×
      (≈3.8× SLOWER)**: the batched MHA fanned the per-head Q/K/V/O projections out as
      ~60 tiny `OverfitParallelFor` dispatches per layer (~720 for 12 layers), whose
      wake/sync overhead swamped the weight-reuse win. Restructured `DecodeBatched` to
      parallelise **over heads** — one `OverfitParallelFor.For(0, HeadCount, …)` dispatch
      per layer, each head doing its Q/K/V projection + attention + O projection
      sequentially into a per-head scratch band, then bands reduced into the output in
      head order (bit-identical). One dispatch/layer instead of ~60 → **0.26× → 3.48×, a
      ~13× swing**, same kernels, same math. **Lesson (carry forward):** batched prefill's
      win is real only when the parallelism granularity is coarse (over heads / big
      matmuls), never per-head — dispatch overhead dominates at fine granularity.
  - [ ] **Parity tests** for each phase: batched output ≡ N × single-token output for any prompt up to ContextLength. Final assertion: `Gpt2ImportParityDiagnostics` still green (full PyTorch parity through the batched path).
- [x] **LM-head hot-path audit (initial)** — confirmed `ProjectParallel` exists but is **dead code** (no call site); `Project` is what GPT-2/Qwen actually use. Wiring `ProjectParallel` into `CachedGptStack.Decode` was tested and reverted: `Parallel.For` allocates ~3 KB / call from task scheduling, which breaks the 0 B / generated token contract for only ~3 % per-token speedup at the GPT-2 Small scale. The 10× speedup in `LmHeadParallelBenchmark` is steady-state; per-token decode is dominated by the `Parallel.For` overhead.
- [ ] **LM head: allocation-free parallel matmul** — wire-up depends on a worker pool that does NOT allocate per call. Candidates: pre-spawned threads with lock-free queue / semaphore signaling, or unsafe manual partitioning over a fixed thread set. Constraint: ≤ 0 B / call. Payoff: most of the ~3.8 ms LM-head matmul on a 32-core box. Single largest remaining lever for GPT-2 tokens/sec.
- [ ] **`Gpt2.Load(...) / CreateSession()` API sugar** — `new GPT1Model(Gpt2Config.Small)` is technically correct but semantically misleading. A typed entry point reads cleaner in the demo.
- [x] **Stabilize `GPT1_GradientCheck_BackwardIsCorrect`** — pre-fix: model weights randomly initialized + tight `relErr < 10 %` threshold → ~1-in-3 sweeps red. Fix: seeded weight init (deterministic per run) + mixed tolerance (`relErr < 50 %` OR `absErr < 5e-4`) that accepts the inherent FP32 finite-diff noise floor (~ loss_precision / (2 × eps) ≈ 2.5e-3 per gradient). Test still catches sign errors, factor-of-2 backward bugs, and zero-vs-non-zero regressions. 5/5 full sweeps green post-fix.

---
