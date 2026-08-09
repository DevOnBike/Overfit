# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Pure C# deep-learning / optimization engine targeting **.NET 10**, with a strong
"zero-allocation, Native-AOT-compatible CPU inference" identity. No native
binaries, no Python runtime, no ONNX Runtime dependency. Public NuGet ID is
`DevOnBike.Overfit`.

## Solution layout

`Overfit.sln` contains five projects:

```text
Sources/Main             DevOnBike.Overfit       library, AOT-compiled in CI
Sources/Benchmark        Benchmarks (exe)        BenchmarkDotNet harness
Tests                    DevOnBike.Overfit.Tests xUnit
Demo/MnistWpfDemo        MnistWpfDemo (exe)      WPF MNIST predictor demo (net10.0-windows)
Demo/Unity               UnitySwarmServer (exe)  swarm engine demo server
```

`Main` exposes `InternalsVisibleTo` to `DevOnBike.Overfit.Tests` and `Benchmarks`,
so tests can reach internals directly. Versions are pinned centrally in
`Directory.Packages.props` (CPM enabled); `Directory.Build.props` enables
`NuGetAudit` and promotes vulnerability warnings (`NU1901-1904`) plus `CS4014`
to errors.

## Git / GitHub boundary (hard rule)

Claude is **read-only** on git history and GitHub. Never run `git commit` / `git push` / `git rebase` /
`git reset --hard`, and never run mutating `gh` or GitHub API calls — no `gh workflow run`, no
`gh release create/edit`, no PR/issue creation. Those are the user's actions, even when a plan lists them
as the next step. Reading is fine (`git status/diff/log`, `gh run list/view`, `gh release view`).
Stop at a clean/staged working tree, report exact commands or UI steps for the user, and verify after
they've run them.

## Filesystem boundary outside the repo (hard rule)

Destructive filesystem operations are confined to this repository (`D:\Overfit`). **Never delete, move,
rename or overwrite anything on the system drive or elsewhere outside the repo — `C:\` in particular —
without the user's explicit permission for that specific path.** This covers `rm`/`Remove-Item`,
`mv`/`Move-Item`, `git mv` outside the tree, redirecting output over an existing file, and `Write` to a
path you did not create. Model fixtures live outside the repo (`C:\gpt2\`, `C:\qwen3b\`, `C:\gemma`,
whisper/embedding models) and are large, hand-collected and NOT reproducible from this repo — losing one
costs a multi-GB re-download at best.

Reading outside the repo is fine (loading fixtures, inspecting logs), as is writing to a temp directory
you created. If a task genuinely needs a delete or move outside the repo, **ask first and name the exact
path** — the user will say yes or no. Asking costs one message; an unrecoverable delete costs a lot more.

## Common commands

```powershell
dotnet build -c Release                                         # whole solution
dotnet test -c Release                                          # all tests (fast ones only — see test discipline)
dotnet test -c Release --filter "FullyQualifiedName~Gpt2"       # subset
dotnet test ./Tests/Tests.csproj -c Release --collect:"XPlat Code Coverage" --results-directory ./coverage

dotnet run -c Release --project Sources/Benchmark -- --filter "*SingleInferenceBenchmark*"
.\Sources\Benchmark\run.cmd                                      # runs all benchmarks (--filter *)

dotnet publish ./Tests/AotSmokeTest/AotSmokeTest.csproj -c Release -r linux-x64 -p:PublishAot=true -p:TreatWarningsAsErrors=true  # real AOT guard (requires C++ toolchain locally)
.\update-code-headers.cmd                                        # applies file-header template (dotnet format / IDE0073)
.\cleanup.cmd                                                    # purge bin/obj/.vs caches
```

`Benchmark/Program.cs` uses `BenchmarkSwitcher.FromAssembly(...)`, so the standard BenchmarkDotNet CLI
works end to end — select a class with `--filter`, or pass nothing for the interactive picker.

**Only one benchmark process may run at a time, and this is enforced.** `Program.cs` takes a `Global\`
named mutex and a second process **exits with code 2** rather than queueing. Two concurrent runs do not
produce two results, they produce two wrong ones: they compete for cores, L3, memory bandwidth and the
same thermal budget — the exact set of things every measurement here is trying to hold still. If you see
exit code 2, something else is measuring; don't start a competing run and don't build anything until it
finishes.

Python conversion scripts (run from `Scripts/`) need a local Python with
`torch`, `transformers`, `huggingface_hub`, `numpy`:

```powershell
python Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/
python Scripts/convert_gguf.py ...
```

## How Claude runs those commands here (`.claude/do.py`)

The commands above are what a **human** types. Claude does not type them directly — **every shell command
goes into `.claude/do.py` and is executed as the single invocation `python D:/Overfit/.claude/do.py`.**
(It was `run.py` until 2026-08-07; the allow-list in `settings.json` still carries the old name in a few
redundant entries, which are harmless because `Bash(python *)` covers both.)

**This applies to the main session AND to subagents, but through different files.** The main session uses
`.claude/do.py`; each agent in `.claude/agents/**` uses its own `.claude/do-<agent-name>.py`, declared in its
own definition. Per-agent files exist because a single shared scratch file is overwritten by two agents
running at once — which is why this rule excluded subagents until 2026-08-08. The rule is discipline, not a
permission boundary: `settings.json` allow-lists `Bash(python *)` broadly, so nothing here is about
suppressing prompts.

It is a discipline rule, and the discipline is what pays — not the permissions. (`settings.json` in fact
allow-lists `Bash(dotnet *)` and `Bash(python *)` broadly, so the original "zero prompts" justification is no
longer why this exists.) What it buys is: the command lives in a file that can be re-read and corrected rather
than retyped; output is filtered in Python instead of by `grep`, so a failing test name survives; environment
variables go through `env=` in `subprocess.run` where they actually take effect; and long scripts avoid the
shell quoting that eats backticks, `$` and regex classes on the way in. Every one of those has gone wrong here
at least once.

Practical consequences, each of which has already gone wrong at least once:

- **Env vars for an A/B go through `env=` in `subprocess.run`**, never as a shell prefix. A prefix does
  not survive the way these commands are invoked, and the arm you thought you were toggling runs
  identical to the other one.
- **`run.py` is scratch.** It is rewritten for each task and is gitignored — never put anything in it
  that needs to survive, and never treat its current contents as documentation of anything.
- **Filter the output in Python, not with `grep`/`sed`.** `dotnet build` on this solution emits far more
  than fits in a reply; print only errors, the diagnostics you asked for, and the test summary line. When
  a test fails, print the **test name** — twice now a real failure has been lost because the filter kept
  only the summary.
- **Watch the quoting.** Long scripts belong in `run.py` written with `Write`, not squeezed into
  `python -c` — backticks, `$`, `\` and regex character classes get eaten by the shell on the way in.

Repeatable versions of the three most common cycles live in `.claude/commands/` — `/check` (build + full
suite), `/bench <filter>` (benchmark + the measurement traps to check before believing the number), and
`/sweep <OVERFIT0xx>` (inventory every site an analyzer rule flags).

## How an anomaly (`AN-*`, `RS-*`, `PS-*`) task is run, start to finish

**When this applies:** any change to *what the guard detects* — a channel, a binding, a threshold, a rule, a
detector. Not refactors, not a rename. **One task at a time.**

**Where the task lives:** `docs/TASKS.md` is the registry and the only place carrying status;
`docs/aiops/aiops-backlog.md` is domain prose and its rows are commentary, not state. Editing a status in
both is how they diverge — that happened on 2026-08-08.

### Before implementing

**1. Read `docs/aiops/aiops-detection-pipeline.md`, `aiops-adding-a-metric.md` and `aiops-repair-plan.md`,
then the code path that produces the number — and quote the decisive arithmetic.** Not the doc comment. Also
read any rule, profile or constant that already exists for this signal, **including its calibration
conditions**, because they name the environment you must reproduce.

**The task description is not a source of truth.** *Earned three times on 2026-08-09:* an `AN-D2` row written
four hours before its own fix; a `+1.00` premise with no artefact computing it anywhere; a 200-line
diagnostic reproducing `series - expectation` when `AnomalyGuard.Adjust` computes `series - expectation +
median` twenty lines away, which made the recorded diagnosis an artefact. And three hours of CPU-limit
measurement answered a question `SustainedThresholdOptions.ForCpuThrottling` states in its own doc.

**2. Name or create the artefacts, before writing code**: the query/binding, a **positive** fixture, a
**negative** fixture, a **missing-data** fixture, and the expected output for each. The missing-data one is
not optional and not a formality — it is the only artefact that distinguishes a working detector from a
dead one.

**3. State the premise out loud**: what produces this number, in what unit, and what observation would refute
the explanation. It is catchable from outside by someone who has not read the code, which is the point —
nobody can challenge a premise that was never stated.

### Implementing

**4. Every change must distinguish `Detected`, `Healthy`, `WarmingUp`, `InsufficientData`, `QueryFailed`.
Absence of series is NEVER `Healthy`.** This is the single most important rule in this subsystem, because a
detector that works is silent almost all the time — so every defect in it *presents as silence*, and silence
is indistinguishable from success. Partial precedent exists and should be consolidated rather than
duplicated: `GuardCycleOutcome` (`Completed`/`Blind`/`Failed`) and `DiscoveryOutcome`
(`Resolved`/`NotFound`/`Ambiguous`) carry the same distinction at the cycle and binding level; this is the
per-signal version.

**5. No threshold without a measurement IN THE MECHANISM'S UNIT.** Measurement alone is not enough: on
2026-08-09 both bad thresholds *were* measured — a CPU limit sized at 83x the average when CFS throttles on
bursts inside a 100 ms period, and a floor calibrated over three minutes when the quantity is a maximum and
the unit is time coverage. Correct arithmetic about the wrong quantity.

**6. No new metric without proving the workload actually emits it** — queried, non-empty, on the pods the
guard watches. Two channels were bound to series that were structurally incapable of moving.

### Proving it

**7. Run a mutation that should break the test. If the test does not fail, the task is not finished.** Assert
the mutation anchor matched exactly once and print the count; refuse to start if the target already differs
from HEAD, because a harness killed mid-run leaves the source mutated and the next run reads that as its
baseline.

**8. Both arms.** Healthy quiet AND faulted loud, same population, peers as control. For cluster-side work
the positive fixture needs a live counterpart: a fixture proves the code path, an **injected fault** proves
the chain.

**9. Read the deployed state back out of the cluster.** `kubectl apply` reports success for a field it
dropped and silently removes anything the file does not carry — that deleted a live `GcCommittedBytes`
binding and reported `configured`.

### The report

Changed files; artefacts read; test results; **mutation result**; **silence risk** — how this specific change
could fail without anyone noticing; and known limitations. The silence-risk line is the one that earns its
place: everything else says what works.

**Close with self-improvement notes: what cost iterations on THIS task, and what would have prevented it.**
Not a ritual and not an apology — a specific, checkable observation, or the honest sentence that nothing
went wrong. Every rule above exists because a mistake was named this way; the ones that were not named
repeated. Two examples of the right shape, both from the day this was written: *"the fault was sized for a
200m limit and not recomputed when the limit moved to 1000m — one variable changed and the consequence was
not propagated"*, and *"my own test observed exactly `MinimumWindows`, which is also the legacy fallback, so
both sides read 24 and the test could not fail"*.

Two things to be strict about here. **A rule in a file is weaker than a gate in code** — on the day this was
written, the only two things that actually caught anything were a parity test and a mutation harness, not
paragraphs — so when the note is worth keeping, say whether it can become a test rather than a sentence.
And **an agent's own account of its run is evidence about its instructions, never evidence that its output
is sound**: one reported "the instructions worked as intended" in a run where it had failed.

Closing the task is then checking the list from step 3, not forming a judgement.

## Read the code before you measure it, and check the number's unit

Two rules, both earned on 2026-08-09 by breaking them four times in one day. They are cheap, and every
violation cost an hour or more of work that had to be thrown away.

**Before measuring what a component does, read the code path that produces it, end to end, and quote the
decisive line.** Not the doc comment — the arithmetic. A 200-line diagnostic was written to reproduce the
seasonal correction and its conclusion recorded as a diagnosis, without anyone opening `AnomalyGuard.Adjust`
first; that method is twenty lines long and its `+ level` term made the whole result an artefact of an
operation the product does not perform. The same shape produced the `+1.00` correlation that had no artefact
behind it at all.

**Before choosing any threshold, name the mechanism and check the number is in the mechanism's unit.** A CPU
limit was sized at "83x the average usage" when CFS throttles on bursts inside a 100 ms period — average
cores is not the unit that governs it, so the headroom was real and irrelevant. A floor was calibrated over
three minutes when the quantity is a maximum and the unit is time coverage. Both numbers were measured
correctly and meant nothing.

**Say it out loud before running the measurement**, in two lines: what produces this number, in what unit,
and what observation would refute the explanation. It is catchable from outside by someone who has not read
the code, which is the entire point — nobody can challenge a premise that was never stated.

## Answer briefly — you and every agent

Default to the short answer: the result, the number, what it means, what is open. **Expand only when the user
asks for it.** This binds the main session and every agent in `.claude/agents/**` equally.

What "brief" does not mean: dropping a measurement that contradicts the conclusion, dropping the reason a
result is uncertain, or dropping what was NOT checked. Cut the narration of how the work was done, the
restatement of what the user just said, and the reasoning that led nowhere — never the evidence.

## Agents

**You have a team, and you may use it without being asked to.** This is standing permission, recorded here
on 2026-08-09 because the default posture is to dispatch nobody unless the user says so, and that default is
wrong in this repository: the agents exist precisely so that work with a specialist owner goes to its owner.
Dispatch `overfit-analyst` when a request arrives as prose and its scope is not obvious, `overfit-architect`
before a change crosses an assembly or a public API, `overfit-developer` for a task from a signed plan,
`overfit-verifier` and `overfit-reviewer` after it, and the conditional specialists when their trigger fires
— above all `overfit-perf-claim-auditor`, which **owns the verdict on any performance claim** and must not
be substituted for.

Two limits on that permission, both from measurement rather than caution. **Say which agent you dispatched
and why**, in the reply, because a finding relayed without its source cannot be weighed. And **a subagent's
self-assessment is evidence about its instructions, never evidence that its output is sound** — see the
2026-08-06 note below, where four agents opened by acknowledging an answer nobody had given. Judgement about
whether the work is right stays with the main session; the dispatch does not move it.

`.claude/agents/` holds **twelve** specialised agents, each with its own context and a `memory:` directory
that persists across sessions. The delivery chain is `overfit-analyst` → `overfit-architect` →
`overfit-developer` → `overfit-verifier` → `overfit-reviewer`, and it is gated: the analyst and architect
write **one** plan file in `docs/specs/`, and the developer refuses to write source until the architect has
signed it. `/overfit-delivery` (a skill, because only the main session can dispatch agents and ask the user
questions) runs that chain and enforces the gates. Conditional
specialists: `overfit-perf-claim-auditor` (**sole owner of the verdict on any performance claim** — others
detect and defer), `overfit-security` (parsers, endpoints, gateway), `overfit-ciso` (threat model, supply
chain, disclosure), `overfit-code-with-description-drift`, `overfit-packages-update`,
`overfit-release-readiness`, `overfit-find-bugs-game` (exploratory, not a gate).

`overfit-developer` is the only one that may modify source; the rest report. The three whose findings can be
an unfixed vulnerability use `memory: local`, which is **not** tracked by git.

**One limit of that arrangement, learned the hard way on 2026-08-06.** Each agent closes with a
`SUGGESTED IMPROVEMENTS TO MY ROLE` section, and that catches stale instructions and missing tools well. It
does **not** catch an agent being wrong about reality: four agents, resumed with no user input, each opened by
acknowledging an answer that had never been given, and one wrote a fabricated quotation into a file. The agent
that failed worst reported "None this run — the instructions worked as intended" in the same turn. From
inside, an invented memory of an answer is indistinguishable from a real one. **Cross-agent failure modes are
visible only from the main session** — treat a subagent's self-assessment as evidence about its instructions,
never as evidence that its output is sound.

## Native-AOT discipline (this is the trip-wire)

Two independent layers guard the library against trim/AOT regressions:

1. **`Sources/Main/BannedSymbols.txt`** (enforced by `Microsoft.CodeAnalysis.BannedApiAnalyzers` with `RS0030` set to **error** in `.editorconfig`) forbids the following at every `dotnet build`, not just at publish:
   - `System.Linq` (the namespace is also `<Using Remove="System.Linq" />` in `Main.csproj`)
   - `System.Reflection`
   - `System.Linq.Expressions.Expression`
   - `System.Activator`
   - `Array.Copy` (use `Span<T>.CopyTo`)
   - Raw `ArrayPool<T>.Shared` (use `PooledBuffer<T>` or `PooledArray`)
2. **`Tests/AotSmokeTest`** is a thin console exe that the `aot-guard` CI job publishes under `PublishAot=true` + `TreatWarningsAsErrors=true`. Libraries cannot be Native-AOT compiled directly (no entry point), so the smoketest is the real AOT consumer — ILCompiler actually runs, IL2026 / IL3050 / IL31xx warnings on reachable code are promoted to errors, and the resulting native binary is executed as a smoke check. Extend `Tests/AotSmokeTest/Program.cs` cautiously: each new touched type or method widens AOT verification scope but may surface latent trim warnings that block publish until the library is fixed.

Use explicit `for`/`foreach` over `Span<T>`, delegates over reflection, explicit
`new` over `Activator`. This rule applies to `Sources/Main` only — tests and
benchmarks may use LINQ.

`Sources/Main` also enforces hot-path conservatism (see `Sources/Main/README.md`):
no LINQ in runtime code, no hidden allocations in inference, no `.ToArray()`,
no `model.Forward(...)` in the inference hot path — go through
`InferenceEngine.Run(input, output)` with caller-owned buffers.

## Source-file guards (MSBuild, build-time errors)

`Main.csproj` runs two `RoslynCodeTaskFactory` guards `BeforeTargets="CoreCompile"`
(BannedApiAnalyzers can only ban named symbols, so these structural rules are MSBuild
tasks that scan `@(Compile)` and `Log.LogError` — they fail every `dotnet build`):

- **`BanJaggedFloatArrays`** — `float[][]` (jagged) is banned in `Sources/Main`
  (`OVERFIT-JAGGED`). Use a flat `float[]` (Span-sliced per row — one allocation,
  cache-friendly) or an Overfit buffer (`PooledBuffer<float>`, `TensorStorage<float>`).
  `int[][]`, `Parameter[][]`, etc. are still allowed.
- **`BanMultipleTopLevelTypes`** — one top-level type per file (`OVERFIT-ONETYPE`):
  each `.cs` declares at most one namespace-level `class`/`struct`/`interface`/`enum`/
  `record`. **Nested types are fine**, and **`partial` declarations of the same type
  across files are fine** (collapsed by name). Assumes block-scoped namespaces
  (top-level types indented 4 spaces). Split helper enums/records/contexts into their
  own files named after the type.

## Code style

`.editorconfig` enforces:

- block-scoped namespaces (warning)
- `csharp_prefer_braces = true`
- `dotnet_sort_system_directives_first`, no separate import groups
- `IDE0073` (file header) as a warning — the template is the AGPL/commercial
  notice in `.editorconfig`. `update-code-headers.cmd` applies it across the
  tree.

`IDisposableAnalyzers` is wired into `Main` — heed its diagnostics; the project
leans heavily on `using` + pooled `TensorStorage<T>` lifetimes.

## Architecture (the parts you need to read multiple files to see)

### Inference vs. training separation

There are **two distinct execution paths** with different allocation policies,
and mixing them is the single most common architectural mistake:

- **Inference**: `InferenceEngine` (caller-owned buffers, zero allocations per
  call) → `IInferenceBackend` → `SequentialInferenceBackend` /
  `OnnxGraphInferenceBackend`. No `AutogradNode`, no `ComputationGraph`.
- **Training**: `ComputationGraph` records a tape of `AutogradNode`s, then
  `graph.Backward(loss)` walks it; `graph.Reset()` reclaims temporaries by
  ownership. Operations that record tape live on the graph
  (`graph.Linear`, `graph.Conv2D`, `graph.Relu`,
  `graph.SoftmaxCrossEntropy`) — the older `TensorMath.*(graph, ...)` style is
  being migrated onto the graph facade (`docs/OverfitArchitectureRefactorPlan.md`).

### Autograd ownership model

Every `AutogradNode` carries an `AutogradNodeOwnership` tag set at creation
that determines who disposes it:

| Ownership | Disposed by |
|-----------|-------------|
| `GraphTemporary` | `graph.Reset()` |
| `GraphAuxiliary` | `graph.Reset()` (e.g. MaxPool index map, Softmax probs) |
| `Parameter` | the owning layer's `Dispose()` |
| `ExternalBorrowed` | the caller |
| `View` | never (no backing storage) |

`Parameter` is a first-class type; optimizers take `IEnumerable<Parameter>`
(`Adam(parms, lr)`, `SGD(...)`), and `layer.TrainableParameters()` is the
canonical way to enumerate them.

### GPT / SLM runtime (KV-cache, zero-alloc decode)

The language-model runtime in `Sources/Main/LanguageModels/Runtime/` is layered
so weights are **never copied** at session creation:

```text
CachedSlmInferenceEngine  ← public entry; FromGpt1(model) wires the adapter
  CachedSlmSession        ← per-session state: KV buffers + position counter
    StackWeights          ← BlockWeights[] + final norm + LM head
      BlockWeights        ← layer norms + per-head attention + FFN
        SingleHeadWeights ← ReadOnlySpan refs into TensorStorage for Q/K/V/O + biases
    KeyValueCache         ← pre-allocated K/V, O(N) decode
```

All weight handles are `ReadOnlySpan<float>` obtained from `TensorStorage` at
decode time, which is why session creation allocates the KV buffers
(~80 MB for GPT-2 Small) and **per-token decode allocates 0 B**.
`CachedGpt1ModelAdapter.RefreshWeightsFromModel()` is a deliberate no-op for
in-place weight updates (LoRA path).

### ONNX import — two importers

- `OnnxImporter.Load(path)` — linear topology → `Sequential`. Faster, simpler.
- `OnnxGraphImporter.Load(path, inputSize, outputSize)` → `OnnxGraphModel`
  (DAG) → wrap in `OnnxGraphInferenceBackend` then
  `InferenceEngine.FromBackend(backend)`. Required for ResNet/DenseNet-style
  skip connections.

External `.data` sidecar files (PyTorch ≥ 2.x default) are resolved
automatically. There is no `Google.Protobuf` dependency — protobuf parsing is
hand-rolled in `Sources/Main/Onnx/`. Unsupported operators throw a clear
`NotSupportedException` naming the operator.

## Performance work — measure, don't assume

**Write the benchmark first, whenever a benchmark makes sense.** Before reasoning about whether a change
is faster — and *before* asserting anything about it in prose — put the question into
`Sources/Benchmark` as a BenchmarkDotNet class with the two shapes side by side and
`[Benchmark(Baseline = true)]` on the old one. A claim about performance that has no benchmark behind it is a
guess, however confident the reasoning sounds; the ratio column is the only thing that settles it. This is
cheap for anything expressible as a small A/B (loop shapes, call shapes, allocation strategies, kernel
variants), so default to writing it rather than arguing. Two traps this repo has already hit, both of which
produce numbers that look authoritative and are worthless:

- **Wrong job for the workload.** The shared `BenchmarkConfig` pins `InvocationCount=1`/`UnrollFactor=1`,
  which fits multi-millisecond model runs but leaves a microbenchmark measuring timer noise — a ~15 µs
  operation produced `RatioSD` 0.44 and a phantom 1.61x regression that was 1.01 once re-run under a
  microbenchmark job (`[SimpleJob]`, default invocation counts). Check `RatioSD` and BDN's own warnings
  before believing a ratio.
- **The scaffolding outweighs the subject.** `ElseRefactorBenchmark` first reported a non-inlined call as
  *faster* than inlining it, because the synthetic branch body contained a saturating `float`->`long` cast
  whose cost depends on where it lands. The benchmark was measuring the cast, not the call. If a result is
  backwards, suspect the benchmark before the runtime — and use `--disasm` plus a variant with the suspect
  operation removed to settle it.

**Two passes, never one.** Write the correct algorithm first — the clearest expression that gets the math
right — and pin it with tests (parity against a reference / known-good output, ideally box-independent like
a cosine-vs-ORT or FD check). Only once correctness is green do you make a **second, separate** iteration
for performance, keeping the validated version as the baseline you A/B against and the parity test as the
guard that the fast path still matches. Do not fuse the two: a clever kernel written before its correctness
is proven is unverifiable, and a perf change that also alters behaviour can't be A/B-isolated. This is how
Winograd (parity cos 1.0 first, *then* measured → reverted as a negative) and whole-matrix Q4_K attention
(split-after parity pinned, *then* a go/no-go micro-bench before any refactor) were done.

Every perf change is a **hypothesis until measured**. Benchmark before/after with BenchmarkDotNet +
`MemoryDiagnoser`, **best-of-N on BOTH sides**, and A/B-isolate the one lever you changed. Document
**negative results** honestly — they are the most valuable output: in this codebase
register-blocking (direct-conv), K-blocking + A-packing (im2col GEMM), Winograd F(2,3) for 3x3 stride-1
convs (parity-correct cos 1.0 but +79% slower on deepcnn, 119.7→214.4 ms — sequential scalar transforms +
16 small GEMMs + 16x U/V/M blow-up beat the 2.25x FLOP cut), the AVX-512 decode port, bias support in
the Q4_K tiled prefill GEMM (`GemmTiled` — a path census showed `bias.IsEmpty` barred 88% of prefill
dispatches, i.e. all attention Q/K/V, from the tiled kernel; lifting it measured **0.999x, an exact
tie**, because `ProjectBatchedWeightStationary` already amortises weight decode across the row tile —
the same thing the tiling does; the "~3x" in the kernel docs is against re-decode-per-row, not against
weight-stationary), and `OverfitPool<T>` all **regressed or tied and were reverted**; the wins were the *opposite* of the
"obvious" move (`TensorPrimitives` bulk-SIMD beat a hand micro-kernel; the simple register-blocked
GEMM beat the cache-blocked one — structure of the data around the technique decides, not the
technique). Mind the **measurement environment**: a thermally-throttled or loaded box invalidates
A/B (detect it with a *canary* — re-measure an unchanged code path; if it shifted, the box did, not
your change). The decode spin-pool assumes dedicated cores, so it is sensitive to background load.
Two corollaries this repo has paid for. **Cross-process before/after does not work here**: a prefill
change read as +5% while the untouched decode path in the same run moved +32% — interleave the
configurations run-by-run in ONE process (ABAB…, not all-A-then-all-B) and time a canary path in
every sample. **Verify the flag you are A/B-ing is actually live**: `OVERFIT_TILED_PREFILL` is a dead
flag whenever a `*.gguf.repack` sidecar sits next to the model, because `IsPrepacked` short-circuits
it — both arms ran an identical mix and the "measurement" was noise. Count the paths taken (a
temporary counter in the dispatcher) before believing any kernel A/B.
Never ship, claim, or commit a perf "win" you have not measured on a stable box — and prefer
measuring over reasoning even when the reasoning feels airtight.

## Automated search instead of hand-tuning (`docs/autoresearch-program.md`)

When a question is **"what value"** rather than **"what mechanism"**, do not hand-tune it. Write the objective
and let a search run it. The worked example is `SyntheticClusterCalibrationSearch`, which fits the synthetic
generator to the recorded lab window: six hand iterations over an afternoon were replaced by 281 evaluations
in 17 seconds, and the fitted shape scored **0.064 against 0.130 for the hand-tuned one**.

**Use it when all three hold**, and it is worth stating the hypothesis out loud first — the search settles it,
it does not invent it:

1. The objective is **cheap and deterministic** (sub-second, in-process, seed-averaged).
2. The objective is **fair**: it cannot be satisfied by a change that improves nothing real. This is the hard
   part and it is what `val_bpb` is for in the original. Between-pod spread was deliberately excluded from
   the calibration objective because on the lab side it is a range over three draws — the same unchanged
   generator moved it from 12% to 28% when an unrelated edit shifted the random stream, so a search scored on
   it would chase a coin flip and report convergence.
3. The **reference is gated first**. `LabWindowValidator` rejects a contaminated recording, and
   `LabWindowFixtureTests` runs it on every `dotnet test` with one reproduction of each real failure so the
   validator cannot be vacuous. Fitting to a broken reference is worse than not running: fast, repeatable,
   and wrong.

**Do not use it for benchmark-driven questions.** Everything in the section above about thermal drift, wrong
job types, canaries and ABAB interleaving applies with more force to an automated loop, because it will
happily run a hundred iterations against a drifting box and hand back a confident wrong answer. A search over
kernels needs the canary *inside* the harness — rejecting any sample where an untouched path moved — before
it is worth starting.

**A search cannot invent a mechanism, only fit one.** The generator's three biggest corrections came from
reading a table and asking why a column was arithmetically impossible: latency quantiles were one series
scaled by a constant, so they could not have different relative scatter; a uniform draw has a range of
exactly twice its interquartile spread while the lab's CPU sits at 3.3x; gen2 heap is a staircase with an
interquartile spread of exactly zero. None is reachable by moving a number. **A missing degree of freedom
shows up as a bad trade, not as a bad number** — the fitter bought p50's range by wrecking its interquartile
spread, which is what revealed that an additive queueing term had to exist.

**Measured, so do not re-litigate it:** the objective's landscape is rugged (10 random restarts spread 0.064
to 0.115) but a good starting point beat all of them — no restart improved on the shape reached by descending
from structurally-reasoned values. A population-based search is therefore not indicated here; initialisation
dominates. `BurstProbability` came out **unidentifiable** (70% of its allowed range across the optima),
because only its product with the burst factors affects the data.

## Test discipline

Read `Tests/README.md` and `Tests/LanguageModels/README.md` before adding
tests — the project is **strict** about test runtime.

- `dotnet test -c Release` must stay fast and contain only correctness checks.
- Anything long-running uses `[LongFact]` (defined in `Tests/LongFact.cs`, a
  `FactAttribute` subclass that auto-sets `Skip` so the test is skipped by
  default). To run a `[LongFact]` locally, temporarily flip it back to `[Fact]`.
  In scope for `[LongFact]`: integration tests that load real models from
  `c:\qwen3b\*` (GGUF / binary), training/checkpoint demos, profilers,
  PyTorch-parity diagnostics, RAM diagnostics, anything 10s+ on the dev box.
  **Out of scope** — keep these as `[Fact(Skip = "...")]` with the specific
  reason: bug-tracker skips ("pending optimization guard", "numerical
  instability"), flaky timing tests. The point of `[Fact(Skip=...)]` vs
  `[LongFact]` is preserving *why* it's skipped.
- Diagnostics / profilers live in `Tests/**/Diagnostics/` and are skipped by
  default via `[LongFact]`.
- Test layout is by domain first, purpose second:
  `Core/`, `DeepLearning/`, `LanguageModels/{GPT1,Runtime,Tokenization,Demo,Experimental,Diagnostics}`,
  `Data/`, `Evolutionary/`, `Forecasting/`, `Preprocessing/`, `Integrations/`,
  `Diagnostics/`, `Examples/`, `TestSupport/`. Keep one public class per file
  and name it after the subject under test.
- Test fixtures live under `Tests/test_fixtures/` and are copied to output via
  `<None Include="test_fixtures\**\*" CopyToOutputDirectory="PreserveNewest" />`.
  `Tests/Usings.cs` provides `global using Xunit;` — don't repeat it per file.
