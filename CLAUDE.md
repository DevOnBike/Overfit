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

**The scratch file is scratch; the lab helpers are not — `Scripts/lab.py`.** Anything that drives the
anomaly-guard lab (run kubectl, query Prometheus, find the guard or workload pods, inject a fault, replay a
window, run the suite) is written there once and imported:

```python
import sys
sys.path.insert(0, r"D:\Overfit\Scripts")
from lab import kubectl, prom, guard_pod, workload_pods, inject, replay_signals, suite
```

**Written on 2026-08-09 because re-pasting is how a defect propagates.** Every scratch script had its own
copy of the same four helpers, and each copy carried the previous copy's bugs: the broken unpacking
`_, target, _ = kubectl(...)` — which returns two values — was pasted three times in one evening and failed
three times. Three more the module now gets right once: `guard_pod()` returns the **live** pod, because right
after a rollout the terminating one is still listed and its log is empty, which reads as "no cycles" and is
really "wrong pod"; `utc()` converts with `calendar.timegm`, because `mktime - timezone` is an hour out under
DST and made one arm scan a different window than the replay it was checking; and `apply_and_read_back`
exists because `kubectl apply` reports success for a field it dropped.

**It lives under `Scripts/`, not `.claude/`, and that is the point.** The first version was written into
`.claude/`, which `.gitignore` excludes wholesale, and was gone by the next morning — after which an agent
re-derived the port-forward helpers by hand from `run_two_hour_check.py`, which is precisely the
paste-propagation the file exists to prevent. A helper that must survive cannot live somewhere version
control is not looking.

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
- **`do.py` is scratch.** It is rewritten for each task and is gitignored — never put anything in it
  that needs to survive, and never treat its current contents as documentation of anything. What needs to
  survive goes to `Scripts/lab.py`.
- **Filter the output in Python, not with `grep`/`sed`.** `dotnet build` on this solution emits far more
  than fits in a reply; print only errors, the diagnostics you asked for, and the test summary line. When
  a test fails, print the **test name** — twice now a real failure has been lost because the filter kept
  only the summary.
- **Watch the quoting.** Long scripts belong in `do.py` written with `Write`, not squeezed into
  `python -c` — backticks, `$`, `\` and regex character classes get eaten by the shell on the way in.

Repeatable versions of the three most common cycles live in `.claude/commands/` — `/check` (build + full
suite), `/bench <filter>` (benchmark + the measurement traps to check before believing the number), and
`/sweep <OVERFIT0xx>` (inventory every site an analyzer rule flags).

## How an anomaly (`AN-*`, `RS-*`, `PS-*`) task is run, start to finish

**The protocol lives in [`docs/aiops/aiops-task-protocol.md`](docs/aiops/aiops-task-protocol.md) — read it
before starting one.** It carries the nine steps and, for each, the specific incident that produced it;
those incidents are the part that makes the steps stick, and they do not fit here. The summary below is a
reminder of what the steps are, not a substitute for reading them.

**When this applies:** any change to *what the guard detects* — a channel, a binding, a threshold, a rule, a
detector. Not refactors, not a rename. **One task at a time.**

**Where the task lives:** `docs/TASKS.md` is the registry and the only place carrying status;
`docs/aiops/aiops-backlog.md` is domain prose and its rows are commentary, not state.

1. Read the code path that produces the number and **quote the decisive arithmetic** — the task description
   is not a source of truth.
2. Name or create the artefacts first: query/binding, positive, negative and **missing-data** fixtures, with
   the expected output for each.
3. State the premise out loud: what produces this number, in what unit, **what would refute it**.
4. Distinguish `Detected` / `Healthy` / `WarmingUp` / `InsufficientData` / `QueryFailed`. **Absence of series
   is NEVER `Healthy`** — this is the rule the whole subsystem turns on, because a working detector is
   silent almost all the time and so every defect presents as silence.
5. No threshold without a measurement **in the mechanism's unit**.
6. No new metric without proving the workload actually emits it.
7. Run a mutation that should break the test. If it does not fail, the task is not finished.
8. **Both arms**, and check each arm was *capable* of a verdict before reading its result.
9. Read the deployed state back out of the cluster.

The report then carries: changed files, artefacts read, test results, **mutation result**, **silence risk**,
known limitations, and **self-improvement notes** — what cost iterations on this task and what would have
prevented it.

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

## Verify before you answer — never guess a path, a symbol or a structure

**If you lack the precise context, the file, or the command output needed to answer, STOP and run a tool.**
Do not guess. Do not invent a placeholder path. Do not assume a file, a key, a field or a directory exists
because it would be reasonable for it to exist. Verify first, then answer.

This is not caution for its own sake — an invented detail is indistinguishable from a checked one in the
output, so it costs nothing to produce and everything to discover. Three failures on 2026-08-09/10, each
from the same root:

- A design plan was built on "the only caller is `RunPeer`", read rather than resolved.
  `find_references` returns **two** production call sites; the second is the path every customer-added
  channel takes, and the proposed change would have left it untouched.
- A script wrote a note into the JSON key `_comment`. The file's comment key is `"// what this is"`. The
  write silently did nothing, and only a read-back assertion caught it.
- A helper returned an empty pod name after a `kubectl` query failed on stderr while stdout came back
  empty. Nothing checked the return value, and the script looped for six minutes and then reported a
  cluster failure that had not happened.

**Two operational rules follow, and both are cheap:**

1. **Assert the thing you just fetched is non-empty before you build on it.** An empty result and a
   negative answer look identical downstream. `kubectl` in particular reports a malformed query on stderr
   and returns an empty stdout with a zero exit code in some shapes.
2. **When you cannot verify, say so in the answer** — name what you could not check and why. "I did not
   check X" is a usable answer. A confident answer resting on an assumption is not, and nobody downstream
   can tell the difference.

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

`.claude/agents/` holds **eleven** specialised agents, each with its own context and a `memory:` directory
that persists across sessions. The delivery chain is `overfit-analyst` → `overfit-architect` →
`overfit-developer` → `overfit-verifier` → `overfit-reviewer`, and it is gated: the analyst and architect
write **one** plan file in `docs/specs/`, and the developer refuses to write source until the architect has
signed it. `/overfit-delivery` (a skill, because only the main session can dispatch agents and ask the user
questions) runs that chain and enforces the gates. Conditional
specialists: `overfit-perf-claim-auditor` (**sole owner of the verdict on any performance claim** — others
detect and defer), `overfit-ciso` (threat model, supply
chain, disclosure, and — merged in on 2026-08-09 — parser/endpoint/gateway review),
`overfit-packages-update`,
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

**The details — the four ways a benchmark lies here, and the table of reverted "wins" — are in
[`docs/performance-discipline.md`](docs/performance-discipline.md). Read it before your first perf task.**

The rule itself is short:

- **Correctness first, in a separate pass.** Get the maths right, pin it with a parity test (cosine vs ORT,
  an FD gradient check, a known-good output), and only then iterate for speed against that baseline. A fast
  path written before its correctness is proven is unverifiable, and a change that moves behaviour and
  timing at once cannot be A/B-isolated.
- **Write the benchmark before the argument.** A performance claim with no BenchmarkDotNet run behind it is
  a guess, however confident the reasoning sounds. Put both shapes in `Sources/Benchmark` with
  `[Benchmark(Baseline = true)]` on the old one.
- **Every perf change is a hypothesis until measured** — `MemoryDiagnoser`, best-of-N on **both** sides, one
  lever isolated, ABAB interleaving in a single process, and a canary path that tells you whether the box
  moved instead of the code.
- **If a result is backwards, suspect the benchmark before the runtime.** And if it is a flat 1.00, suspect
  that the lever you are toggling is not live — that has happened here.
- **Report negative results.** They are the most valuable output of this work; the linked file lists six,
  including two that were parity-correct and still reverted.

`overfit-perf-claim-auditor` owns the verdict on any performance claim and must not be substituted for.

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
