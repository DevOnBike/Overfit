---
name: overfit-spec
description: Spec-driven development for the Overfit engine. Use when starting a new feature, model/op/kernel/loader/runtime change, or any change touching multiple files — writes an Overfit-native spec (architecture path, AOT / zero-alloc / parity gates, git-read-only boundary) and a gated plan BEFORE coding. Prefer this over the generic overfit-spec-driven-development skill inside this repo.
model: opus
color: blue
---

# Overfit Spec-Driven Development

## Overview

Write a short, Overfit-native specification before writing engine code. In this codebase the
expensive mistakes are architectural (mixing the inference and training paths), disciplinary
(a hidden allocation on a hot path, a trim-hostile API reaching the AOT smoketest, a jagged
`float[][]`), and epistemic (shipping a perf "win" that was never measured). The spec exists to
catch those *before* the diff, and to end every change at a **clean/staged tree** — never a commit.

This is the repo-specific counterpart to the generic `overfit-spec-driven-development` skill. When they
disagree, **this one wins inside Overfit** (the generic one has web examples, dangling skill
references, and "commit the spec / test before commit" advice that violates Overfit's git boundary).

## When to use

- A new model family, operator, kernel, loader, or runtime path
- Any change spanning multiple files or crossing the inference/training boundary
- A performance change (these are *always* spec'd — correctness first, then a separate measured pass)
- Anything reachable from `Tests/AotSmokeTest/Program.cs` (widens AOT verification scope)

**When NOT to use:** a one-line fix, a doc typo, a rename, or a self-contained change with obvious
acceptance. A two-line spec (objective + acceptance) is still fine for small things.


## Who owns what — read this before using the workflow below

**This skill is the shared FORMAT and CHECKLIST for a plan. It does not drive the change.** The phases below
describe the shape of the work; each one is owned by an agent, and the transitions between them are owned by
`/overfit-delivery`:

| phase | owner |
|---|---|
| SPECIFY | `overfit-analyst` — problem, goal, users, success metric, scope, acceptance criteria |
| DESIGN | `overfit-architect` — boundaries, execution path, allocation policy, AOT reach, quality parameters |
| TASKS / IMPLEMENT | `overfit-developer` — one task at a time, correctness pass then a separate measured pass |
| VERIFY | `overfit-verifier` then `overfit-reviewer` — does the evidence prove it, and does the diff match the plan |
| STAGE | the user. Never an agent. |

**Use this skill for its section templates, its gate questions and its Overfit-specific checks** — the
execution path, the verification oracle, the AOT reach, the allocation policy. Do not use it as a second
process that advances phases on its own: if this skill and the agent chain both think they are driving,
neither gate means anything.

There is also a generic `overfit-spec-driven-development` skill in this repository whose scope overlaps this one.
**Inside Overfit, this file wins** — the generic one carries web examples and advice that violates the git
boundary here.

## The gated workflow

```
SPECIFY ──▶ DESIGN ──▶ TASKS ──▶ IMPLEMENT ──▶ VERIFY & STAGE
   │          │          │           │              │
 human      human      human    parity-first    stop at staged,
 review     review     review   then perf        report commands
```

Do not advance a phase until the human has validated the current one. Surface assumptions **first**:

```
ASSUMPTIONS:
1. This is an INFERENCE path change (InferenceEngine, caller-owned buffers) — not training.
2. Parity oracle = cosine vs ORT ≥ 0.9999 on the existing fixture at Tests/test_fixtures/…
3. Reachable from AotSmokeTest? No — internal to the decode loop, not added to the smoketest.
→ Correct me now or I proceed on these.
```

### Phase 1 — SPECIFY

Write the spec into **`docs/specs/<slug>-plan.md`** — the same single file `overfit-analyst` and
`overfit-architect` use, never a second document beside it. There is exactly one plan per change; a spec and
a plan that disagree are worse than either alone, and nothing reconciles them once they have separate authors.
If a plan already exists for this change, **add to it** rather than starting one. See `docs/specs/README.md`
for who owns which sections.

Reframe vague asks as **testable success criteria**: not "make decode faster" but "Qwen-3B Q4_K decode ≥ X tok/s best-of-5 on the dev box,
bit-identical output (or cosine ≥ 0.9999 if reassociated), suite still green."

Spec template (keep it short):

```markdown
# Spec: <feature>

## Objective
What & why. Who calls it. What "done" looks like.

## Execution path            ← the single most important line
INFERENCE (InferenceEngine → IInferenceBackend, caller-owned buffers, zero alloc/call)
   — or —
TRAINING (ComputationGraph tape → Backward → Reset; AutogradNode ownership tags)
Mixing the two is the #1 architectural mistake here. State which, and why.

## Verification oracle        ← name it BEFORE coding
- Parity reference: cosine vs ORT/PyTorch/HF (~1.0) │ FD gradient check (abs-diff floor 5e-4 near 0)
  │ byte-parity vs a convert_*.py │ coherent generation on a real model (c:\qwen3b, c:\gpt2, …)
- A/B baseline for any perf change (the validated-correct version you keep as the guard)

## AOT reach
Is this reachable from Tests/AotSmokeTest/Program.cs? If yes: no LINQ/Reflection/Activator/
Expression/Array.Copy/raw ArrayPool.Shared; delegates over reflection; may need to widen the smoketest.

## Allocation policy
Hot path: 0 alloc/call (PooledBuffer<T>/PooledArray/TensorStorage<T>; no ToArray, no jagged float[][],
no Stopwatch.StartNew → ValueStopwatch). Load path: minimise PEAK RAM (Unpooled weights, no scratch byte[]).

## Commands
Build:  dotnet build -c Release
Test:   dotnet test -c Release [--filter FullyQualifiedName~<X>]
AOT:    dotnet publish ./Tests/AotSmokeTest/AotSmokeTest.csproj -c Release -r linux-x64 -p:PublishAot=true -p:TreatWarningsAsErrors=true
Bench:  dotnet run -c Release --project Sources/Benchmark -- --filter "*<X>*"

## Success criteria
Specific, testable. Parity threshold + (for perf) measured best-of-N on BOTH sides.

## Boundaries (see the pre-filled block below)

## Open questions
```

### Phase 2 — DESIGN

A reviewable technical plan. It MUST answer, explicitly:

1. **Which path** (inference vs training) and where the new code lives (`Sources/Main/…`).
2. **New op?** Define the forward, then the backward, and how it will be **FD-verified** before use.
3. **New loader?** Loading is **one-directional** (external → Overfit only — safetensors/GGUF/ONNX/.bin,
   all native, no Python at runtime). **Never** design an Overfit→other-format exporter. Validate on a
   real model, not a synthetic one.
4. **Perf change?** Two passes, never fused: (a) clearest correct version, pinned by a parity test;
   (b) a *separate* measured iteration A/B'd against (a). State the hypothesis and how you'll measure
   (BenchmarkDotNet + MemoryDiagnoser, best-of-N both sides, a canary path to detect a drifting box).
5. **Ownership** (training): tag every `AutogradNode` (`GraphTemporary`/`GraphAuxiliary`/`Parameter`/
   `ExternalBorrowed`/`View`) so `graph.Reset()` reclaims correctly.

### Phase 3 — TASKS

Discrete, dependency-ordered, ≤ ~5 files each. Every task carries acceptance + a verify command:

```markdown
- [ ] Task: <desc>
  - Acceptance: <what must be true>
  - Verify: dotnet test -c Release --filter FullyQualifiedName~<Test>
  - Files: <paths>
```

Correctness tasks come first and pin parity; a perf task is only ever *after* its parity test is green.
Heavy/integration/real-model tests use `[LongFact]` (auto-skipped) — keep the default `dotnet test -c
Release` fast and correctness-only. Bug-tracker/flaky skips stay `[Fact(Skip="specific reason")]`.

### Phase 4 — IMPLEMENT

One task at a time. Match surrounding code (Allman braces, block-scoped namespaces, one top-level type
per file, file header via `.\update-code-headers.cmd`). Keep the validated-correct version as the A/B
baseline for any perf task. Re-run the task's verify before moving on.

### Phase 5 — VERIFY & STAGE (hard stop)

- Full `dotnet build -c Release` and `dotnet test -c Release` green (fast suite).
- If AOT-reachable, the AOT publish is green (or note it can't be run locally without the C++ toolchain).
- For perf: report the measured before/after (best-of-N both sides) and **document negatives honestly** —
  a reverted lever is a valid, valuable outcome (Winograd, AVX-512 decode, OverfitPool all regressed and
  were reverted).
- **STOP at a clean or staged working tree.** Claude is read-only on git/GitHub. Do **not** `git
  commit/push/rebase/reset`, and do **not** run mutating `gh`. Report the exact commands/UI steps for the
  human, and verify after they run them.

## Boundaries — pre-filled for Overfit (paste into every spec)

**Always**
- Build/test with `-c Release` (never Debug).
- Inference hot path allocates 0 B/call; go through `InferenceEngine.Run(input, output)` with caller buffers.
- `PooledBuffer<T>` / `PooledArray` / `TensorStorage<T>` for scratch — raw `ArrayPool<T>.Shared` is RS0030-banned.
- Flat `float[]` (Span-sliced), never jagged `float[][]`. One top-level type per `.cs`.
- Correctness + parity FIRST, then a **separate** measured perf pass; measure, don't assume.

**Ask first**
- Touching the autograd ownership model or the `ComputationGraph`/`InferenceEngine` split.
- Adding a NuGet (CPM in `Directory.Packages.props`; analyzer pkgs must be ≤ the SDK's Roslyn, currently 5.0).
- Changing the ONNX importers, public API surface, or any behaviour on what is nominally a perf change.
- Extending `Tests/AotSmokeTest/Program.cs` (each added type widens AOT scope and may surface trim warnings).

**Never**
- `git commit/push/rebase/reset` or mutating `gh` — the human owns all git/GitHub actions.
- `System.Linq` / `System.Reflection` / `Activator` / `Expression` / `Array.Copy` in `Sources/Main`.
- An Overfit→other-format exporter (loading is one-directional).
- Reference the Redaction Gateway in public README/ROADMAP (on-prem commercial know-how — feature docs only).
- Commit secrets/keystores, or ship/claim a perf win you have not measured on a stable box.

## Red flags

- Writing engine code before the execution path (inference vs training) is stated.
- A perf kernel written before its parity test is green (unverifiable by construction).
- A backward pass shipped without an FD check.
- "It builds green" used as proof — a green build says nothing about parity, allocations, or the AOT guard.
- Reaching for a commit as the next step (that's the human's action, always).

## Verification checklist

- [ ] Spec saved to a file; execution path + verification oracle + AOT reach stated.
- [ ] Boundaries block pasted and any project-specific Ask-first items called out.
- [ ] Human reviewed and approved the spec and the plan.
- [ ] Success criteria are specific and testable (parity threshold; perf = measured best-of-N both sides).
- [ ] Ends at a clean/staged tree with the exact commit commands handed to the human.
