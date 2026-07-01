# Skills — evaluate & self-improve agent skills, locally

Prompts and agent skills are code, so this module lets you **test and optimize them like code** — entirely on a
local model: deterministic (greedy/seeded → byte-reproducible), offline, zero per-eval API cost, private, with a
rubric grader whose JSON is **guaranteed valid** (`JsonSchemaConstraint`, unlike a cloud `--output-schema`). It is
the "X is testable" DNA extended from RAG (`../Retrieval/Evaluation`, `docs/rag-testing.md`) to skills.

Two namespaces:

| Namespace | What it does |
|-----------|--------------|
| `Skills.Evaluation` | score a skill/prompt (pass rate, lift, trigger accuracy) |
| `Skills.Optimization` | **SkillOpt** — self-improve a skill's instructions, keeping only held-out gains |

## Evaluation

Two grading tiers, both required — deterministic answers *"did it do the basics?"*, the rubric answers *"is it
good?"*:

| Type | Role |
|------|------|
| `SkillEvalCase` | one case: `Prompt`, `ShouldTrigger`, `ExpectedChecks[]` (which checks apply) |
| `ISkillRunner` / `OverfitSkillRunner` | runs a prompt **ON vs OFF** on a local model; measures trigger via `ToolCallConstraint` |
| `ISkillGrader` + `CheckRegistry` | id-keyed deterministic checks (scales without per-case explosion) |
| `RubricGrader` | schema-locked local-judge scoring (`{overall_pass, score, reason}`) |
| `SkillEvaluator` → `SkillEvalReport` | runs cases × [ON, OFF] × graders → pass rates + **`Lift`** + trigger accuracy |

The four levers a cloud eval can't afford cheaply — all free locally: **(1) mode = eval kind** (deterministic
regression gate *or* an N-trial capability distribution), **(2) a built-in ON/OFF baseline arm** so `Lift` measures
what the skill actually adds ("retire it when the bare model matches it"), **(3) deterministic trigger grading**,
**(4) a schema-guaranteed rubric**.

```csharp
using var client = OverfitClient.LoadGguf(modelPath);            // greedy → reproducible
var runner = new OverfitSkillRunner(client, "concise_answer",
    "Answer with just the answer.", "Reply with ONLY the direct answer, in as few words as possible.");
var registry = new CheckRegistry()
    .Register(new ShortAnswerGrader())                            // your ISkillGrader
    .Register(new RubricGrader(client, "Correct AND at most a few words.", "concise_and_correct"));
var report = SkillEvaluator.Evaluate(cases, runner, registry);
Console.WriteLine($"lift={report.Lift:P0} trigger={report.TriggerAccuracy:P0}");
```

## Optimization (SkillOpt)

Weight-free, text-space self-improvement. Each round: score the current skill on TRAIN → hand its failures to an
`ISkillEditor` (which proposes one bounded revision, and is shown the already-rejected edits so it doesn't loop) →
score that candidate on **held-out VALIDATION** → keep it **only on a strict improvement** (the selection gate =
the anti-overfit guard). A weak editor is safe: bad edits are simply rejected, so the loop **can improve a prompt
but never regress it**.

| Type | Role |
|------|------|
| `SkillOptimizer.Optimize(...)` → `SkillOptResult` | the loop; returns best instructions + validation score + accept/reject trace |
| `ISkillEditor` / `OverfitSkillEditor` | proposes a revision; the real one emits a schema-locked `{reasoning, revised_instructions}` |
| `CaseFailure` | a failing case (prompt + produced output) the editor reasons over |

```csharp
var result = SkillOptimizer.Optimize(
    initialInstructions: "Answer the question.",
    runnerFactory: i => new OverfitSkillRunner(client, "concise_answer", "…", i, measureTrigger: false),
    editor: new OverfitSkillEditor(client),
    registry, trainCases, validationCases, rounds: 3);
Console.WriteLine($"{result.BestValidationScore:P0}: \"{result.BestInstructions}\"");
```

The `Lift`/validation gate is the selection criterion for `SkillOpt` (ROADMAP #2, arXiv:2605.23904).

## Design notes

- **Model-free core.** `ISkillRunner` / `ISkillGrader` / `ISkillEditor` are injected, so `SkillEvaluator` and
  `SkillOptimizer` are unit-tested without a model (fast, deterministic); the real `Overfit*` adapters are
  validated on real Qwen under `[LongFact]`.
- **Determinism is the moat for regression** — always report the decoding config; a non-deterministic grade isn't
  a regression test (it *is* the right tool for a capability distribution).
- **Grade outcomes, not paths.** Reserve trajectory checks for "a required tool ran".
- **Description is trigger lever #1** — write it in user-intent language; make trigger cases (incl. negatives) the
  bulk of the set.
- **Judge/editor/router quality is model-dependent** — a capable model (7B+) gives better rubric verdicts, skill
  routing and edits; on a tiny model lean on the deterministic graders (and the held-out gate keeps SkillOpt safe
  regardless).

## Run it

```powershell
dotnet run -c Release --project Demo/SkillEvalConsole -- C:\qwen3b\qwen.q4km.gguf            # evaluate
dotnet run -c Release --project Demo/SkillEvalConsole -- C:\qwen3b\qwen.q4km.gguf optimize   # SkillOpt
```

Full methodology: [`docs/skill-eval.md`](../../../../docs/skill-eval.md).
