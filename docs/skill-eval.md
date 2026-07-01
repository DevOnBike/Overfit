# Skill & prompt eval — locally, deterministically, for free

Agent skills and prompts are code, so they should be *tested* like code. Overfit runs the whole eval loop
against a **local model**: deterministic (greedy/seeded → byte-reproducible), offline, zero per-eval API cost,
private, and with a rubric grader whose JSON is **guaranteed valid**. Same "X is testable" line as
[RAG testing](rag-testing.md).

This mirrors the industry methodology (OpenAI's *eval skills*, Phil Schmid's *testing skills*) — a prompt
dataset + two grading tiers — but the local model turns the parts a cloud grader pays for into freebies.

## The two grading tiers (both required)

- **Deterministic graders** — *"did it do the basics?"* Model-free C# predicates over the run: output contains
  X, the JSON parses, a required tool was used, the token budget held. Fast, CI-safe. (`ISkillGrader`.)
- **Rubric grader** — *"is it good / did it follow conventions?"* A local judge model scores the output against
  a free-text rubric, its reply forced to `{overall_pass, score, reason}` by `JsonSchemaConstraint` so it
  **always parses** (a cloud `--output-schema` can still emit invalid JSON). (`RubricGrader`.)

Report them separately — never blend into one number.

## The four levers a cloud eval can't afford (all cheap only locally)

1. **Mode = the kind of eval.** Deterministic (greedy/seed) = a byte-reproducible **regression gate** (cloud
   can't). Distribution (temperature, N trials) = a **capability eval**, and N local trials are ~free —
   resolving the "just run 3–5 trials" tension.
2. **Built-in baseline/lift arm.** Every case runs skill-**ON** and skill-**OFF** (same model + seed); the
   report's `Lift` is `PassRateOn − PassRateOff`. A near-zero lift means the bare model already does this — a
   candidate to *retire*. Only cheap because inference is local, so it's the default, not an extra.
3. **Deterministic trigger measurement.** The skill's `description` drives trigger accuracy, so it's the
   highest-signal check — `OverfitSkillRunner` grades it by constraining the model (via `ToolCallConstraint`)
   to pick the skill vs a `none` tool: a hard, reproducible routing decision graded against `ShouldTrigger`.
4. **Schema-guaranteed rubric** via `JsonSchemaConstraint` — the judge cannot go off-format.

## Anatomy

| Piece | Type | Role |
|-------|------|------|
| `SkillEvalCase` | record | one case: `Prompt`, `ShouldTrigger`, `ExpectedChecks[]` (which checks apply) |
| `ISkillRunner` / `OverfitSkillRunner` | interface / impl | runs a prompt ON+OFF on a local model; measures trigger |
| `ISkillGrader` + `CheckRegistry` | interface + dispatch | id-keyed deterministic checks (scales without per-case explosion) |
| `RubricGrader` | `ISkillGrader` | schema-locked local-judge scoring |
| `SkillEvaluator` → `SkillEvalReport` | static + record | runs cases × [ON,OFF] × graders → pass rates + `Lift` + trigger accuracy |

## Use it

```csharp
using var client = OverfitClient.LoadGguf(modelPath);            // greedy → reproducible

var runner = new OverfitSkillRunner(client,
    skillName: "concise_answer",
    skillDescription: "Answer a short factual question with just the answer.",
    skillInstructions: "Reply with ONLY the direct answer, in as few words as possible.");

var registry = new CheckRegistry()
    .Register(new ShortAnswerGrader())                            // your deterministic ISkillGrader
    .Register(new RubricGrader(client, "Correct AND at most a few words.", "concise_and_correct"));

var cases = new[]
{
    new SkillEvalCase("fr", "Capital of France?", ShouldTrigger: true, new[] { "short_answer", "concise_and_correct" }),
    // … 10–20 cases from real usage, plus negative controls (ShouldTrigger: false) …
};

var report = SkillEvaluator.Evaluate(cases, runner, registry);
Console.WriteLine($"lift={report.Lift:P0} trigger={report.TriggerAccuracy:P0}");
```

Run the turnkey demo:

```powershell
dotnet run -c Release --project Demo/SkillEvalConsole -- C:\qwen3b\qwen.q4km.gguf
```

## Guardrails

- **Determinism is the moat for regression** — report the decoding config (greedy/seed) with the scores; a
  non-deterministic grade isn't a regression test (it *is* the right tool for a capability distribution).
- **Grade outcomes, not paths** — check the answer; reserve trajectory checks for "a required tool ran".
- **Description is lever #1** — write it in user-intent language; make trigger cases (incl. negatives) the bulk
  of the set.
- **Small-model judge caveat** — the rubric wants a capable local judge (7B+ ideal); on a tiny judge, lean on
  the deterministic graders and treat the rubric as advisory.

The `Lift` gate is also the selection criterion for the planned **SkillOpt** self-improvement loop (see
`ROADMAP.md`): accept a skill edit only when it strictly improves the held-out lift.
