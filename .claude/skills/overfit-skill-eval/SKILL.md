---
name: overfit-skill-eval
description: Build and run an eval harness for an agent skill or prompt LOCALLY with Overfit — deterministic (seeded/greedy), offline, zero API cost, with schema-guaranteed rubric grading. Use when asked to test/evaluate/score a skill or prompt, catch prompt regressions, or measure whether a prompt change helped.
---

# Evaluate agent skills locally with Overfit

Use this skill when the user wants to **test a skill or prompt systematically** — "does my prompt still work?",
"score this skill", "catch prompt regressions", "did that change help or hurt?". It mirrors the OpenAI
"eval skills" methodology but runs entirely on a **local Overfit model**: deterministic, offline, no per-eval
API cost, and the rubric grader's JSON is **guaranteed valid** via `JsonSchemaConstraint` (a cloud
`--output-schema` can still emit invalid JSON; ours cannot).

Two grading tiers, both required:
- **Deterministic graders** answer *"did it do the basics?"* — model-free C# predicates over the run result
  (output contains X, the right tool was called, the JSON parses, token/step budget respected). Fast, CI-safe.
- **Rubric grader** answers *"did it follow conventions/quality?"* — a local judge model scores against a
  rubric, output forced to `{overall_pass, score, checks[]}`.

## Instructions

1. **Define success first (before touching the harness).** Split the goal into four buckets and write them
   down: **outcome** (what must be true of the answer), **process** (which tools/steps must run), **style**
   (conventions/format), **efficiency** (token/step budget). If any bucket is unclear, ask the user.

2. **Pick the local model + make it deterministic.** Use `OverfitClient.LoadGguf(path, ...)` (or a running
   `overfit serve` OpenAI endpoint). Grade with **greedy decoding** (temperature 0) or a fixed seed so runs
   are byte-reproducible — this reproducibility is the whole point vs a cloud grader.

3. **Build a small prompt dataset (10–20 cases).** Each case: `id, prompt, should_trigger, expected_checks[]`
   — the case declares WHICH named checks apply (dispatch them from a `CHECK_REGISTRY` keyed by id; this scales
   rules without combinatorial explosion). Include **negative controls** (`should_trigger=false`) so you catch
   over-triggering, not just failures. **The skill `description` is the #1 lever on trigger accuracy** — write
   it in user-intent language, not API/mechanism language, and make trigger cases the bulk of the set.
   Trigger-selection is testable **deterministically** with `ToolCallConstraint` (constrain generation to the
   skill/tool-name enum → the model's pick is a hard, gradeable choice — a cloud grader can't do this reproducibly).

4. **Run each case, capture the trajectory.** For a plain prompt: `ChatSession.Send(prompt)` → capture the
   text + `ChatSession.LastStats` (tokens, tok/s). For an agentic skill: drive it with `ReActAgent` and
   capture the **step/tool-call trajectory** (the local equivalent of OpenAI's JSONL trace). Save each run's
   artifact to disk so a failing case is inspectable.

5. **Write deterministic graders** as small C# predicates returning `(id, pass, notes)` — e.g.
   `output.Contains("...")`, `trajectory.Any(s => s.Tool == "search")`, `JsonDocument.Parse` succeeds,
   `stats.Tokens <= budget`. Keep them model-free (they mirror `RagEvaluator`/`CorpusLinter` in
   `Sources/Main/LanguageModels/Retrieval/Evaluation/` — reuse that style and its fast, deterministic ethos).

6. **Add the rubric grader (model-assisted, schema-locked).** Feed the run's output + a rubric to the judge
   model with `JsonSchemaConstraint` bound to this schema so the reply is always parseable:
   ```json
   { "type": "object",
     "properties": {
       "overall_pass": { "type": "boolean" },
       "score": { "type": "integer" },
       "checks": { "type": "array", "items": {
         "type": "object",
         "properties": { "id": {"type":"string"}, "pass": {"type":"boolean"}, "notes": {"type":"string"} },
         "required": ["id","pass"] } }
     },
     "required": ["overall_pass","score","checks"] }
   ```
   Prefer a stronger local model as the judge than the one under test.

7. **Aggregate + report.** Per-case pass/score/checks + an aggregate (pass rate, mean score, budget stats).
   Persist the report so successive runs are **comparable** — that comparability across runs is the deliverable
   (a single score means nothing; the delta between runs is the signal).

8. **Iterate.** Change the skill/prompt → re-run → compare scores. Only accept a change when the score
   **strictly improves** on a held-out slice (this selection gate is also the core of the planned `SkillOpt`
   self-improvement loop — see `ROADMAP.md`).

## The two local-model levers cloud evals can't afford

- **Mode = the kind of eval, and both are cheap locally.** A cloud grader is forced into stochastic runs
  ("do 3–5 trials", 3–5× the API cost). Locally you pick: **Deterministic** (greedy/seed) = one
  byte-reproducible run → a **regression gate** (impossible on cloud); **Distribution** (temperature, N trials)
  = a spread → a **capability eval**, and N local trials are ~free. State the mode per eval.
- **Baseline/lift arm — run skill-ON vs skill-OFF (same model + seed) and report the DELTA.** Absolute score
  is noise; the lift over the unaided model is the signal, and it's the "retire the skill when the bare model
  matches it" test. Only cheap because inference is local/free — make it the default, not an extra.
- **Capability eval vs regression eval** — a new skill starts with a low pass rate (capability: is it worth
  keeping?); a graduated skill gets a strict no-drop regression gate. Different thresholds, same harness.

## Notes & guardrails

- **Grade outcomes, not paths** — check the answer, accept creative solution routes; reserve trajectory checks
  for "a required tool actually ran", not for enforcing one exact sequence.
- **Determinism is the moat for regression** — always report the decoding config (greedy/seed) with the scores;
  a non-deterministic grade isn't a regression test (but IS the right tool for a capability distribution).
- **Two tiers are both mandatory** — deterministic-only misses quality drift; rubric-only is noisy and can
  rationalize a broken output. Report them separately, never blend into one number.
- **Small-model judge caveat** — rubric grading wants a capable local judge (7B+ ideal); on a tiny model,
  lean harder on deterministic graders and treat the rubric as advisory.
- **Turnkey facade (planned):** a `SkillEvaluator` library facade under
  `Sources/Main/LanguageModels/Skills/Evaluation/` is proposed to package steps 4–7 (`SkillEvalCase[]` +
  graders → `SkillEvalReport`). Until it lands, wire the steps above directly over `OverfitClient` /
  `ReActAgent` / `JsonSchemaConstraint`, or `overfit serve` + a small runner.
