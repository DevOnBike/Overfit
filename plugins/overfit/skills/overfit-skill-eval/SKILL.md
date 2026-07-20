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

Plus one check on the **eval itself**, not the run:
- **Overfitting judge** (`OverfittingJudge`) answers *"is this eval measuring skill VALUE, or just memorisation?"*
  A 95% pass rate proves nothing if the rubric rewards parroting the skill's wording. Informational only.

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

7. **Grade the EVAL itself — the overfitting check.** A 95% pass rate proves nothing if the eval rewards the
   agent for parroting the skill's phrasing. `OverfittingJudge.Analyze(judge, skillMarkdown, rubricCriteria, cases)`
   classifies every rubric criterion (`outcome` / `technique` / `vocabulary`) and every declared check
   (`broad` / `narrow`) → a 0..1 score + `OverfittingSeverity` (Low &lt; 0.20 ≤ Moderate &lt; 0.50 ≤ High).
   Two questions drive it: would a domain expert who never read the skill **fail** this item (→ overfitted), and
   does it test what the model **already knows** — common APIs, shell escaping, general best practice
   (→ overfitted)? A genuinely novel technique with **no practical alternative** is NOT overfitting. The
   classification vocabulary is pinned by a `JsonSchemaConstraint` **string enum**, so the judge cannot invent a
   label or emit unparseable JSON — no parse-retry loop. It needs only the SKILL.md + eval definition (never a
   run result) → one call per skill, parallelizable with the scenarios. **Informational only — never gate on it**
   (thresholds are uncalibrated). Act on it by rewriting flagged items toward outcomes: *"identified the root
   cause"* beats *"measured cold/warm/no-op builds"*.

   **Three traps, all found by actually running it — not theory:**
   - **Give it token headroom.** The judge emits one entry per item; a long `reasoning` × N items overruns
     `maxNewTokens`, the JSON is truncated mid-string, and the *entire* assessment is discarded. A 4-criteria
     eval needed **~3000** `maxNewTokens` on a 3B (1200 was silently not enough). Budget proportional to N.
   - **`Unknown` is not `Low`.** If the judge classifies nothing (truncated/unusable reply), you get score `0.0`
     + `OverfittingSeverity.Unknown` — *never* `Low`. A 0.0 that means "we never assessed it" must never render
     as a clean ✅. Items the judge refuses even after the repair pass come back as `unclassified` and are
     excluded from the score (counting them as `outcome` would manufacture a clean eval).
   - **Judge size dominates quality.** On a 3B judge, measured: it reliably catches literal *"use this exact
     phrase"* traps, but it **never once emitted `technique`**, and it false-positived a genuine outcome
     ("references the package and compiles") as `vocabulary`. Use **7B+** and treat a small-judge score as a
     hint to go read the flagged items yourself — not as a number to track.

8. **Aggregate + report.** Per-case pass/score/checks + an aggregate (pass rate, mean score, budget stats).
   Persist the report so successive runs are **comparable** — that comparability across runs is the deliverable
   (a single score means nothing; the delta between runs is the signal).

9. **Iterate.** Change the skill/prompt → re-run → compare scores. Only accept a change when the score
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
- **Never quote a lift without its error bars.** `SkillEvalReport.Significance` (`LiftSignificance`) gives a 95%
  interval + an **exact McNemar p-value**. The design is paired (same case, both arms, same seed), so only the
  DISCORDANT cases carry information — and `Lift == (helped − hurt) / total` exactly. Calibrate your intuition:
  **5 wins out of 5 is p = 0.0625 — NOT significant.** Six straight wins is the minimum for p &lt; 0.05. On a
  10-case set, "+20%" is usually one lucky case. `IsSignificant` is statistical only — apply your own magnitude
  threshold on top ("real" ≠ "big enough to keep").
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
- **A passing eval can still be worthless** — run the `OverfittingJudge` whenever you author or change an eval,
  and fix Moderate/High findings before you trust the lift. But never gate CI on the overfitting score itself:
  it is an uncalibrated design warning about the eval, not a verdict on the skill. Read the flagged ITEMS; the
  number is the least trustworthy part of the output.
- **Trust per-element classifications over the judge's own summary.** Measured on a real run: the judge returned
  `overall_overfitting_score = 0` and the prose *"does not overfit"* while simultaneously classifying an item as
  `vocabulary`. That is why the score is `0.6 × per-element + 0.4 × holistic` — the systematic part must
  dominate the vibe.
- **Turnkey facade:** `SkillEvaluator` + `OverfitSkillRunner` + `CheckRegistry` + `RubricGrader` +
  `OverfittingJudge` ship under `Sources/Main/LanguageModels/Skills/Evaluation/` — `SkillEvalCase[]` + graders →
  `SkillEvalReport` (pass rate + ON/OFF lift + trigger accuracy). Use them directly; drop down to
  `OverfitClient` / `ReActAgent` / `JsonSchemaConstraint`, or `overfit serve` + a small runner, only for
  bespoke flows.
