// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.Schemas;

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Grades the EVAL, not the run. A skill's eval is <b>overfitted</b> when it rewards the agent for repeating
    /// the skill's phrasing, syntax or methodology rather than for producing a genuinely better outcome it could
    /// NOT have produced without the skill. A high pass rate on an overfitted eval proves memorisation, not value
    /// — so this is the check that keeps <see cref="SkillEvaluator"/>'s lift honest.
    ///
    /// <para>The judge classifies every rubric criterion (<c>outcome</c> / <c>technique</c> / <c>vocabulary</c>)
    /// and every declared check (<c>broad</c> / <c>narrow</c>), then scores 0..1. Those classifications are
    /// pinned by a <see cref="JsonSchemaConstraint"/> <b>string enum</b>, so the judge physically cannot emit an
    /// out-of-vocabulary label or unparseable JSON — no parse-retry loop, and a small local judge stays on-rails.</para>
    ///
    /// <para><b>Informational, never a gate.</b> The thresholds are uncalibrated; treat the score as a design
    /// warning on the eval and keep it out of pass/fail. It needs only the SKILL.md and the eval definition,
    /// never a run result, so it can go in parallel with the scenarios.</para>
    /// </summary>
    public static class OverfittingJudge
    {
        /// <summary>Marks an item the judge would not classify even after the repair pass. Such items are
        /// EXCLUDED from the score (numerator and denominator) rather than counted as harmless.</summary>
        internal const string Unclassified = "unclassified";

        /// <summary>Skill markdown longer than this is truncated (the tail rarely changes the classification).</summary>
        private const int MaxSkillChars = 48_000;

        /// <summary>
        /// The judge's output contract. Authored as a real file — <c>Schemas/OverfittingJudge.json</c> — and
        /// woven into this assembly at build time as a <c>const</c> by the <c>EmbedJsonSchemas</c> task in
        /// Main.csproj. Edit the .json, never this. It is NOT an <c>EmbeddedResource</c>: reading one needs
        /// <c>Assembly.GetManifestResourceStream</c> (System.Reflection), which BannedSymbols.txt forbids because
        /// reflection breaks Native AOT — a const ships in the same .dll for free and costs nothing at runtime.
        ///
        /// <para>Items are identified by <c>index</c>, not by echoing their text: it is exact (a judge that
        /// paraphrases can't be matched back), it is cheaper (no long strings echoed), and it makes a SKIPPED
        /// item detectable — which the schema itself cannot prevent, because the schema compiler ignores
        /// <c>minItems</c>. The two <c>enum</c>s close the classification vocabulary by construction.</para>
        /// </summary>
        internal const string JudgeSchema = OverfitSchemas.OverfittingJudge;

        private const string Framing =
            "You assess whether an AI skill's EVALUATION DEFINITION is overfitted.\n\n"
            + "A skill teaches an LLM something new. The eval tests whether loading the skill produces better\n"
            + "outcomes. An eval is OVERFITTED when it rewards the agent for repeating the skill's specific\n"
            + "phrasing, syntax or methodology rather than for a genuinely better result.\n\n"
            + "Ask TWO questions of every criterion and check:\n"
            + "1. DOMAIN EXPERT TEST: would a knowledgeable developer who gives a correct, high-quality answer\n"
            + "   but has NOT read this skill FAIL it? If yes -> overfitted.\n"
            + "2. LLM KNOWLEDGE TEST: does it test knowledge the model already has (common APIs, standard\n"
            + "   syntax, shell escaping, general best practice)? If yes -> overfitted.\n\n"
            + "IMPORTANT: if the skill teaches a genuinely novel technique with NO practical alternative, testing\n"
            + "for it is NOT overfitting - it is testing the outcome.\n\n"
            + "Rubric classifications:\n"
            + "  outcome    - tests WHAT was achieved; a different valid approach also scores well.\n"
            + "  technique  - tests a specific METHOD or diagnostic STEP from the skill, not a finding.\n"
            + "  vocabulary - tests the skill's exact terminology, labels or syntax.\n"
            + "Check classifications:\n"
            + "  broad  - multiple valid approaches pass.\n"
            + "  narrow - only the skill's specific pattern passes.\n\n"
            + "EXAMPLES\n"
            + "- vocabulary (HIGH): \"Measured cold, warm and no-op builds\" -> a dev measuring \"clean/cached/null\"\n"
            + "  does the same thing; the eval tests the skill's labels. Better: \"Established a reproducible\n"
            + "  baseline across cache states\".\n"
            + "- technique (MODERATE): \"Built twice to check incrementality\" -> tests the diagnostic STEP; an agent\n"
            + "  that inspects the binlog directly answers just as well. Better: \"Identified the root cause\".\n"
            + "- narrow (HIGH): a check for the exact flag `--clreventlevel` -> another profiler achieving the same\n"
            + "  trace fails. Better: a check that a scoped trace was produced.\n"
            + "- outcome (NOT overfitted): \"Used /bl:{} so each build writes a unique binlog\" -> there is no\n"
            + "  practical alternative, so this IS the outcome.\n\n"
            + "Each item below is NUMBERED. Report each one using its exact number in \"index\".\n"
            + "You MUST return one entry for EVERY numbered item - do not skip any, do not merge any.\n"
            + "Keep every \"reasoning\" to ONE short sentence (15 words max) - long replies get truncated and\n"
            + "the whole assessment is then discarded.\n"
            + "confidence is an INTEGER 0-100. overall_overfitting_score is an INTEGER 0-100 (higher = more\n"
            + "overfitted). Reply ONLY with JSON.";

        /// <summary>
        /// Runs the judge over a skill and its eval definition and returns the classification + score.
        /// Never throws on a bad reply (this signal is advisory): an unusable reply yields a Low/0 result whose
        /// <see cref="OverfittingResult.OverallReasoning"/> explains what happened.
        /// </summary>
        /// <param name="judge">Judge model — prefer one stronger than the model under test.</param>
        /// <param name="skillMarkdown">The SKILL.md content being evaluated (truncated past 48k chars).</param>
        /// <param name="rubricCriteria">The rubric criteria the eval grades against (the primary signal).</param>
        /// <param name="cases">The eval cases; their declared <see cref="SkillEvalCase.ExpectedChecks"/> are the checks.</param>
        public static OverfittingResult Analyze(
            OverfitClient judge,
            string skillMarkdown,
            IReadOnlyList<string> rubricCriteria,
            IReadOnlyList<SkillEvalCase> cases)
        {
            ArgumentNullException.ThrowIfNull(judge);
            ArgumentNullException.ThrowIfNull(rubricCriteria);
            ArgumentNullException.ThrowIfNull(cases);

            var checks = CollectChecks(cases);
            var skill = Truncate(skillMarkdown ?? string.Empty);

            var rubric = new OverfittingResult.RubricAssessment?[rubricCriteria.Count];
            var assertions = new OverfittingResult.AssertionAssessment?[checks.Count];

            var reply = AskAndFill(
                judge, BuildPrompt(skill, rubricCriteria, cases, checks, null, null),
                rubric, assertions, rubricCriteria, checks);

            // An unusable reply (commonly: the judge rambled and the JSON was truncated) must NOT be reported as
            // Low/clean — that is the same falsely-reassuring result this whole check exists to prevent. Say
            // Unknown and explain.
            if (reply.Error != null && Filled(rubric) == 0 && Filled(assertions) == 0)
            {
                return new OverfittingResult(
                    0.0, OverfittingSeverity.Unknown, [], [],
                    "the eval was NOT assessed — the judge returned an unusable reply (" + reply.Error
                    + "). A truncated reply usually means maxNewTokens is too low for this many items, "
                    + "or the judge model is too small to hold the format.");
            }

            // The schema cannot enforce completeness — the compiler ignores minItems — so the judge CAN skip
            // items, and a small one does. Averaging over only the survivors would UNDERSTATE overfitting (the
            // skipped item is often the awkward one), so re-ask for exactly the gaps.
            var missingRubric = MissingIndices(rubric);
            var missingChecks = MissingIndices(assertions);
            if (missingRubric.Count > 0 || missingChecks.Count > 0)
            {
                AskAndFill(
                    judge, BuildPrompt(skill, rubricCriteria, cases, checks, missingRubric, missingChecks),
                    rubric, assertions, rubricCriteria, checks);
            }

            // Anything STILL missing is recorded explicitly as "unclassified" and excluded from the score —
            // never silently defaulted to "outcome", which would manufacture a clean-looking eval.
            var unresolvedRubric = MissingIndices(rubric);
            var unresolvedChecks = MissingIndices(assertions);
            MarkUnclassifiedRubric(rubric, rubricCriteria, unresolvedRubric);
            MarkUnclassifiedChecks(assertions, checks, unresolvedChecks);

            var reasoning = reply.Reasoning;
            if (unresolvedRubric.Count > 0 || unresolvedChecks.Count > 0)
            {
                reasoning =
                    "[INCOMPLETE: the judge did not classify " + unresolvedRubric.Count + " rubric item(s) and "
                    + unresolvedChecks.Count + " check(s), even after a repair pass; they are excluded from the "
                    + "score, so it may understate overfitting.] " + reasoning;
            }

            var score = ComputeScore(rubric, assertions, reply.LlmOverall);

            // If the repair pass also failed and NOTHING ended up classified, the score is meaningless — a 0.0
            // here means "we don't know", never "clean".
            var severity = Classified(rubric) + Classified(assertions) == 0
                ? OverfittingSeverity.Unknown
                : Band(score);

            return new OverfittingResult(score, severity, Freeze(rubric), Freeze(assertions), reasoning);
        }

        private static Reply AskAndFill(
            OverfitClient judge,
            string prompt,
            OverfittingResult.RubricAssessment?[] rubric,
            OverfittingResult.AssertionAssessment?[] assertions,
            IReadOnlyList<string> rubricCriteria,
            IReadOnlyList<string> checks)
        {
            judge.Reset();
            var json = judge.Complete(prompt, constraint: new JsonSchemaConstraint(judge.Tokenizer, JudgeSchema));

            try
            {
                return Fill(json, rubric, assertions, rubricCriteria, checks);
            }
            catch (JsonException ex)
            {
                return new Reply(0.0, string.Empty, ex.Message);
            }
        }

        /// <summary>Internal for tests: parsing/scoring is pure and worth pinning without spinning up a model.</summary>
        internal static Reply Fill(
            string json,
            OverfittingResult.RubricAssessment?[] rubric,
            OverfittingResult.AssertionAssessment?[] assertions,
            IReadOnlyList<string> rubricCriteria,
            IReadOnlyList<string> checks)
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("rubric_assessments", out var ra) && ra.ValueKind == JsonValueKind.Array)
            {
                foreach (var el in ra.EnumerateArray())
                {
                    var i = Index(el);
                    // A first pass wins over a repair pass, and an out-of-range index is dropped rather than
                    // trusted — the judge is not allowed to invent items.
                    if (i < 0 || i >= rubric.Length || rubric[i] != null)
                    {
                        continue;
                    }
                    rubric[i] = new OverfittingResult.RubricAssessment(
                        rubricCriteria[i], Str(el, "classification"), Confidence(el), Str(el, "reasoning"));
                }
            }

            if (root.TryGetProperty("assertion_assessments", out var aa) && aa.ValueKind == JsonValueKind.Array)
            {
                foreach (var el in aa.EnumerateArray())
                {
                    var i = Index(el);
                    if (i < 0 || i >= assertions.Length || assertions[i] != null)
                    {
                        continue;
                    }
                    assertions[i] = new OverfittingResult.AssertionAssessment(
                        checks[i], Str(el, "classification"), Confidence(el), Str(el, "reasoning"));
                }
            }

            var overall = root.TryGetProperty("overall_overfitting_score", out var os) && os.TryGetInt32(out var ov)
                ? Math.Clamp(ov / 100.0, 0.0, 1.0)
                : 0.0;

            return new Reply(overall, Str(root, "overall_reasoning"), null);
        }

        private static string BuildPrompt(
            string skill,
            IReadOnlyList<string> rubricCriteria,
            IReadOnlyList<SkillEvalCase> cases,
            IReadOnlyList<string> checks,
            List<int>? onlyRubric,
            List<int>? onlyChecks)
        {
            var repair = onlyRubric != null || onlyChecks != null;

            var sb = new StringBuilder(Framing.Length + skill.Length + 1024);
            sb.Append(Framing);

            if (repair)
            {
                sb.Append("\n\nYou previously skipped some items. Classify ONLY the numbered items listed below,\n")
                  .Append("keeping their original numbers.");
            }

            sb.Append("\n\n=== SKILL DOCUMENT ===\n<<<SKILL_START>>>\n").Append(skill).Append("\n<<<SKILL_END>>>\n");

            sb.Append("\n=== EVAL DEFINITION ===\n<<<EVAL_START>>>\nRUBRIC CRITERIA:\n");
            AppendNumbered(sb, rubricCriteria, onlyRubric);

            if (!repair)
            {
                sb.Append("\nCASES (context for the criteria):\n");
                for (var i = 0; i < cases.Count; i++)
                {
                    var c = cases[i];
                    sb.Append("- id=").Append(c.Id)
                      .Append(" shouldTrigger=").Append(c.ShouldTrigger ? "true" : "false")
                      .Append(" prompt=\"").Append(c.Prompt).Append("\"\n");
                }
            }

            sb.Append("\nDECLARED CHECKS:\n");
            AppendNumbered(sb, checks, onlyChecks);
            sb.Append("<<<EVAL_END>>>\n");

            return sb.ToString();
        }

        private static void AppendNumbered(StringBuilder sb, IReadOnlyList<string> items, List<int>? only)
        {
            var wrote = false;
            for (var i = 0; i < items.Count; i++)
            {
                if (only != null && !only.Contains(i))
                {
                    continue;
                }
                sb.Append('[').Append(i).Append("] ").Append(items[i]).Append('\n');
                wrote = true;
            }
            if (!wrote)
            {
                sb.Append("(none)\n");
            }
        }

        /// <summary>Distinct check ids across all cases, in first-seen order — the index space for checks.</summary>
        private static List<string> CollectChecks(IReadOnlyList<SkillEvalCase> cases)
        {
            var checks = new List<string>();
            for (var i = 0; i < cases.Count; i++)
            {
                var declared = cases[i].ExpectedChecks;
                if (declared == null)
                {
                    continue;
                }
                for (var j = 0; j < declared.Count; j++)
                {
                    if (!checks.Contains(declared[j]))
                    {
                        checks.Add(declared[j]);
                    }
                }
            }
            return checks;
        }

        private static string Truncate(string skill)
            => skill.Length > MaxSkillChars
                ? skill.Substring(0, MaxSkillChars) + "\n[TRUNCATED at " + MaxSkillChars + " chars]"
                : skill;

        private static List<int> MissingIndices<T>(T?[] slots) where T : class
        {
            var missing = new List<int>();
            for (var i = 0; i < slots.Length; i++)
            {
                if (slots[i] == null)
                {
                    missing.Add(i);
                }
            }
            return missing;
        }

        /// <summary>Items that carry a real classification — <see cref="Unclassified"/> placeholders don't count,
        /// because they represent an item the judge refused, not a verdict on it.</summary>
        private static int Classified(IReadOnlyList<OverfittingResult.RubricAssessment?> rubric)
        {
            var n = 0;
            for (var i = 0; i < rubric.Count; i++)
            {
                if (rubric[i] is { } r && r.Classification != Unclassified)
                {
                    n++;
                }
            }
            return n;
        }

        private static int Classified(IReadOnlyList<OverfittingResult.AssertionAssessment?> assertions)
        {
            var n = 0;
            for (var i = 0; i < assertions.Count; i++)
            {
                if (assertions[i] is { } a && a.Classification != Unclassified)
                {
                    n++;
                }
            }
            return n;
        }

        private static int Filled<T>(T?[] slots) where T : class
        {
            var n = 0;
            for (var i = 0; i < slots.Length; i++)
            {
                if (slots[i] != null)
                {
                    n++;
                }
            }
            return n;
        }

        private static void MarkUnclassifiedRubric(
            OverfittingResult.RubricAssessment?[] rubric, IReadOnlyList<string> criteria, List<int> missing)
        {
            for (var i = 0; i < missing.Count; i++)
            {
                var idx = missing[i];
                rubric[idx] = new OverfittingResult.RubricAssessment(
                    criteria[idx], Unclassified, 0.0, "the judge did not classify this item");
            }
        }

        private static void MarkUnclassifiedChecks(
            OverfittingResult.AssertionAssessment?[] assertions, IReadOnlyList<string> checks, List<int> missing)
        {
            for (var i = 0; i < missing.Count; i++)
            {
                var idx = missing[i];
                assertions[idx] = new OverfittingResult.AssertionAssessment(
                    checks[idx], Unclassified, 0.0, "the judge did not classify this item");
            }
        }

        private static IReadOnlyList<T> Freeze<T>(T?[] slots) where T : class
        {
            var list = new List<T>(slots.Length);
            for (var i = 0; i < slots.Length; i++)
            {
                if (slots[i] is { } v)
                {
                    list.Add(v);
                }
            }
            return list;
        }

        /// <summary>
        /// Weighted per-element score blended with the judge's own holistic number. Per-element dominates (it is
        /// systematic); the holistic view gets a minority vote. Rubric outweighs checks — checks are secondary
        /// gates. Unclassified items are skipped entirely, so they neither inflate nor deflate the average.
        /// </summary>
        internal static double ComputeScore(
            IReadOnlyList<OverfittingResult.RubricAssessment?> rubric,
            IReadOnlyList<OverfittingResult.AssertionAssessment?> assertions,
            double llmOverall)
        {
            var rubricSum = 0.0;
            var rubricCount = 0;
            for (var i = 0; i < rubric.Count; i++)
            {
                if (rubric[i] is not { } r || r.Classification == Unclassified)
                {
                    continue;
                }
                rubricSum += RubricWeight(r.Classification) * r.Confidence;
                rubricCount++;
            }

            var assertionSum = 0.0;
            var assertionCount = 0;
            for (var i = 0; i < assertions.Count; i++)
            {
                if (assertions[i] is not { } a || a.Classification == Unclassified)
                {
                    continue;
                }
                assertionSum += AssertionWeight(a.Classification) * a.Confidence;
                assertionCount++;
            }

            var rubricAvg = rubricCount > 0 ? rubricSum / rubricCount : 0.0;
            var assertionAvg = assertionCount > 0 ? assertionSum / assertionCount : 0.0;

            // With no checks classified, the rubric IS the whole eval — don't let an absent 0.3 term deflate it.
            var computed = assertionCount > 0 ? (0.7 * rubricAvg) + (0.3 * assertionAvg) : rubricAvg;
            return Math.Clamp((0.6 * computed) + (0.4 * llmOverall), 0.0, 1.0);
        }

        private static double RubricWeight(string classification) => classification switch
        {
            "vocabulary" => 1.0,
            "technique" => 0.5,
            _ => 0.0,               // "outcome" — the healthy case
        };

        private static double AssertionWeight(string classification) => classification switch
        {
            "narrow" => 1.0,
            _ => 0.0,               // "broad"
        };

        internal static OverfittingSeverity Band(double score) => score switch
        {
            < 0.20 => OverfittingSeverity.Low,
            < 0.50 => OverfittingSeverity.Moderate,
            _ => OverfittingSeverity.High,
        };

        private static int Index(JsonElement el)
            => el.TryGetProperty("index", out var v) && v.TryGetInt32(out var i) ? i : -1;

        private static string Str(JsonElement el, string name)
            => el.TryGetProperty(name, out var v) ? v.GetString() ?? string.Empty : string.Empty;

        private static double Confidence(JsonElement el)
            => el.TryGetProperty("confidence", out var c) && c.TryGetInt32(out var v)
                ? Math.Clamp(v / 100.0, 0.0, 1.0)
                : 1.0;   // an unstated confidence is a full-weight vote

        /// <summary>One judge reply: its holistic score, its narrative, and a parse error if it was unusable.</summary>
        internal readonly record struct Reply(double LlmOverall, string Reasoning, string? Error);
    }
}
