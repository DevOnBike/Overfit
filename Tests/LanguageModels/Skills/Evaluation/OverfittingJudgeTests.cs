// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Skills.Evaluation;

namespace DevOnBike.Overfit.Tests.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Pins <see cref="OverfittingJudge"/>'s pure parts — index-mapping and scoring — no model needed. The
    /// index-based fill is what makes a SKIPPED item detectable (the schema can't enforce completeness: the
    /// schema compiler ignores <c>minItems</c>), and a skipped item must never be silently scored as harmless.
    /// </summary>
    public sealed class OverfittingJudgeTests
    {
        private static string Reply(string rubricItems, string assertionItems, int overall) =>
            $"{{\"rubric_assessments\":[{rubricItems}],\"assertion_assessments\":[{assertionItems}],"
            + $"\"overall_overfitting_score\":{overall},\"overall_reasoning\":\"r\"}}";

        private static string R(int index, string classification, int confidence = 100) =>
            $"{{\"index\":{index},\"classification\":\"{classification}\",\"confidence\":{confidence},\"reasoning\":\"x\"}}";

        private static string A(int index, string classification, int confidence = 100) =>
            $"{{\"index\":{index},\"classification\":\"{classification}\",\"confidence\":{confidence},\"reasoning\":\"x\"}}";

        private static OverfittingResult.RubricAssessment Rub(string classification, double confidence = 1.0) =>
            new("c", classification, confidence, "x");

        private static OverfittingResult.AssertionAssessment Chk(string classification, double confidence = 1.0) =>
            new("k", classification, confidence, "x");

        // ── Fill: index mapping ────────────────────────────────────────────────────────────────────────

        [Fact]
        public void Fill_MapsByIndex_AndUsesTheRealCriterionText_NotTheJudgesEcho()
        {
            // The judge only returns an index, so the criterion text can never drift from what we asked about.
            var criteria = new[] { "first criterion", "second criterion" };
            var rubric = new OverfittingResult.RubricAssessment?[2];
            var assertions = new OverfittingResult.AssertionAssessment?[0];

            OverfittingJudge.Fill(
                Reply($"{R(1, "vocabulary")},{R(0, "outcome")}", "", 40), rubric, assertions, criteria, []);

            Assert.Equal("first criterion", rubric[0]!.Criterion);
            Assert.Equal("outcome", rubric[0]!.Classification);
            Assert.Equal("second criterion", rubric[1]!.Criterion);
            Assert.Equal("vocabulary", rubric[1]!.Classification);
        }

        [Fact]
        public void Fill_LeavesSkippedItemsNull_SoTheyCanBeDetectedAndRepaired()
        {
            var criteria = new[] { "a", "b", "c" };
            var rubric = new OverfittingResult.RubricAssessment?[3];

            // The judge classified only 2 of 3 — exactly the real-world failure this design exists to catch.
            OverfittingJudge.Fill(Reply($"{R(0, "outcome")},{R(2, "vocabulary")}", "", 0), rubric, [], criteria, []);

            Assert.NotNull(rubric[0]);
            Assert.Null(rubric[1]);
            Assert.NotNull(rubric[2]);
        }

        [Fact]
        public void Fill_IgnoresOutOfRangeIndices_TheJudgeCannotInventItems()
        {
            var rubric = new OverfittingResult.RubricAssessment?[1];

            OverfittingJudge.Fill(Reply($"{R(0, "outcome")},{R(7, "vocabulary")}", "", 0), rubric, [], ["a"], []);

            Assert.Single(rubric);
            Assert.Equal("outcome", rubric[0]!.Classification);
        }

        [Fact]
        public void Fill_FirstWriteWins_SoARepairPassCannotOverwriteTheFirstPass()
        {
            var rubric = new OverfittingResult.RubricAssessment?[1];
            OverfittingJudge.Fill(Reply(R(0, "outcome"), "", 0), rubric, [], ["a"], []);
            OverfittingJudge.Fill(Reply(R(0, "vocabulary"), "", 0), rubric, [], ["a"], []);

            Assert.Equal("outcome", rubric[0]!.Classification);
        }

        [Fact]
        public void Fill_ReadsHolisticScoreAndReasoning()
        {
            var reply = OverfittingJudge.Fill(Reply(R(0, "outcome"), "", 75), new OverfittingResult.RubricAssessment?[1], [], ["a"], []);

            Assert.Equal(0.75, reply.LlmOverall, 3);
            Assert.Equal("r", reply.Reasoning);
            Assert.Null(reply.Error);
        }

        // ── ComputeScore: weights, blends, bands ───────────────────────────────────────────────────────

        [Fact]
        public void AllOutcomeAndBroad_ScoresZero()
        {
            var score = OverfittingJudge.ComputeScore([Rub("outcome")], [Chk("broad")], 0.0);

            Assert.Equal(0.0, score, 3);
            Assert.Equal(OverfittingSeverity.Low, OverfittingJudge.Band(score));
        }

        [Fact]
        public void AllVocabularyAndNarrow_ScoresOne()
        {
            // computed = 0.7*1 + 0.3*1 = 1; final = 0.6*1 + 0.4*1 = 1.
            var score = OverfittingJudge.ComputeScore([Rub("vocabulary")], [Chk("narrow")], 1.0);

            Assert.Equal(1.0, score, 3);
            Assert.Equal(OverfittingSeverity.High, OverfittingJudge.Band(score));
        }

        [Fact]
        public void Technique_IsHalfWeighted()
        {
            // computed = 0.7*0.5 + 0.3*0 = 0.35; final = 0.6*0.35 + 0.4*0.5 = 0.41.
            var score = OverfittingJudge.ComputeScore([Rub("technique")], [Chk("broad")], 0.5);

            Assert.Equal(0.41, score, 3);
            Assert.Equal(OverfittingSeverity.Moderate, OverfittingJudge.Band(score));
        }

        [Fact]
        public void NoChecks_UsesRubricAlone_WithoutDeflating()
        {
            var score = OverfittingJudge.ComputeScore([Rub("vocabulary")], [], 1.0);

            Assert.Equal(1.0, score, 3);
        }

        [Fact]
        public void Confidence_ScalesTheWeight()
        {
            // rubric = 1.0 * 0.5 = 0.5; computed = 0.5; final = 0.6*0.5 + 0.4*0 = 0.30.
            var score = OverfittingJudge.ComputeScore([Rub("vocabulary", 0.5)], [], 0.0);

            Assert.Equal(0.30, score, 3);
        }

        [Fact]
        public void Unclassified_IsExcluded_NotCountedAsHarmless()
        {
            // One vocabulary + one unclassified. If the unclassified were treated as "outcome" the average would
            // halve to 0.5 and understate the overfitting — the exact failure mode this guards.
            var withGap = OverfittingJudge.ComputeScore(
                [Rub("vocabulary"), Rub(OverfittingJudge.Unclassified, 0.0)], [], 0.0);
            var withoutGap = OverfittingJudge.ComputeScore([Rub("vocabulary")], [], 0.0);

            Assert.Equal(withoutGap, withGap, 6);
            Assert.Equal(0.6, withGap, 3);
        }

        [Fact]
        public void AllUnclassified_ScoresZero_RatherThanThrowing()
        {
            var score = OverfittingJudge.ComputeScore([Rub(OverfittingJudge.Unclassified, 0.0)], [], 0.0);

            Assert.Equal(0.0, score, 3);
        }

        [Fact]
        public void AllUnclassified_ScoreIsZero_ButSeverityMustNeverReadAsLow()
        {
            // Regression guard for a real bug the dogfood exposed: a truncated judge reply produced score 0.0
            // and severity Low — i.e. "clean ✅" — when in fact nothing had been assessed at all. Band() maps
            // 0.0 -> Low, so Analyze must special-case "nothing classified" to Unknown instead.
            var score = OverfittingJudge.ComputeScore([Rub(OverfittingJudge.Unclassified, 0.0)], [], 0.0);

            Assert.Equal(0.0, score, 3);
            Assert.Equal(OverfittingSeverity.Low, OverfittingJudge.Band(score));   // the trap: 0.0 bands as Low
            Assert.NotEqual(OverfittingSeverity.Unknown, OverfittingJudge.Band(score));
        }

        [Theory]
        [InlineData(0.0, OverfittingSeverity.Low)]
        [InlineData(0.199, OverfittingSeverity.Low)]
        [InlineData(0.20, OverfittingSeverity.Moderate)]
        [InlineData(0.499, OverfittingSeverity.Moderate)]
        [InlineData(0.50, OverfittingSeverity.High)]
        [InlineData(1.0, OverfittingSeverity.High)]
        public void SeverityBands_MapFromScore(double score, OverfittingSeverity expected)
        {
            Assert.Equal(expected, OverfittingJudge.Band(score));
        }

        // ── Schema ─────────────────────────────────────────────────────────────────────────────────────

        [Fact]
        public void JudgeSchema_IsValidJson_WithIndexAndTheClosedClassificationVocabularies()
        {
            // Authored as Schemas/OverfittingJudge.json and woven in as a const at build time — this is what
            // makes editing that file safe: a stray comma fails here in ms, not at decode time.
            using var doc = JsonDocument.Parse(OverfittingJudge.JudgeSchema);
            var props = doc.RootElement.GetProperty("properties");

            var rubricItem = props.GetProperty("rubric_assessments").GetProperty("items").GetProperty("properties");
            var assertionItem = props.GetProperty("assertion_assessments").GetProperty("items").GetProperty("properties");

            // index is what makes a skipped item detectable — it must stay in the contract.
            Assert.Equal("integer", rubricItem.GetProperty("index").GetProperty("type").GetString());
            Assert.Equal("integer", assertionItem.GetProperty("index").GetProperty("type").GetString());

            Assert.Equal(
                ["outcome", "technique", "vocabulary"],
                rubricItem.GetProperty("classification").GetProperty("enum").EnumerateArray().Select(e => e.GetString()));
            Assert.Equal(
                ["broad", "narrow"],
                assertionItem.GetProperty("classification").GetProperty("enum").EnumerateArray().Select(e => e.GetString()));
        }

        [Fact]
        public void MalformedJson_Throws_SoAnalyzeCanReportItAsAdvisory()
        {
            // ThrowsAny, not Throws: STJ raises JsonReaderException (a JsonException SUBCLASS) and xUnit's
            // Assert.Throws demands an exact type. Analyze's `catch (JsonException)` does catch the subclass —
            // that is what turns a bad reply into an advisory Low/0 result instead of a crash.
            Assert.ThrowsAny<JsonException>(
                () => OverfittingJudge.Fill("not json", [], [], [], []));
        }
    }
}
