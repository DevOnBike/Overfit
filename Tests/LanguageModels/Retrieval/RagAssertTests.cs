// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// The assertion façade in action: a RAG evaluation becomes a plain pass/fail gate. A passing gate is silent;
    /// a failing one throws <see cref="RagAssertionException"/> with the offending cases in the message — so a
    /// corpus / chunking / embedder regression turns into a red CI test. Deterministic, no model.
    /// </summary>
    public sealed class RagAssertTests
    {
        private const int Dim = 5;

        private static readonly Dictionary<string, float[]> Vectors = new(StringComparer.Ordinal)
        {
            ["q-paris"] = [1f, 0f, 0f, 0f, 0f],
            ["q-berlin"] = [0f, 1f, 0f, 0f, 0f],
            ["q-false"] = [0f, 0f, 0f, 0f, 1f],   // orthogonal to every doc → ungrounded
        };

        private static float[] Embed(string q) => Vectors.TryGetValue(q, out var v) ? (float[])v.Clone() : new float[Dim];

        private static VectorStore Store()
        {
            var s = new VectorStore(Dim);
            s.Add("paris", [1f, 0f, 0f, 0f, 0f], "Paris is the capital of France and a major European city.");
            s.Add("berlin", [0f, 1f, 0f, 0f, 0f], "Berlin is the capital of Germany.");
            return s;
        }

        [Fact]
        public void RecallAtLeast_Silent_WhenAboveBar()
        {
            var report = new RagEvaluator(Store(), Embed)
                .EvaluateRetrieval([new RetrievalCase("q-paris", "paris")], topK: 1);

            RagAssert.RecallAtLeast(report, 1.0);   // does not throw
        }

        [Fact]
        public void RecallAtLeast_Throws_WithMissedQuery_WhenBelowBar()
        {
            var report = new RagEvaluator(Store(), Embed)
                .EvaluateRetrieval([new RetrievalCase("q-paris", "berlin")], topK: 1);   // expects berlin, gets paris

            var ex = Assert.Throws<RagAssertionException>(() => RagAssert.RecallAtLeast(report, 1.0));
            Assert.Contains("q-paris", ex.Message);
            Assert.Contains("recall", ex.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void Stable_Throws_OnBrittleGroup_NamingIt()
        {
            var report = new RagEvaluator(Store(), Embed)
                .EvaluateParaphraseStability([new ParaphraseGroup("france-intent", "q-paris", "q-berlin")], topK: 1);

            var ex = Assert.Throws<RagAssertionException>(() => RagAssert.Stable(report));
            Assert.Contains("france-intent", ex.Message);
        }

        [Fact]
        public void NoGroundedFalsePremises_Silent_OnUngrounded_Throws_OnSprung()
        {
            var eval = new RagEvaluator(Store(), Embed);

            RagAssert.NoGroundedFalsePremises(eval.EvaluateFalsePremise([new FalsePremiseCase("q-false")], 0.5));   // safe
            Assert.Throws<RagAssertionException>(
                () => RagAssert.NoGroundedFalsePremises(eval.EvaluateFalsePremise([new FalsePremiseCase("q-paris")], 0.5)));
        }

        [Fact]
        public void FindShortDocuments_FlagsFragmentsAndEmpties()
        {
            var s = new VectorStore(Dim);
            s.Add("good", [1f, 0f, 0f, 0f, 0f], "This is a sufficiently long passage that carries an actual answer.");
            s.Add("short", [0f, 1f, 0f, 0f, 0f], "FAQ");
            s.Add("empty", [0f, 0f, 1f, 0f, 0f], null);

            var shortDocs = new CorpusLinter(s).FindShortDocuments(minChars: 40);

            Assert.Contains("short", shortDocs);
            Assert.Contains("empty", shortDocs);
            Assert.DoesNotContain("good", shortDocs);
            Assert.Throws<RagAssertionException>(() => RagAssert.NoShortDocuments(shortDocs));
        }

        [Fact]
        public void NoOrphans_Silent_WhenEmpty_Throws_WhenAny()
        {
            RagAssert.NoOrphans([]);   // no throw
            Assert.Throws<RagAssertionException>(() => RagAssert.NoOrphans(["lonely-doc"]));
        }
    }
}
