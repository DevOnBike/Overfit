// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// The RAG Stability Harness is itself testable WITHOUT a model: a deterministic fake embedder maps known
    /// phrases to fixed vectors over a tiny one-hot corpus, so retrieval is fully predictable. This is the point
    /// — "RAG is testable": expected-source recall, paraphrase stability, and false-premise traps become plain
    /// assertions a CI run can gate on.
    /// </summary>
    public sealed class RagEvaluatorTests
    {
        private const int Dim = 5;

        // Known phrases → fixed query vectors. Docs live on the first four axes; the 5th axis is "off-corpus"
        // (orthogonal to every document → a false premise).
        private static readonly Dictionary<string, float[]> Vectors = new(StringComparer.Ordinal)
        {
            ["capital of France"] = [1f, 0f, 0f, 0f, 0f],          // → paris
            ["What's the French capital?"] = [0.95f, 0.1f, 0f, 0f, 0f], // → paris (paraphrase)
            ["capital of Germany"] = [0f, 1f, 0f, 0f, 0f],          // → berlin
            ["capital of Atlantis"] = [0f, 0f, 0f, 0f, 1f],         // false premise: orthogonal to all docs
            ["capital of Paris"] = [1f, 0f, 0f, 0f, 0f],            // false premise that still matches a doc
        };

        private static float[] FakeEmbed(string query)
            => Vectors.TryGetValue(query, out var v) ? (float[])v.Clone() : new float[Dim];

        private static VectorStore BuildStore()
        {
            var store = new VectorStore(Dim);
            store.Add("paris", [1f, 0f, 0f, 0f, 0f], "Paris is the capital of France.");
            store.Add("berlin", [0f, 1f, 0f, 0f, 0f], "Berlin is the capital of Germany.");
            store.Add("tokyo", [0f, 0f, 1f, 0f, 0f], "Tokyo is the capital of Japan.");
            store.Add("madrid", [0f, 0f, 0f, 1f, 0f], "Madrid is the capital of Spain.");
            return store;
        }

        [Fact]
        public void EvaluateRetrieval_HitsExpectedSources_AtRankOne()
        {
            var eval = new RagEvaluator(BuildStore(), FakeEmbed);

            var report = eval.EvaluateRetrieval(
                [
                    new RetrievalCase("capital of France", "paris"),
                    new RetrievalCase("capital of Germany", "berlin"),
                ],
                topK: 3);

            Assert.Equal(2, report.Hits);
            Assert.Equal(1.0, report.RecallAtK);
            Assert.Equal(1.0, report.MeanReciprocalRank);          // both expected sources rank 1
            Assert.All(report.Cases, c => Assert.True(c.Hit));
            Assert.Equal("paris", report.Cases[0].RetrievedIds[0]);
        }

        [Fact]
        public void EvaluateRetrieval_Misses_WhenExpectedSourceNotInTopK()
        {
            var eval = new RagEvaluator(BuildStore(), FakeEmbed);

            var report = eval.EvaluateRetrieval([new RetrievalCase("capital of France", "tokyo")], topK: 1);

            Assert.Equal(0, report.Hits);
            Assert.Equal(0.0, report.RecallAtK);
            Assert.False(report.Cases[0].Hit);
            Assert.Equal(0, report.Cases[0].Rank);
        }

        [Fact]
        public void EvaluateParaphraseStability_FlagsBrittleGroup()
        {
            var eval = new RagEvaluator(BuildStore(), FakeEmbed);

            var report = eval.EvaluateParaphraseStability(
                [
                    new ParaphraseGroup("france", "capital of France", "What's the French capital?"), // both → paris
                    new ParaphraseGroup("brittle", "capital of France", "capital of Germany"),         // paris vs berlin
                ],
                topK: 1,
                minJaccard: 0.6);

            var france = report.Groups[0];
            var brittle = report.Groups[1];

            Assert.True(france.IsStable);
            Assert.Equal(1.0, france.MeanJaccard);
            Assert.False(brittle.IsStable);
            Assert.Equal(0.0, brittle.MeanJaccard);
            Assert.Equal(1, report.UnstableCount);
        }

        [Fact]
        public void EvaluateFalsePremise_SafeOnOrthogonalQuery_SprungOnMatch()
        {
            var eval = new RagEvaluator(BuildStore(), FakeEmbed);

            var report = eval.EvaluateFalsePremise(
                [
                    new FalsePremiseCase("capital of Atlantis", "Atlantis isn't in the corpus"), // top score 0 → safe
                    new FalsePremiseCase("capital of Paris"),                                     // matches the paris doc
                ],
                groundedThreshold: 0.5);

            Assert.False(report.Cases[0].Grounded);
            Assert.Equal(0f, report.Cases[0].TopScore);
            Assert.True(report.Cases[1].Grounded);
            Assert.Equal(1, report.TrapsSprung);
        }
    }
}
