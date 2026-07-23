// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// Pins hybrid retrieval end-to-end, and — in <see cref="Search_FindsTheExactIdentifier_WhereDenseSearchAlone"/> —
    /// demonstrates the failure it was built to fix, rather than merely asserting the plumbing works.
    ///
    /// <para>Embeddings here are hand-written rather than produced by a model, so the test is deterministic,
    /// runs in microseconds and needs no fixture. They are shaped to reproduce the real-world case: several
    /// chunks that are near-identical <i>in meaning</i>, exactly one of which carries the identifier the user
    /// actually asked for.</para>
    /// </summary>
    public sealed class HybridRetrieverTests
    {
        [Fact]
        public void Search_FindsTheExactIdentifier_WhereDenseSearchAlone()
        {
            var retriever = new HybridRetriever(dimension: 3);

            // Three chunks on the same topic — cosine-wise almost indistinguishable. Only "target" carries the
            // policy number, and its vector is deliberately the FURTHEST from the query, so dense search ranks
            // it last. This is the everyday enterprise case: the identifier carries the meaning, the prose does not.
            retriever.Add("chatter-a", [1f, 0.02f, 0f], "Termination of cover requires written notice.");
            retriever.Add("chatter-b", [1f, 0.01f, 0f], "Cover may be terminated by the insurer at any time.");
            retriever.Add("target", [1f, 0.00f, 0f], "Policy PL-88-40021 termination schedule and fees.");

            float[] queryVector = [1f, 0.02f, 0f];
            const string queryText = "PL-88-40021";

            // Dense arm alone: ranks by vector proximity, so the document the user asked for comes LAST.
            var denseOnly = retriever.Vectors.Search(queryVector, 3);
            Assert.Equal("chatter-a", denseOnly[0].Id);
            Assert.Equal("target", denseOnly[2].Id);

            // Lexical arm alone: the identifier is unique, so it is the only hit.
            var lexicalOnly = retriever.Lexical.Search(queryText, 3);
            Assert.Equal("target", lexicalOnly[0].Id);

            // Hybrid: the lexical rank-1 outweighs the dense rank-3 and the answer surfaces.
            var hybrid = retriever.Search(queryVector, queryText, topK: 3);
            Assert.Equal("target", hybrid[0].Id);
        }

        [Fact]
        public void Search_StillAnswersPurelySemanticQueries()
        {
            var retriever = new HybridRetriever(dimension: 3);

            // The complement of the test above: no shared vocabulary at all between query and answer, so only
            // the dense arm can find it. Hybrid must not have broken that.
            retriever.Add("cancel", [1f, 0f, 0f], "How to terminate your cover before renewal.");
            retriever.Add("claims", [0f, 1f, 0f], "Submitting a claim after an accident.");
            retriever.Add("billing", [0f, 0f, 1f], "Monthly instalment schedule.");

            var hybrid = retriever.Search([1f, 0f, 0f], "cancelling my policy", topK: 3);

            Assert.Equal("cancel", hybrid[0].Id);
        }

        [Fact]
        public void Add_KeepsBothArmsInSync()
        {
            var retriever = new HybridRetriever(dimension: 2);
            retriever.Add("a", [1f, 0f], "alpha");
            retriever.Add("b", [0f, 1f], "beta");

            Assert.Equal(2, retriever.Count);
            Assert.Equal(2, retriever.Vectors.Count);
            Assert.Equal(2, retriever.Lexical.Count);
        }

        [Fact]
        public void Add_DefaultsThePayloadToTheIndexedText()
        {
            var retriever = new HybridRetriever(dimension: 2);
            retriever.Add("a", [1f, 0f], "the chunk body");

            var hits = retriever.Search([1f, 0f], "chunk", topK: 1);

            Assert.Equal("the chunk body", hits[0].Payload);
        }

        [Fact]
        public void Add_ExplicitPayloadOverridesTheText()
        {
            var retriever = new HybridRetriever(dimension: 2);
            retriever.Add("a", [1f, 0f], "searchable text", payload: "displayed text");

            var hits = retriever.Search([1f, 0f], "searchable", topK: 1);

            Assert.Equal("displayed text", hits[0].Payload);
        }

        [Fact]
        public void Search_EmptyCorpus_ReturnsNothing()
        {
            var retriever = new HybridRetriever(dimension: 2);
            Assert.Empty(retriever.Search([1f, 0f], "anything", topK: 5));
        }

        [Fact]
        public void Search_NeverReturnsMoreThanTheCorpusHolds()
        {
            var retriever = new HybridRetriever(dimension: 2);
            retriever.Add("a", [1f, 0f], "alpha");

            Assert.Single(retriever.Search([1f, 0f], "alpha", topK: 10));
        }

        [Fact]
        public void Search_DeeperCandidatePool_CanPromoteADocumentBothArmsRankLow()
        {
            var retriever = new HybridRetriever(dimension: 2);

            // "agreed" is mid-ranked by BOTH arms; the others are top of exactly one. With a shallow pool it
            // is never seen by fusion, with a deep one its agreement wins — the reason the default pool is 4x topK.
            retriever.Add("dense-top", [1f, 0f], "unrelated wording entirely");
            retriever.Add("agreed", [0.9f, 0.1f], "shared keyword here");
            retriever.Add("lexical-top", [0f, 1f], "shared keyword shared keyword shared keyword");

            var deep = retriever.Search([1f, 0f], "shared keyword", topK: 3, candidatesPerArm: 3);

            Assert.Contains(deep, m => m.Id == "agreed");
        }

        [Fact]
        public void Constructor_WrappingExistingIndexes_UsesThem()
        {
            var vectors = new VectorStore(2);
            var lexical = new Bm25Index();
            vectors.Add("a", [1f, 0f], "payload");
            lexical.Add("a", "alpha", "payload");

            var retriever = new HybridRetriever(vectors, lexical);

            Assert.Equal(1, retriever.Count);
            Assert.Equal("a", retriever.Search([1f, 0f], "alpha", topK: 1)[0].Id);
        }

        [Fact]
        public void Search_RejectsInvalidTopK()
        {
            var retriever = new HybridRetriever(dimension: 2);
            retriever.Add("a", [1f, 0f], "alpha");

            Assert.Throws<ArgumentOutOfRangeException>(() => retriever.Search([1f, 0f], "alpha", topK: 0));
        }
    }
}
