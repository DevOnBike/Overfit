// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Embeddings
{
    /// <summary>
    /// End-to-end validation on real BGE-small-en-v1.5 (CLS pool + query instruction) and E5-small-v2
    /// (mean pool + <c>query:</c> / <c>passage:</c> prefixes). Mirrors the MiniLM suite: tokenizer
    /// parity, semantic ordering, VectorStore retrieval, and (optional) bit-parity vs HF/PyTorch when
    /// a reference JSON is present. [LongFact] — requires <c>OVERFIT_BGE_DIR</c> / <c>OVERFIT_E5_DIR</c>.
    /// </summary>
    public sealed class BgeAndE5EmbeddingEndToEndTests
    {
        private static float Cosine(float[] a, float[] b)
        {
            var dot = 0f;
            for (var i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
            }
            return dot; // both L2-normalized
        }

        // ──────────────── BGE-small-en-v1.5 ────────────────

        [LongFact]
        public void RealBge_QueryFindsRelevantPassage()
        {
            TestModelPaths.Bge.RequireConfigJsonPath();
            TestModelPaths.Bge.RequireVocabPath();
            TestModelPaths.Bge.RequireSafetensorsPath();

            using var embedder = SentenceEmbedder.ForBgeEnV15(TestModelPaths.Bge.Dir);
            Assert.Equal(384, embedder.Dimension);
            Assert.NotNull(embedder.QueryPrefix);
            Assert.Null(embedder.PassagePrefix);

            var store = new VectorStore(embedder.Dimension);
            embedder.AddTo(store, "guitar", "A man is playing a guitar.");
            embedder.AddTo(store, "cooking", "She is preparing dinner in the kitchen.");
            embedder.AddTo(store, "finance", "The stock market fell sharply today.");

            Span<float> query = new float[embedder.Dimension];
            embedder.EmbedQuery("Someone plays a musical instrument.", query);

            var top = store.Search(query, topK: 1);
            Assert.Single(top);
            Assert.Equal("guitar", top[0].Id);
        }

        [LongFact]
        public void RealBge_MatchesReferenceVectorWhenAvailable()
        {
            var refPath = Path.Combine(TestModelPaths.Bge.Dir, "bge_reference_embeddings.json");
            if (!File.Exists(refPath))
            {
                return;
            }

            TestModelPaths.Bge.RequireConfigJsonPath();
            using var embedder = SentenceEmbedder.ForBgeEnV15(TestModelPaths.Bge.Dir);

            var (sentence, expected) = ReadReference(refPath);
            var got = embedder.Embed(sentence); // raw, no prefix — matches the reference recipe
            Assert.Equal(expected.Length, got.Length);
            var cos = Cosine(got, expected);
            Assert.True(cos > 0.999f, $"BGE cosine vs HF reference {cos:F5} below 0.999 parity bar");
        }

        // ──────────────── E5-small-v2 ────────────────

        [LongFact]
        public void RealE5_QueryPassagePrefixingDrivesRetrieval()
        {
            TestModelPaths.E5.RequireConfigJsonPath();
            TestModelPaths.E5.RequireVocabPath();
            TestModelPaths.E5.RequireSafetensorsPath();

            using var embedder = SentenceEmbedder.ForE5(TestModelPaths.E5.Dir);
            Assert.Equal(384, embedder.Dimension);
            Assert.Equal("query: ", embedder.QueryPrefix);
            Assert.Equal("passage: ", embedder.PassagePrefix);

            var store = new VectorStore(embedder.Dimension);
            embedder.AddPassageTo(store, "guitar", "A man is playing a guitar.");
            embedder.AddPassageTo(store, "cooking", "She is preparing dinner in the kitchen.");
            embedder.AddPassageTo(store, "finance", "The stock market fell sharply today.");

            Span<float> query = new float[embedder.Dimension];
            embedder.EmbedQuery("Someone plays a musical instrument.", query);

            var top = store.Search(query, topK: 1);
            Assert.Single(top);
            Assert.Equal("guitar", top[0].Id);
        }

        [LongFact]
        public void RealE5_MatchesReferenceVectorWhenAvailable()
        {
            var refPath = Path.Combine(TestModelPaths.E5.Dir, "e5_reference_embeddings.json");
            if (!File.Exists(refPath))
            {
                return;
            }

            TestModelPaths.E5.RequireConfigJsonPath();
            using var embedder = SentenceEmbedder.ForE5(TestModelPaths.E5.Dir);

            var (sentence, expected) = ReadReference(refPath);
            // The reference JSON's `sentence` field already includes the "query: " prefix (that's what the
            // Python script fed the model); use raw Embed here, NOT EmbedQuery (which would double-prefix).
            var got = embedder.Embed(sentence);
            Assert.Equal(expected.Length, got.Length);
            var cos = Cosine(got, expected);
            Assert.True(cos > 0.999f, $"E5 cosine vs HF reference {cos:F5} below 0.999 parity bar");
        }

        private static (string Sentence, float[] Vector) ReadReference(string path)
        {
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            var sentence = root.GetProperty("sentence").GetString()!;
            var arr = root.GetProperty("embedding");
            var vec = new float[arr.GetArrayLength()];
            var i = 0;
            foreach (var e in arr.EnumerateArray())
            {
                vec[i++] = e.GetSingle();
            }
            return (sentence, vec);
        }
    }
}
