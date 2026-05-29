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
    /// End-to-end embedding validation on the real sentence-transformers/all-MiniLM-L6-v2 weights:
    /// text → WordPiece → BertEncoder → mean-pool + L2 → vector. Asserts the embeddings behave
    /// semantically (paraphrases score higher than unrelated sentences) and drive the VectorStore
    /// retrieval path. [LongFact] — requires the model at <c>OVERFIT_MINILM_DIR</c> (default C:\minilm).
    ///
    /// Optionally checks bit-parity against a reference vector if
    /// <c>minilm_reference_embeddings.json</c> is present in the model dir
    /// (produced by the sentence-transformers Python lib for a fixed sentence).
    /// </summary>
    public sealed class MiniLmEmbeddingEndToEndTests
    {
        private static float Cosine(float[] a, float[] b)
        {
            var dot = 0f;
            for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; }
            return dot; // both are L2-normalized
        }

        [LongFact]
        public void RealMiniLm_ParaphrasesScoreHigherThanUnrelated()
        {
            var dir = TestModelPaths.MiniLm.Dir;
            // Touch the required files so a missing fixture fails with an actionable message.
            TestModelPaths.MiniLm.RequireConfigJsonPath();
            TestModelPaths.MiniLm.RequireVocabPath();
            TestModelPaths.MiniLm.RequireSafetensorsPath();

            using var embedder = SentenceEmbedder.FromPretrained(dir);
            Assert.Equal(384, embedder.Dimension);

            var anchor = embedder.Embed("A man is playing a guitar.");
            var paraphrase = embedder.Embed("Someone is playing a musical instrument.");
            var unrelated = embedder.Embed("The stock market fell sharply today.");

            var simParaphrase = Cosine(anchor, paraphrase);
            var simUnrelated = Cosine(anchor, unrelated);

            Assert.True(
                simParaphrase > simUnrelated + 0.15f,
                $"paraphrase cosine {simParaphrase:F3} should clearly exceed unrelated {simUnrelated:F3}");
        }

        [LongFact]
        public void RealMiniLm_DrivesVectorStoreRetrieval()
        {
            var dir = TestModelPaths.MiniLm.Dir;
            TestModelPaths.MiniLm.RequireConfigJsonPath();
            TestModelPaths.MiniLm.RequireVocabPath();
            TestModelPaths.MiniLm.RequireSafetensorsPath();

            using var embedder = SentenceEmbedder.FromPretrained(dir);
            var store = new VectorStore(embedder.Dimension);

            embedder.AddTo(store, "guitar", "A man is playing a guitar.");
            embedder.AddTo(store, "cooking", "She is preparing dinner in the kitchen.");
            embedder.AddTo(store, "finance", "The stock market fell sharply today.");

            Span<float> query = new float[embedder.Dimension];
            embedder.Embed("Someone plays a musical instrument.", query);

            var top = store.Search(query, topK: 1);
            Assert.Single(top);
            Assert.Equal("guitar", top[0].Id);
        }

        [LongFact]
        public void RealMiniLm_MatchesReferenceVectorWhenAvailable()
        {
            var dir = TestModelPaths.MiniLm.Dir;
            var refPath = Path.Combine(dir, "minilm_reference_embeddings.json");
            if (!File.Exists(refPath))
            {
                // No reference dropped — semantic tests above cover correctness. Skip parity silently.
                return;
            }

            TestModelPaths.MiniLm.RequireConfigJsonPath();
            using var embedder = SentenceEmbedder.FromPretrained(dir);

            var (sentence, expected) = ReadReference(refPath);
            var got = embedder.Embed(sentence);
            Assert.Equal(expected.Length, got.Length);

            var cos = Cosine(got, expected);
            Assert.True(cos > 0.999f, $"cosine vs sentence-transformers reference {cos:F5} below 0.999 parity bar");
        }

        private static (string Sentence, float[] Vector) ReadReference(string path)
        {
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            var sentence = root.GetProperty("sentence").GetString()!;
            var arr = root.GetProperty("embedding");
            var vec = new float[arr.GetArrayLength()];
            var i = 0;
            foreach (var e in arr.EnumerateArray()) { vec[i++] = e.GetSingle(); }
            return (sentence, vec);
        }
    }
}
