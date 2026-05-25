// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// The in-process Embeddings API (<see cref="DevOnBike.Overfit.LanguageModels.Runtime.CachedLlamaSession.Embed"/>)
    /// on a real Qwen GGUF: vectors are the right dimension, L2-normalised, deterministic, and
    /// carry enough signal that a semantically related sentence is closer than an unrelated one.
    /// [LongFact] — loads the model.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class EmbeddingsTests
    {
        private readonly ITestOutputHelper _out;
        public EmbeddingsTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Embed_RealQwen_Normalized_Deterministic_SemanticallyOrdered()
        {
            if (!File.Exists(TestModelPaths.Qwen3B.Q4KmGgufPath))
            {
                _out.WriteLine("Qwen Q4_K_M GGUF not present — skipping.");
                return;
            }

            using var engine = GgufLlamaLoader.Load(TestModelPaths.Qwen3B.Q4KmGgufPath);
            var tok = QwenTokenizer.Load(TestModelPaths.Qwen3B.Dir);
            using var session = engine.CreateSession(256);

            float[] Embed(string text) => session.Embed(tok.Encode(text));

            var cat = Embed("The cat sat on the mat.");
            var catAgain = Embed("The cat sat on the mat.");
            var kitten = Embed("A kitten rested on the rug.");
            var physics = Embed("Quantum chromodynamics is a gauge theory of the strong force.");

            // Right dimension + L2-normalised.
            Assert.Equal(session.EmbeddingDimension, cat.Length);
            Assert.True(MathF.Abs(L2(cat) - 1f) < 1e-3f, $"not unit-norm: {L2(cat):F4}");

            // Deterministic: same text → identical vector (cosine == 1).
            Assert.True(Cosine(cat, catAgain) > 0.9999f, $"non-deterministic embed: cos={Cosine(cat, catAgain):F5}");

            var related = Cosine(cat, kitten);
            var unrelated = Cosine(cat, physics);
            _out.WriteLine($"cos(cat, kitten)={related:F4}  cos(cat, physics)={unrelated:F4}");

            // Semantic ordering: the related sentence is closer than the unrelated one.
            Assert.True(related > unrelated, $"semantic ordering failed: related={related:F4} <= unrelated={unrelated:F4}");
        }

        private static float L2(float[] v)
        {
            var s = 0f;
            foreach (var x in v) { s += x * x; }
            return MathF.Sqrt(s);
        }

        private static float Cosine(float[] a, float[] b)
        {
            float dot = 0f, na = 0f, nb = 0f;
            for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
            return dot / (MathF.Sqrt(na) * MathF.Sqrt(nb) + 1e-12f);
        }
    }
}
