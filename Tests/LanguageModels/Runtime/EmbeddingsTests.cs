// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

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

        [FixtureFact(TestFixture.Qwen3BQ4KmGguf, "3s")]
        public void Embed_RealQwen_Normalized_Deterministic_SemanticallyOrdered()
        {

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

        /// <summary>
        /// <b>The guard for `XC-131`, and it is deliberately a NEGATIVE assertion.</b> Embed must pool the
        /// hidden state AFTER the final norm — the tensor HuggingFace calls <c>last_hidden_state</c> and the
        /// one llama.cpp pools — not the pre-norm state that <see cref="CachedLlamaSession.LastHiddenState"/>
        /// exposes. Until 2026-08-27 it pooled the pre-norm one, one identifier away in the same class.
        ///
        /// <para><b>Why the sibling test above did not catch that, and why this one is shaped differently.</b>
        /// It asserts unit norm, determinism, and <c>related &gt; unrelated</c>. All three hold for the wrong
        /// vector: the pre-norm state is a perfectly good direction, it is just not the one anybody means by
        /// an embedding. The oracle was RELATIVE and the defect is ABSOLUTE direction, so no amount of
        /// ranking assertions could see it. This test compares the two candidate tensors against each other
        /// instead, which needs no external reference and therefore runs wherever the ordinary Qwen fixture
        /// is present — not only on a box carrying the 639 MB Qwen3-Embedding file the parity test needs.</para>
        ///
        /// <para>The margin is large because the final norm's per-channel gain is strongly anisotropic, so
        /// the two vectors are different directions rather than a rounding apart. The measured cosine is
        /// printed, so a future reader can see how much room the 0.99 threshold has.</para>
        /// </summary>
        [FixtureFact(TestFixture.Qwen3BQ4KmGguf, "3s")]
        public void Embed_PoolsThePostFinalNormState_NotThePreNormOneLastHiddenStateExposes()
        {
            using var engine = GgufLlamaLoader.Load(TestModelPaths.Qwen3B.Q4KmGgufPath);
            var tok = QwenTokenizer.Load(TestModelPaths.Qwen3B.Dir);
            using var session = engine.CreateSession(256);

            // normalize:false so the comparison is against the raw pooled tensor, not a rescaled copy.
            var pooled = session.Embed(tok.Encode("The cat sat on the mat."), EmbeddingPooling.LastToken, normalize: false);

            // Embed leaves the stack holding the LAST token's state in both snapshots, so LastHiddenState is
            // the pre-norm counterpart of exactly the vector Embed just returned.
            var preNorm = session.LastHiddenState.ToArray();

            var cos = Cosine(pooled, preNorm);
            _out.WriteLine($"cos(pooled, preNorm)={cos:F5}  (must stay below 0.99)");

            Assert.Equal(session.EmbeddingDimension, pooled.Length);
            Assert.True(
                cos < 0.99f,
                $"Embed appears to pool the PRE-final-norm state: cos(pooled, LastHiddenState)={cos:F5}. "
                + "Embeddings must come from the post-norm hidden (see XC-131).");
        }

        private static float L2(float[] v)
        {
            var s = 0f;
            foreach (var x in v)
            {
                s += x * x;
            }
            return MathF.Sqrt(s);
        }

        private static float Cosine(float[] a, float[] b)
        {
            float dot = 0f, na = 0f, nb = 0f;
            for (var i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            return dot / (MathF.Sqrt(na) * MathF.Sqrt(nb) + 1e-12f);
        }
    }
}
