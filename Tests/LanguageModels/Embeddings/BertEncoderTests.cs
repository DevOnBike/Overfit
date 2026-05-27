// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Embeddings;

namespace DevOnBike.Overfit.Tests.LanguageModels.Embeddings
{
    /// <summary>
    /// <see cref="BertEncoder"/> structural checks on a tiny randomly-initialized config (no model file):
    /// the forward runs end-to-end, pooling yields a unit-norm vector of the right length, encoding is
    /// deterministic, and distinct inputs map to distinct embeddings. Real-weight parity is a [LongFact]
    /// in the MiniLM end-to-end suite.
    /// </summary>
    public sealed class BertEncoderTests
    {
        private static BertConfig Tiny() => new(
            hiddenSize: 16,
            numLayers: 2,
            numHeads: 2,
            intermediateSize: 32,
            maxPositionEmbeddings: 32,
            vocabSize: 50,
            typeVocabSize: 2,
            layerNormEps: 1e-12f);

        [Fact]
        public void Embed_ProducesUnitNormVectorOfHiddenSize()
        {
            using var enc = new BertEncoder(Tiny());
            var v = enc.Embed(new[] { 2, 7, 19, 3 }); // [CLS]-ish … [SEP]-ish

            Assert.Equal(16, v.Length);
            var norm = 0f;
            foreach (var x in v) { norm += x * x; }
            Assert.True(MathF.Abs(MathF.Sqrt(norm) - 1f) < 1e-4f, $"expected unit norm, got {MathF.Sqrt(norm)}");
        }

        [Fact]
        public void Embed_IsDeterministic()
        {
            using var enc = new BertEncoder(Tiny());
            var a = enc.Embed(new[] { 2, 7, 19, 3 });
            var b = enc.Embed(new[] { 2, 7, 19, 3 });
            Assert.Equal(a, b);
        }

        [Fact]
        public void Embed_DifferentInputsDifferentVectors()
        {
            using var enc = new BertEncoder(Tiny());
            var a = enc.Embed(new[] { 2, 7, 19, 3 });
            var b = enc.Embed(new[] { 2, 11, 23, 3 });

            var same = true;
            for (var i = 0; i < a.Length; i++) { if (MathF.Abs(a[i] - b[i]) > 1e-6f) { same = false; break; } }
            Assert.False(same, "different token sequences should produce different embeddings");
        }

        [Fact]
        public void Embed_LastTokenPooling_Works()
        {
            using var enc = new BertEncoder(Tiny());
            var mean = enc.Embed(new[] { 2, 7, 19, 3 }, EmbeddingPooling.Mean);
            var last = enc.Embed(new[] { 2, 7, 19, 3 }, EmbeddingPooling.LastToken);

            Assert.Equal(16, last.Length);
            var same = true;
            for (var i = 0; i < mean.Length; i++) { if (MathF.Abs(mean[i] - last[i]) > 1e-6f) { same = false; break; } }
            Assert.False(same, "mean and last-token pooling should differ");
        }

        [Fact]
        public void Embed_RejectsEmptySequence()
        {
            using var enc = new BertEncoder(Tiny());
            Assert.Throws<ArgumentException>(() => enc.Embed(Array.Empty<int>()));
        }
    }
}
