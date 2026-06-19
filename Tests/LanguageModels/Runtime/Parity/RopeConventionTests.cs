// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Rope;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Fast, model-free regression guards for the RoPE rotation convention. The 2026-05-29 flagship bug
    /// was that the GGUF loader applied adjacent-pair RoPE to Qwen2 weights that are stored in HF/NEOX
    /// (split-half) layout — invisible at position 0 (identity rotation) but corrupting every later
    /// position, so attention collapsed onto the current token and long prompts produced garbage. These
    /// tests pin the two conventions and the relative-position property that makes RoPE work.
    /// </summary>
    public sealed class RopeConventionTests
    {
        private const int HeadDim = 8;

        /// <summary>The defining RoPE property: dot(rot(q, m), rot(k, n)) depends only on (m - n).</summary>
        [Theory]
        [InlineData(false)] // adjacent-pair
        [InlineData(true)]  // split-half
        public void Rotation_PreservesRelativePosition(bool splitHalf)
        {
            var table = new RopeTable(maxSequenceLength: 64, headDimension: HeadDim, theta: 10_000f, scaling: null, splitHalf: splitHalf);
            Assert.Equal(splitHalf, table.SplitHalf);

            // Two arbitrary head vectors.
            float[] q = [0.3f, -1.1f, 0.7f, 0.2f, -0.5f, 0.9f, 1.3f, -0.4f];
            float[] k = [-0.8f, 0.6f, 0.1f, -1.2f, 0.4f, 0.7f, -0.3f, 1.0f];

            // Same relative distance (m - n = 3) at two different absolute positions must give the same score.
            var s1 = ScoreAt(table, q, k, m: 5, n: 2, splitHalf);
            var s2 = ScoreAt(table, q, k, m: 20, n: 17, splitHalf);
            Assert.True(MathF.Abs(s1 - s2) < 1e-3f, $"relative-position invariance broken: {s1} vs {s2}");

            // A different relative distance must give a different score (rotation actually does something).
            var s3 = ScoreAt(table, q, k, m: 5, n: 0, splitHalf);
            Assert.True(MathF.Abs(s1 - s3) > 1e-4f, "rotation had no effect across different relative distances");
        }

        /// <summary>Adjacent-pair and split-half are genuinely different rotations (so the flag matters).</summary>
        [Fact]
        public void AdjacentAndSplitHalf_Differ()
        {
            var cos = new float[HeadDim / 2];
            var sin = new float[HeadDim / 2];
            for (var i = 0; i < cos.Length; i++)
            {
                cos[i] = MathF.Cos(0.5f * (i + 1));
                sin[i] = MathF.Sin(0.5f * (i + 1));
            }

            float[] v = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
            var adjacent = (float[])v.Clone();
            var splitHalf = (float[])v.Clone();
            RopeKernel.Apply(adjacent, cos, sin, splitHalf: false);
            RopeKernel.Apply(splitHalf, cos, sin, splitHalf: true);

            var anyDifferent = false;
            for (var i = 0; i < v.Length; i++)
            {
                if (MathF.Abs(adjacent[i] - splitHalf[i]) > 1e-5f)
                {
                    anyDifferent = true;
                }
            }
            Assert.True(anyDifferent, "adjacent-pair and split-half rotations produced identical output");
        }

        /// <summary>Position 0 is identity for BOTH conventions — which is exactly why the bug hid there.</summary>
        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void Position0_IsIdentity(bool splitHalf)
        {
            var table = new RopeTable(8, HeadDim, 10_000f, null, splitHalf);
            float[] v = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
            var rotated = (float[])v.Clone();
            RopeKernel.Apply(rotated, table, position: 0);
            for (var i = 0; i < v.Length; i++)
            {
                Assert.True(MathF.Abs(rotated[i] - v[i]) < 1e-5f, $"position 0 not identity at dim {i}");
            }
        }

        /// <summary>
        /// Full wiring guard: loading a real (tiny) GGUF must set <c>GPT1Config.RopeSplitHalf</c> from the
        /// architecture — qwen2 → split-half (the 2026-05-29 fix), llama → adjacent. Fixtures are ~40 KB,
        /// values irrelevant (only arch + metadata + tensor shapes matter). Catches a regression in the
        /// arch→convention mapping OR its wiring, fast, without the 2 GB model.
        /// Regenerate with <c>Scripts/make_tiny_gguf_fixtures.py</c>.
        /// </summary>
        [Theory]
        [InlineData("tiny-qwen2.gguf", true)]
        [InlineData("tiny-llama.gguf", false)]
        public void GgufLoader_SetsRopeSplitHalf_FromArchitecture(string fixture, bool expectedSplitHalf)
        {
            var path = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "gguf", fixture);
            if (!File.Exists(path))
            {
                Assert.Fail($"fixture missing: {path}");
            }

            using var engine = GgufLlamaLoader.Load(path, quantize: false, mmap: false);
            Assert.Equal(expectedSplitHalf, engine.Config.RopeSplitHalf);
        }

        private static float ScoreAt(RopeTable table, float[] q, float[] k, int m, int n, bool splitHalf)
        {
            var qr = (float[])q.Clone();
            var kr = (float[])k.Clone();
            RopeKernel.Apply(qr, table.CosAt(m), table.SinAt(m), splitHalf);
            RopeKernel.Apply(kr, table.CosAt(n), table.SinAt(n), splitHalf);
            var dot = 0f;
            for (var i = 0; i < qr.Length; i++)
            {
                dot += qr[i] * kr[i];
            }
            return dot;
        }
    }
}
