// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Unit tests for <see cref="KeyValueCache"/> in <see cref="KvCacheDType.Q8"/> mode — the per-vector
    /// int8 KV store. Covers the quantize-on-write / dequantize-on-read round-trip (cosine ≈ 1) through
    /// both the int8 read surface and <c>DequantizeKeyRange</c>, slot isolation across (layer, head,
    /// position), the sliding-window <c>Evict</c> shift of both bytes and scales, and the zero-vector edge.
    /// No model — fast.
    /// </summary>
    public sealed class Q8KeyValueCacheTests
    {
        private const int Layers = 2;
        private const int Heads = 2;
        private const int MaxSeq = 8;
        private const int HeadDim = 16;

        private static KeyValueCache CreateQ8()
            => KeyValueCache.Create(Layers, Heads, MaxSeq, HeadDim, KvCacheDType.Q8);

        [Fact]
        public void WriteThenRead_RoundTrips_WithinInt8Error()
        {
            using var cache = CreateQ8();

            var key = Vector(HeadDim, seed: 1.3f);
            var val = Vector(HeadDim, seed: -0.7f);

            cache.WriteKey(1, 0, 0, key);
            cache.WriteValue(1, 0, 0, val);
            cache.Advance();

            AssertRoundTrip(key, cache.GetKeyQuants(1, 0, 0, 1), cache.GetKeyScales(1, 0, 0, 1)[0]);
            AssertRoundTrip(val, cache.GetValueQuants(1, 0, 0, 1), cache.GetValueScales(1, 0, 0, 1)[0]);
        }

        [Fact]
        public void DistinctSlots_DoNotAlias()
        {
            using var cache = CreateQ8();

            // Write a different vector into every (layer, head, position=0..2) slot.
            for (var l = 0; l < Layers; l++)
            {
                for (var h = 0; h < Heads; h++)
                {
                    for (var p = 0; p < 3; p++)
                    {
                        cache.WriteKey(l, h, p, Vector(HeadDim, seed: l * 10 + h * 3 + p + 1));
                    }
                }
            }
            cache.Advance(3);

            // Each slot reads back as its own vector (no cross-slot aliasing).
            for (var l = 0; l < Layers; l++)
            {
                for (var h = 0; h < Heads; h++)
                {
                    for (var p = 0; p < 3; p++)
                    {
                        var expected = Vector(HeadDim, seed: l * 10 + h * 3 + p + 1);
                        AssertRoundTrip(expected, cache.GetKeyQuants(l, h, p, 1), cache.GetKeyScales(l, h, p, 1)[0]);
                    }
                }
            }
        }

        [Fact]
        public void Evict_ShiftsBytesAndScales_KeepingLiveWindowContiguous()
        {
            using var cache = CreateQ8();

            const int n = 5;
            for (var p = 0; p < n; p++)
            {
                cache.WriteKey(0, 1, p, Vector(HeadDim, seed: p + 1));
                cache.WriteValue(0, 1, p, Vector(HeadDim, seed: -(p + 1)));
            }
            cache.Advance(n);

            cache.Evict(2);   // drop positions 0,1 → live window holds old 2,3,4 at slots 0,1,2

            Assert.Equal(3, cache.CurrentLength);
            Assert.Equal(2, cache.BasePosition);
            for (var p = 0; p < 3; p++)
            {
                AssertRoundTrip(Vector(HeadDim, seed: p + 3), cache.GetKeyQuants(0, 1, p, 1), cache.GetKeyScales(0, 1, p, 1)[0]);
                AssertRoundTrip(Vector(HeadDim, seed: -(p + 3)), cache.GetValueQuants(0, 1, p, 1), cache.GetValueScales(0, 1, p, 1)[0]);
            }
        }

        [Fact]
        public void TruncateTo_AndReset_AdjustLength()
        {
            using var cache = CreateQ8();
            cache.Advance(5);
            cache.TruncateTo(2);
            Assert.Equal(2, cache.CurrentLength);

            cache.Reset();
            Assert.Equal(0, cache.CurrentLength);
            Assert.Equal(0, cache.BasePosition);
        }

        [Fact]
        public void ZeroVector_YieldsZeroScaleAndBytes()
        {
            using var cache = CreateQ8();

            cache.WriteKey(0, 0, 0, new float[HeadDim]);
            cache.Advance();

            Assert.Equal(0f, cache.GetKeyScales(0, 0, 0, 1)[0]);
            foreach (var b in cache.GetKeyQuants(0, 0, 0, 1))
            {
                Assert.Equal(0, b);
            }
        }

        [Fact]
        public void DequantizeRange_MatchesInt8ReadSurface()
        {
            using var cache = CreateQ8();
            for (var p = 0; p < 4; p++)
            {
                cache.WriteKey(0, 0, p, Vector(HeadDim, seed: p + 1));
            }
            cache.Advance(4);

            var dq = new float[4 * HeadDim];
            cache.DequantizeKeyRange(0, 0, 0, 4, dq);

            var bytes = cache.GetKeyQuants(0, 0, 0, 4);
            var scales = cache.GetKeyScales(0, 0, 0, 4);
            for (var p = 0; p < 4; p++)
            {
                for (var d = 0; d < HeadDim; d++)
                {
                    Assert.Equal(scales[p] * bytes[p * HeadDim + d], dq[p * HeadDim + d], 6);
                }
            }
        }

        [Fact]
        public void F32Mode_WriteAndDequantize_AreLossless()
        {
            using var cache = KeyValueCache.Create(Layers, Heads, MaxSeq, HeadDim);   // F32 (default)
            var v = Vector(HeadDim, seed: 2.5f);
            cache.WriteKey(0, 0, 0, v);
            cache.Advance();

            var dq = new float[HeadDim];
            cache.DequantizeKeyRange(0, 0, 0, 1, dq);
            for (var d = 0; d < HeadDim; d++)
            {
                Assert.Equal(v[d], dq[d]);   // exact copy — no quantization in F32 mode
            }
        }

        private static float[] Vector(int n, float seed)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++)
            {
                // Deterministic, varied magnitudes (so the quantizer's max-abs scaling is exercised).
                v[i] = MathF.Sin((i + 1) * 0.37f + seed) * (1f + (i % 4));
            }
            return v;
        }

        private static void AssertRoundTrip(float[] original, ReadOnlySpan<sbyte> bytes, float scale)
        {
            var recon = new float[original.Length];
            for (var i = 0; i < original.Length; i++)
            {
                recon[i] = scale * bytes[i];
            }

            // Cosine ≈ 1 (direction preserved) and per-element abs error within one quantization step.
            double dot = 0, na = 0, nb = 0, maxAbsErr = 0;
            for (var i = 0; i < original.Length; i++)
            {
                dot += original[i] * recon[i];
                na += original[i] * original[i];
                nb += recon[i] * recon[i];
                maxAbsErr = Math.Max(maxAbsErr, Math.Abs(original[i] - recon[i]));
            }
            var cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);

            Assert.True(cosine > 0.999, $"cosine {cosine:F5} too low");
            Assert.True(maxAbsErr <= scale + 1e-6, $"maxAbsErr {maxAbsErr} exceeds one quant step {scale}");
        }
    }
}
