// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Mechanics of <see cref="KeyValueCache.Evict"/> (sliding window): the oldest slots
    /// are dropped, the rest shift down contiguously, <see cref="KeyValueCache.BasePosition"/>
    /// tracks how many were evicted, and reads see the surviving window.
    /// </summary>
    public sealed class KeyValueCacheEvictTests
    {
        private const int Layers = 2, KvHeads = 1, MaxSeq = 6, HeadDim = 2;

        [Fact]
        public void Evict_DropsOldestSlots_ShiftsRestDown_AndTracksBasePosition()
        {
            using var cache = KeyValueCache.Create(Layers, KvHeads, MaxSeq, HeadDim);

            // Fill all 6 positions with position-tagged K/V on every layer.
            for (var p = 0; p < MaxSeq; p++)
            {
                for (var l = 0; l < Layers; l++)
                {
                    Write(cache.GetKeyWriteSpan(l, 0, p), p * 10);
                    Write(cache.GetValueWriteSpan(l, 0, p), p * 100);
                }
                cache.Advance();
            }

            Assert.True(cache.IsFull);
            Assert.Equal(6, cache.CurrentLength);
            Assert.Equal(0, cache.BasePosition);

            cache.Evict(2);

            Assert.Equal(4, cache.CurrentLength);
            Assert.Equal(2, cache.BasePosition);

            // Slot s now holds what used to be at position s+2, on every layer.
            for (var l = 0; l < Layers; l++)
            {
                var keys = cache.GetKeyReadSpan(l, 0, fromPosition: 0, length: 4);
                var values = cache.GetValueReadSpan(l, 0, fromPosition: 0, length: 4);
                for (var s = 0; s < 4; s++)
                {
                    Assert.Equal((s + 2) * 10, keys[s * HeadDim]);
                    Assert.Equal((s + 2) * 10 + 1, keys[s * HeadDim + 1]);
                    Assert.Equal((s + 2) * 100, values[s * HeadDim]);
                }
            }
        }

        [Fact]
        public void Evict_ThenAdvanceAgain_NewSlotKeepsAbsolutePosition()
        {
            using var cache = KeyValueCache.Create(1, 1, MaxSeq, HeadDim);
            for (var p = 0; p < MaxSeq; p++) { cache.Advance(); }

            cache.Evict(2);                       // length 4, base 2
            var writeSlot = cache.CurrentLength;  // next physical slot = 4
            cache.Advance();                      // length 5

            // Absolute position of the new token = BasePosition + writeSlot = 2 + 4 = 6.
            Assert.Equal(2, cache.BasePosition);
            Assert.Equal(6, cache.BasePosition + writeSlot);
            Assert.Equal(5, cache.CurrentLength);
        }

        [Fact]
        public void Reset_ClearsBasePosition()
        {
            using var cache = KeyValueCache.Create(1, 1, MaxSeq, HeadDim);
            for (var p = 0; p < MaxSeq; p++) { cache.Advance(); }
            cache.Evict(3);
            Assert.Equal(3, cache.BasePosition);

            cache.Reset();
            Assert.Equal(0, cache.BasePosition);
            Assert.Equal(0, cache.CurrentLength);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(7)]
        public void Evict_InvalidCount_Throws(int count)
        {
            using var cache = KeyValueCache.Create(1, 1, MaxSeq, HeadDim);
            for (var p = 0; p < MaxSeq; p++) { cache.Advance(); }
            Assert.Throws<ArgumentOutOfRangeException>(() => cache.Evict(count));
        }

        private static void Write(Span<float> dst, float baseValue)
        {
            for (var i = 0; i < dst.Length; i++) { dst[i] = baseValue + i; }
        }
    }
}
