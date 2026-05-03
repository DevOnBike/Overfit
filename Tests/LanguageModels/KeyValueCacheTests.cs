// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    public sealed class KeyValueCacheTests
    {
        [Fact]
        public void Constructor_ExposesShapeAndInitialState()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 2,
                headCount: 4,
                maxSequenceLength: 8,
                headDimension: 16);

            Assert.Equal(2, cache.Shape.LayerCount);
            Assert.Equal(4, cache.Shape.HeadCount);
            Assert.Equal(8, cache.Shape.MaxSequenceLength);
            Assert.Equal(16, cache.Shape.HeadDimension);
            Assert.Equal(0, cache.CurrentLength);
            Assert.Equal(8, cache.MaxLength);
            Assert.False(cache.IsFull);
        }

        [Fact]
        public void Constructor_InvalidShape_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new KeyValueCache(
                    new KeyValueCacheShape(
                        layerCount: 0,
                        headCount: 1,
                        maxSequenceLength: 1,
                        headDimension: 1)));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new KeyValueCache(
                    new KeyValueCacheShape(
                        layerCount: 1,
                        headCount: 0,
                        maxSequenceLength: 1,
                        headDimension: 1)));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new KeyValueCache(
                    new KeyValueCacheShape(
                        layerCount: 1,
                        headCount: 1,
                        maxSequenceLength: 0,
                        headDimension: 1)));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new KeyValueCache(
                    new KeyValueCacheShape(
                        layerCount: 1,
                        headCount: 1,
                        maxSequenceLength: 1,
                        headDimension: 0)));
        }

        [Fact]
        public void WriteAndReadKey_RoundTripsSinglePosition()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 2,
                headCount: 3,
                maxSequenceLength: 4,
                headDimension: 5);

            var write = cache.GetKeyWriteSpan(
                layerIndex: 1,
                headIndex: 2,
                position: 0);

            for (var i = 0; i < write.Length; i++)
            {
                write[i] = 10 + i;
            }

            cache.Advance();

            var read = cache.GetKeyReadSpan(
                layerIndex: 1,
                headIndex: 2,
                fromPosition: 0,
                length: 1);

            Assert.Equal(5, read.Length);

            for (var i = 0; i < read.Length; i++)
            {
                Assert.Equal(10 + i, read[i]);
            }
        }

        [Fact]
        public void WriteAndReadValue_RoundTripsMultiplePositions()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 3);

            var pos0 = cache.GetValueWriteSpan(0, 0, 0);
            pos0[0] = 1;
            pos0[1] = 2;
            pos0[2] = 3;

            var pos1 = cache.GetValueWriteSpan(0, 0, 1);
            pos1[0] = 4;
            pos1[1] = 5;
            pos1[2] = 6;

            cache.Advance(2);

            var read = cache.GetValueReadSpan(
                layerIndex: 0,
                headIndex: 0,
                fromPosition: 0,
                length: 2);

            Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, read.ToArray());
        }

        [Fact]
        public void Advance_UpdatesCurrentLengthAndIsFull()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 3,
                headDimension: 2);

            cache.Advance();

            Assert.Equal(1, cache.CurrentLength);
            Assert.False(cache.IsFull);

            cache.Advance(2);

            Assert.Equal(3, cache.CurrentLength);
            Assert.True(cache.IsFull);
        }

        [Fact]
        public void Advance_BeyondMaxLength_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            cache.Advance(2);

            Assert.Throws<InvalidOperationException>(() =>
                cache.Advance());
        }

        [Fact]
        public void Advance_Negative_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cache.Advance(-1));
        }

        [Fact]
        public void Reset_ClearsLengthAndData()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            var write = cache.GetKeyWriteSpan(0, 0, 0);
            write[0] = 123;
            write[1] = 456;

            cache.Advance();
            cache.Reset();

            Assert.Equal(0, cache.CurrentLength);
            Assert.False(cache.IsFull);

            cache.Advance();

            var read = cache.GetKeyReadSpan(0, 0, 0, 1);

            Assert.Equal(0, read[0]);
            Assert.Equal(0, read[1]);
        }

        [Fact]
        public void GetWriteSpan_InvalidIndex_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cache.GetKeyWriteSpan(1, 0, 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cache.GetKeyWriteSpan(0, 1, 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cache.GetKeyWriteSpan(0, 0, 2));
        }

        [Fact]
        public void GetReadSpan_BeyondCurrentLength_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            cache.Advance(1);

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cache.GetKeyReadSpan(0, 0, 0, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cache.GetValueReadSpan(0, 0, 1, 1));
        }

        [Fact]
        public void GetReadSpan_ZeroLength_IsAllowed()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var read = cache.GetKeyReadSpan(
                layerIndex: 0,
                headIndex: 0,
                fromPosition: 0,
                length: 0);

            Assert.Equal(0, read.Length);
        }

        [Fact]
        public void Dispose_ThenUse_Throws()
        {
            var cache = KeyValueCache.Create(
                layerCount: 1,
                headCount: 1,
                maxSequenceLength: 2,
                headDimension: 2);

            cache.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                cache.Reset());

            Assert.Throws<ObjectDisposedException>(() =>
                cache.GetKeyWriteSpan(0, 0, 0));
        }

        [Fact]
        public void Shape_ComputesElementCounts()
        {
            var shape = new KeyValueCacheShape(
                layerCount: 2,
                headCount: 3,
                maxSequenceLength: 4,
                headDimension: 5);

            Assert.Equal(120, shape.ElementsPerCache);
            Assert.Equal(240, shape.TotalElements);
        }
    }
}
