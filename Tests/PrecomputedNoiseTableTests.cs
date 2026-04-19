// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Storage;

namespace DevOnBike.Overfit.Tests
{
    public sealed class PrecomputedNoiseTableTests
    {
        [Fact]
        public void Length_ReturnsRequestedSize()
        {
            var table = new PrecomputedNoiseTable(length: 1024, seed: 42);
            Assert.Equal(1024, table.Length);
        }

        [Fact]
        public void Constructor_ThrowsOnNonPositiveLength()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new PrecomputedNoiseTable(length: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new PrecomputedNoiseTable(length: -1));
        }

        [Fact]
        public void GetSlice_ReturnsExactRequestedWindow()
        {
            var table = new PrecomputedNoiseTable(length: 1024, seed: 42);
            var slice = table.GetSlice(offset: 128, length: 64);

            Assert.Equal(64, slice.Length);
        }

        [Fact]
        public void GetSlice_ThrowsOnOutOfRangeWindow()
        {
            var table = new PrecomputedNoiseTable(length: 1024, seed: 42);

            Assert.Throws<ArgumentOutOfRangeException>(() => table.GetSlice(offset: 1024, length: 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => table.GetSlice(offset: 0, length: 1025));
            Assert.Throws<ArgumentOutOfRangeException>(() => table.GetSlice(offset: 1000, length: 100));
        }

        [Fact]
        public void SampleOffset_AlwaysProducesValidSliceBounds()
        {
            const int length = 4096;
            const int sliceLength = 256;

            var table = new PrecomputedNoiseTable(length, seed: 42);
            var rng = new Random(1);

            for (var trial = 0; trial < 10_000; trial++)
            {
                var offset = table.SampleOffset(rng, sliceLength);
                Assert.InRange(offset, 0, length - sliceLength);

                // The contract: any offset returned must be a legal GetSlice argument
                // for the same sliceLength.
                var slice = table.GetSlice(offset, sliceLength);
                Assert.Equal(sliceLength, slice.Length);
            }
        }

        [Fact]
        public void SampleOffset_ThrowsIfSliceExceedsTable()
        {
            var table = new PrecomputedNoiseTable(length: 128, seed: 42);

            Assert.Throws<ArgumentOutOfRangeException>(() => table.SampleOffset(new Random(1), sliceLength: 129));
        }

        [Fact]
        public void SameSeed_ProducesSameTableContents()
        {
            var a = new PrecomputedNoiseTable(length: 8192, seed: 2024);
            var b = new PrecomputedNoiseTable(length: 8192, seed: 2024);

            var sliceA = a.GetSlice(0, 8192);
            var sliceB = b.GetSlice(0, 8192);

            for (var i = 0; i < 8192; i++)
            {
                Assert.Equal(sliceA[i], sliceB[i], 5);
            }
        }

        [Fact]
        public void DifferentSeed_ProducesDifferentTableContents()
        {
            var a = new PrecomputedNoiseTable(length: 1024, seed: 1);
            var b = new PrecomputedNoiseTable(length: 1024, seed: 2);

            var sliceA = a.GetSlice(0, 1024);
            var sliceB = b.GetSlice(0, 1024);

            // Arrays of 1024 independent N(0,1) draws almost certainly differ in nearly every
            // position; require at least half to differ to be robust under any reasonable RNG.
            var differing = 0;

            for (var i = 0; i < 1024; i++)
            {
                if (Math.Abs(sliceA[i] - sliceB[i]) > 1e-6f)
                {
                    differing++;
                }
            }

            Assert.True(differing > 512);
        }

        [Fact]
        public void Contents_AreApproximatelyStandardNormal()
        {
            // Draw a sizable table and check the empirical mean and variance match
            // N(0, 1) within generous tolerances. Too-tight a tolerance produces a flaky
            // test; too-loose a one lets bugs through. These bounds pass with high margin
            // under correct Box-Muller output.
            const int length = 200_000;

            var table = new PrecomputedNoiseTable(length, seed: 12345);
            var slice = table.GetSlice(0, length);

            double sum = 0;
            double sumSq = 0;

            for (var i = 0; i < length; i++)
            {
                sum += slice[i];
                sumSq += (double)slice[i] * slice[i];
            }

            var mean = sum / length;
            var variance = (sumSq / length) - (mean * mean);

            Assert.InRange(mean, -0.05, 0.05);
            Assert.InRange(variance, 0.95, 1.05);
        }

        [Fact]
        public void GetSlice_IsAllocationStable()
        {
            const int tableLength = 4096;
            const int sliceLength = 128;
            // Mask selected so that offset + sliceLength never exceeds tableLength.
            // The largest legal offset is tableLength - sliceLength = 3968; the next lower
            // power-of-two boundary (2048) gives a clean bitmask that cycles through many
            // distinct, always-legal offsets without a modulo or conditional.
            const int offsetMask = 0x7FF; // [0, 2047]

            var table = new PrecomputedNoiseTable(tableLength, seed: 42);

            // Warmup (JIT, ThreadStatic, etc.).
            _ = table.GetSlice(0, sliceLength);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 10_000; i++)
            {
                var slice = table.GetSlice(i & offsetMask, sliceLength);
                // Touch the slice so the JIT cannot optimize the call away.
                if (slice.Length < 0)
                {
                    throw new InvalidOperationException();
                }
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(allocated <= 64, $"GetSlice allocated {allocated} bytes.");
        }
    }
}