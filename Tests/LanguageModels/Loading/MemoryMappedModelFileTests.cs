// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Fast unit tests for <see cref="MemoryMappedModelFile"/> — the zero-copy seam the
    /// mmap GGUF loader builds on. Verifies slices read the right bytes, bounds are
    /// enforced, and the slices are genuinely memory-mapped (not managed arrays).
    /// </summary>
    public sealed class MemoryMappedModelFileTests
    {
        [Fact]
        public void Slice_ReadsExactBytesAtOffset()
        {
            var path = Path.GetTempFileName();
            try
            {
                var data = new byte[256];
                for (var i = 0; i < data.Length; i++) { data[i] = (byte)i; }
                File.WriteAllBytes(path, data);

                using var map = new MemoryMappedModelFile(path);
                Assert.Equal(256, map.Length);

                var mid = map.Slice(64, 32).Span;
                Assert.Equal(32, mid.Length);
                for (var i = 0; i < mid.Length; i++)
                {
                    Assert.Equal((byte)(64 + i), mid[i]);
                }

                // Whole-file slice round-trips every byte.
                var all = map.Slice(0, 256).Span;
                for (var i = 0; i < 256; i++) { Assert.Equal((byte)i, all[i]); }
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Slice_IsBackedByMemoryMap_NotManagedArray()
        {
            var path = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(path, new byte[128]);
                using var map = new MemoryMappedModelFile(path);

                var mem = map.Slice(0, 64);
                // A managed-array-backed Memory would yield an ArraySegment here; a
                // memory-mapped one does not. This is exactly Q4KWeight.IsMemoryMapped.
                Assert.False(MemoryMarshal.TryGetArray(mem, out _));
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Slice_OutOfRange_Throws()
        {
            var path = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(path, new byte[64]);
                using var map = new MemoryMappedModelFile(path);

                Assert.Throws<ArgumentOutOfRangeException>(() => map.Slice(32, 64));
                Assert.Throws<ArgumentOutOfRangeException>(() => map.Slice(-1, 4));
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Slice_AfterDispose_Throws()
        {
            var path = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(path, new byte[64]);
                var map = new MemoryMappedModelFile(path);
                map.Dispose();

                Assert.Throws<ObjectDisposedException>(() => map.Slice(0, 4));
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
