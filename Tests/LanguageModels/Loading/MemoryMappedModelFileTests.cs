// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.Exceptions;
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
                for (var i = 0; i < data.Length; i++)
                {
                    data[i] = (byte)i;
                }
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
                for (var i = 0; i < 256; i++)
                {
                    Assert.Equal((byte)i, all[i]);
                }
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

        /// <summary>
        /// `XC-117`. The length must come from the handle that is actually mapped, never from
        /// <c>FileInfo</c>. On Windows <c>FileInfo.Length</c> reads a reparse point's own
        /// metadata, so it returns 0 through a file symbolic link while the map — built with
        /// <c>capacity: 0</c>, meaning "to the end of the real file" — spans the target's bytes.
        /// That disagreement made every <c>Slice</c> fail its own bounds check with
        /// "exceeds mapped length 0", and `GgufLlamaLoader.LoadEmbedding` was the first casualty.
        /// </summary>
        [Fact]
        public void Length_ThroughFileSymbolicLink_IsTheTargetsRealLength()
        {
            var dir = Path.Combine(Path.GetTempPath(), "overfit-xc117-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);

            try
            {
                var target = Path.Combine(dir, "target.bin");
                var data = new byte[4097]; // deliberately not a multiple of the page size
                for (var i = 0; i < data.Length; i++)
                {
                    data[i] = (byte)i;
                }
                File.WriteAllBytes(target, data);

                var link = Path.Combine(dir, "link.bin");

                try
                {
                    File.CreateSymbolicLink(link, target);
                }
                catch (IOException e)
                {
                    Assert.Skip("this OS/account cannot create a symbolic link: " + e.Message);
                    return;
                }
                catch (UnauthorizedAccessException e)
                {
                    Assert.Skip("this OS/account cannot create a symbolic link: " + e.Message);
                    return;
                }

                using var map = new MemoryMappedModelFile(link);

                Assert.Equal(4097, map.Length);

                // The bounds check must let the last byte through, and the bytes must be the
                // target's. A length of 0 makes this throw; a page-rounded length reads past EOF.
                var tail = map.Slice(4096, 1).Span;
                Assert.Equal(data[4096], tail[0]);
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        /// <summary>
        /// `XC-117`, the other half. The view's capacity is NOT a legal substitute for the file
        /// length: it is rounded up to the page. Measured on this 4097-byte file the view reports
        /// 8192, and on <c>C:\qwen3b\qwen.q4km.gguf</c> it reports 2104934400 against a real
        /// 2104932768. Taking the length from there would let <c>Slice</c> hand out bytes past
        /// the end of the file, which is a quieter defect than the one it replaced.
        /// </summary>
        [Fact]
        public void Length_IsTheExactFileLength_NotThePageRoundedViewCapacity()
        {
            var path = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(path, new byte[4097]);

                using var map = new MemoryMappedModelFile(path);

                Assert.Equal(4097, map.Length);
                Assert.Throws<ArgumentOutOfRangeException>(() => map.Slice(4097, 1));
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        /// A file that really is empty must fail with a message naming the file and suggesting
        /// something a person can act on — not with "exceeds mapped length 0" out of a later
        /// <c>Slice</c>, which names the symptom and never the cause.
        /// </summary>
        [Fact]
        public void EmptyFile_ThrowsFormatExceptionNamingTheFile()
        {
            var path = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(path, []);

                var ex = Assert.Throws<OverfitFormatException>(() => new MemoryMappedModelFile(path));

                Assert.Contains(path, ex.Message, StringComparison.Ordinal);
                Assert.Contains("symbolic link", ex.Message, StringComparison.Ordinal);
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
