// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Pins the depth cap on GGUF metadata parsing. Found by OVERFIT022 (direct recursion): the array branch of
    /// the metadata reader recursed once per nesting level, and GGUF lets an array's element type be another
    /// array — so a crafted file nests without limit at ~12 bytes per level. Since a .NET
    /// <c>StackOverflowException</c> cannot be caught, that was an unrecoverable kill of whatever process
    /// embedded Overfit, triggered by a downloaded model file. This is the regression guard for the fix.
    /// </summary>
    public sealed class GgufNestedArrayDepthTests
    {
        private const uint GgufMagic = 0x46554747; // "GGUF"
        private const uint ArrayType = 9;
        private const uint Uint8Type = 0;

        /// <summary>
        /// Builds a minimal GGUF header whose single metadata entry is an array nested
        /// <paramref name="nesting"/> levels deep, innermost holding zero elements.
        /// </summary>
        private static byte[] BuildNestedArrayGguf(int nesting)
        {
            using var ms = new MemoryStream();
            using var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

            w.Write(GgufMagic);
            w.Write(3u);   // version
            w.Write(0ul);  // tensor count
            w.Write(1ul);  // metadata kv count

            var key = "nested"u8.ToArray();
            w.Write((ulong)key.Length);
            w.Write(key);

            w.Write(ArrayType);

            // Each level: element type + count. All but the innermost hold exactly one child array.
            for (var i = 0; i < nesting - 1; i++)
            {
                w.Write(ArrayType);
                w.Write(1ul);
            }

            w.Write(Uint8Type);
            w.Write(0ul);

            w.Flush();
            return ms.ToArray();
        }

        [Fact]
        public void DeeplyNestedArray_ThrowsCatchableFormatException_InsteadOfOverflowingTheStack()
        {
            // Far beyond the cap but nowhere near enough to overflow a real stack — the point is that the
            // reader refuses by contract, not that this particular input happened to be survivable.
            var bytes = BuildNestedArrayGguf(nesting: 5000);
            using var stream = new MemoryStream(bytes);

            // Catchable: the host application can report a bad model file and carry on. That is the whole
            // difference from a stack overflow, which no catch block can intercept.
            var ex = Assert.ThrowsAny<OverfitException>(() =>
            {
                using var reader = new GgufReader(stream);
            });
            Assert.Contains("deep", ex.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void ShallowNestedArray_StillParses()
        {
            // One level of nesting is what real models actually use (arrays of strings / numbers), so the
            // cap must not break them.
            var bytes = BuildNestedArrayGguf(nesting: 2);
            using var stream = new MemoryStream(bytes);

            using var reader = new GgufReader(stream);

            Assert.True(reader.Metadata.ContainsKey("nested"));
        }
    }
}
