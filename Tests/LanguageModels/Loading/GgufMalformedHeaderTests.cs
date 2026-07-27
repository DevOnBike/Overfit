// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Pins the contract for a corrupt or hostile GGUF header: **every length and count in the header is
    /// attacker-controlled**, because a `.gguf` is something the user downloads from a model hub. A file that
    /// declares more than it contains must be refused with a catchable
    /// <see cref="OverfitFormatException"/>, before anything is sized from the declared number.
    ///
    /// <para>The failure this guards against is not a crash in Overfit — it is a crash in the <b>host</b>. A
    /// ~30-byte file that declares a billion metadata entries makes the loader ask for a dictionary of that
    /// capacity; the resulting allocation attempt lands in whatever application embedded the engine. Sibling
    /// case, already fixed and guarded by <see cref="GgufNestedArrayDepthTests"/>: nested arrays overflowing
    /// the stack, which is worse still because it cannot be caught at all.</para>
    ///
    /// <para>The check that makes all of these decidable is cheap and exact: <b>a declared count can never
    /// exceed the bytes left in the file</b>, since every element costs at least one byte. That is the same
    /// shape as the existing <c>SafetensorsReader</c> guard (<c>headerLen > stream.Length - 8</c>).</para>
    ///
    /// <para>Sizes here are deliberately large enough to be provably impossible but small enough to fail
    /// fast — a test must not itself try to commit gigabytes to prove a point.</para>
    /// </summary>
    public sealed class GgufMalformedHeaderTests
    {
        private const uint GgufMagic = 0x46554747; // "GGUF"
        private const uint SupportedVersion = 3;
        private const uint Uint8Type = 0;
        private const uint Uint32Type = 4;
        private const uint ArrayType = 9;

        /// <summary>A header claiming <paramref name="metaCount"/> metadata entries and
        /// <paramref name="tensorCount"/> tensors, while supplying none of either.</summary>
        private static byte[] BuildHeader(ulong tensorCount, ulong metaCount)
        {
            using var ms = new MemoryStream();
            using var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

            w.Write(GgufMagic);
            w.Write(SupportedVersion);
            w.Write(tensorCount);
            w.Write(metaCount);

            w.Flush();

            return ms.ToArray();
        }

        [Fact]
        public void MetadataCountLargerThanTheFile_IsRefused_NotAllocated()
        {
            // 24 bytes of file claiming a billion key/value pairs. Each pair needs at least a length prefix,
            // so this is impossible on its face — and must be rejected before the count sizes anything.
            var bytes = BuildHeader(tensorCount: 0, metaCount: 1_000_000_000);

            using var stream = new MemoryStream(bytes);

            Assert.Throws<OverfitFormatException>(() => new GgufReader(stream));
        }

        [Fact]
        public void TensorCountLargerThanTheFile_IsRefused_NotAllocated()
        {
            var bytes = BuildHeader(tensorCount: 1_000_000_000, metaCount: 0);

            using var stream = new MemoryStream(bytes);

            Assert.Throws<OverfitFormatException>(() => new GgufReader(stream));
        }

        [Fact]
        public void StringLengthLargerThanTheFile_IsRefused_NotAllocated()
        {
            // A single metadata key declaring a 100 MB name in a file of 32 bytes. 100 MB is small enough
            // that an unguarded reader would succeed in allocating it — which is the point: the bug is not
            // that the allocation fails, it is that a 32-byte file gets to choose its size.
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                w.Write(GgufMagic);
                w.Write(SupportedVersion);
                w.Write(0ul);   // tensor count
                w.Write(1ul);   // metadata kv count
                w.Write(100_000_000ul); // declared key length
            }

            using var stream = new MemoryStream(ms.ToArray());

            Assert.Throws<OverfitFormatException>(() => new GgufReader(stream));
        }

        [Fact]
        public void TensorDimensionCountLargerThanTheFile_IsRefused_NotAllocated()
        {
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                w.Write(GgufMagic);
                w.Write(SupportedVersion);
                w.Write(1ul);   // one tensor
                w.Write(0ul);   // no metadata

                var name = "t"u8.ToArray();
                w.Write((ulong)name.Length);
                w.Write(name);

                w.Write(500_000_000u);  // declared dimension count
            }

            using var stream = new MemoryStream(ms.ToArray());

            Assert.Throws<OverfitFormatException>(() => new GgufReader(stream));
        }

        [Fact]
        public void MetadataArrayCountLargerThanTheFile_IsRefused_NotAllocated()
        {
            // The element count of an array *value* is a separate number from the KV count checked above, and
            // it is the one that feeds the tokenizer: `tokenizer.ggml.tokens` is a string array with one entry
            // per vocabulary item. A count of two billion asks for a two-billion-element object[] — 16 GB of
            // references — out of ~40 bytes of file.
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                w.Write(GgufMagic);
                w.Write(SupportedVersion);
                w.Write(0ul);   // tensor count
                w.Write(1ul);   // metadata kv count

                var key = "tokenizer.ggml.tokens"u8.ToArray();
                w.Write((ulong)key.Length);
                w.Write(key);

                w.Write(ArrayType);
                w.Write(Uint8Type);         // element type
                w.Write(2_000_000_000ul);   // element count
            }

            using var stream = new MemoryStream(ms.ToArray());

            Assert.Throws<OverfitFormatException>(() => new GgufReader(stream));
        }

        [Fact]
        public void HeaderTruncatedMidEntry_IsRefused_WithACatchableException()
        {
            // An honest header (one metadata entry) whose bytes simply run out — the ordinary corrupt-download
            // case, as opposed to a crafted one. Any catchable exception is acceptable; a hang or a process
            // kill is not.
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                w.Write(GgufMagic);
                w.Write(SupportedVersion);
                w.Write(0ul);
                w.Write(1ul);

                var key = "general.architecture"u8.ToArray();
                w.Write((ulong)key.Length);
                w.Write(key);
                w.Write(Uint32Type);
                // The four bytes of the value are missing.
            }

            using var stream = new MemoryStream(ms.ToArray());

            var thrown = Record.Exception(() => new GgufReader(stream));

            Assert.NotNull(thrown);
            Assert.IsNotType<OutOfMemoryException>(thrown);
        }

        [Fact]
        public void EmptyAndStubFiles_AreRefused_WithACatchableException()
        {
            foreach (var length in new[] { 0, 1, 4, 8, 15 })
            {
                using var stream = new MemoryStream(new byte[length]);

                var thrown = Record.Exception(() => new GgufReader(stream));

                Assert.NotNull(thrown);
                Assert.IsNotType<OutOfMemoryException>(thrown);
            }
        }

        [Fact]
        public void AWellFormedEmptyHeader_IsStillAccepted()
        {
            // The guard must reject impossible counts without rejecting a legitimately empty file — otherwise
            // it is not a validation, it is a different bug.
            var bytes = BuildHeader(tensorCount: 0, metaCount: 0);

            using var stream = new MemoryStream(bytes);
            using var reader = new GgufReader(stream);

            Assert.Empty(reader.Tensors);
            Assert.Empty(reader.Metadata);
        }
    }
}
