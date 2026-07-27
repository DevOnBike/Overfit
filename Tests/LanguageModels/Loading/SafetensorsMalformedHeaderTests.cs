// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;
using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Pins the contract for a corrupt or hostile <c>.safetensors</c> file — the HuggingFace-native weight
    /// format, downloaded like any other model artefact.
    ///
    /// <para>This reader already validates the two things that are obviously lengths: the JSON header size
    /// against the file, and each tensor's declared byte range against its dtype and shape. What those checks
    /// do <b>not</b> tie down is whether the byte range exists at all. A header may name a tensor spanning
    /// bytes [0, 4 TB) with a shape that agrees with it, and both checks pass — the numbers are consistent
    /// with each other, just not with the file.</para>
    ///
    /// <para>That matters because <see cref="SafetensorsReader.ElementCount"/> is what a caller sizes its
    /// buffer from; the class documentation itself shows <c>new float[info.ElementCount]</c>. A reader that
    /// reports a trillion elements for a 200-byte file has handed the caller the allocation, which is a worse
    /// place for it to fail than here.</para>
    /// </summary>
    public sealed class SafetensorsMalformedHeaderTests
    {
        /// <summary>Assembles a safetensors file from a raw JSON header and a data block of
        /// <paramref name="dataBytes"/> zero bytes.</summary>
        private static byte[] Build(string headerJson, int dataBytes)
        {
            var header = Encoding.UTF8.GetBytes(headerJson);
            var file = new byte[8 + header.Length + dataBytes];

            BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan(0, 8), (ulong)header.Length);
            header.CopyTo(file.AsSpan(8));

            return file;
        }

        [Fact]
        public void TensorExtentBeyondTheFile_IsRefused()
        {
            // Internally consistent and physically impossible: 10^12 F32 elements is exactly the 4 TB byte
            // range declared, in a file of a couple of hundred bytes.
            var file = Build(
                """{"w":{"dtype":"F32","shape":[1000000,1000000],"data_offsets":[0,4000000000000]}}""",
                dataBytes: 16);

            using var stream = new MemoryStream(file);

            Assert.Throws<OverfitFormatException>(() => new SafetensorsReader(stream));
        }

        [Fact]
        public void NegativeDataOffsets_AreRefused()
        {
            var file = Build(
                """{"w":{"dtype":"F32","shape":[4],"data_offsets":[-1000,-984]}}""",
                dataBytes: 16);

            using var stream = new MemoryStream(file);

            Assert.Throws<OverfitFormatException>(() => new SafetensorsReader(stream));
        }

        [Fact]
        public void ReversedDataOffsets_AreRefused()
        {
            // end < begin makes the byte length negative, which no downstream check expects.
            var file = Build(
                """{"w":{"dtype":"F32","shape":[4],"data_offsets":[16,0]}}""",
                dataBytes: 16);

            using var stream = new MemoryStream(file);

            Assert.Throws<OverfitFormatException>(() => new SafetensorsReader(stream));
        }

        [Fact]
        public void NegativeShapeDimension_IsRefused()
        {
            var file = Build(
                """{"w":{"dtype":"F32","shape":[-4,4],"data_offsets":[0,16]}}""",
                dataBytes: 16);

            using var stream = new MemoryStream(file);

            Assert.Throws<OverfitFormatException>(() => new SafetensorsReader(stream));
        }

        [Fact]
        public void ShapeProductOverflowingInt64_IsRefused()
        {
            // Three dimensions whose product wraps past long.MaxValue — the element count then comes out
            // small or negative and agrees with nothing.
            var file = Build(
                """{"w":{"dtype":"F32","shape":[3037000500,3037000500,4],"data_offsets":[0,16]}}""",
                dataBytes: 16);

            using var stream = new MemoryStream(file);

            Assert.Throws<OverfitFormatException>(() => new SafetensorsReader(stream));
        }

        [Fact]
        public void TruncatedAndGarbageFiles_AreRefused_WithACatchableException()
        {
            var inputs = new[]
            {
                Array.Empty<byte>(),
                new byte[4],
                new byte[8],                                    // a length prefix and nothing else
                Build("""{"w":{"dtype":"F32",""", dataBytes: 0), // header cut mid-JSON
                Build("not json", dataBytes: 0),
                Build("[]", dataBytes: 0),                      // valid JSON, wrong shape
            };

            foreach (var file in inputs)
            {
                using var stream = new MemoryStream(file);

                var thrown = Record.Exception(() => new SafetensorsReader(stream));

                Assert.NotNull(thrown);
                Assert.IsNotType<OutOfMemoryException>(thrown);
            }
        }

        [Fact]
        public void AWellFormedTinyFile_IsStillAccepted()
        {
            // The guard must reject impossible extents without rejecting a legitimate small file — otherwise
            // it is not a validation, it is a different bug.
            var file = Build(
                """{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}""",
                dataBytes: 16);

            using var stream = new MemoryStream(file);
            using var reader = new SafetensorsReader(stream);

            Assert.Equal(4, reader.ElementCount("w"));

            var destination = new float[4];
            reader.LoadF32("w", destination);

            Assert.All(destination, v => Assert.Equal(0f, v));
        }
    }
}
