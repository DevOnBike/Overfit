// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Unit tests for the native <see cref="SafetensorsReader"/> over synthetic
    /// files (no external model needed): header parse, shape/dtype, and F32/F16/BF16
    /// dequant round-trips.
    /// </summary>
    public sealed class SafetensorsReaderTests
    {
        // w_f32 [2,3] = 6 floats, w_f16 [4], w_bf16 [3] — contiguous in the data block.
        private static readonly float[] F32Values = [1.5f, -2.25f, 0f, 3.75f, 100f, -0.5f];
        private static readonly float[] F16Values = [0.5f, 1f, -1f, 2f];     // exact in F16
        private static readonly float[] BF16Values = [1f, 2f, -4f];          // exact in BF16

        private const string Header =
            "{\"__metadata__\":{\"format\":\"pt\"}," +
            "\"w_f32\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}," +
            "\"w_f16\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[24,32]}," +
            "\"w_bf16\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[32,38]}}";

        [Fact]
        public void ParsesHeader_NamesDtypesShapesMetadata()
        {
            using var reader = new SafetensorsReader(BuildStream(), ownsStream: true);

            Assert.Equal(3, reader.Tensors.Count);
            Assert.Equal("pt", reader.Metadata["format"]);

            Assert.Equal(SafetensorsDType.F32, reader.Tensors["w_f32"].DType);
            Assert.Equal(new long[] { 2, 3 }, reader.Tensors["w_f32"].Shape);
            Assert.Equal(6, reader.Tensors["w_f32"].ElementCount);
            Assert.Equal(6, reader.ElementCount("w_f32"));

            Assert.Equal(SafetensorsDType.F16, reader.Tensors["w_f16"].DType);
            Assert.Equal(4, reader.ElementCount("w_f16"));

            Assert.Equal(SafetensorsDType.BF16, reader.Tensors["w_bf16"].DType);
            Assert.Equal(3, reader.ElementCount("w_bf16"));
        }

        [Fact]
        public void LoadF32_RoundTripsF32_Exactly()
        {
            using var reader = new SafetensorsReader(BuildStream(), ownsStream: true);
            var dst = new float[6];
            reader.LoadF32("w_f32", dst);
            Assert.Equal(F32Values, dst);
        }

        [Fact]
        public void LoadF32_ConvertsF16()
        {
            using var reader = new SafetensorsReader(BuildStream(), ownsStream: true);
            var dst = new float[4];
            reader.LoadF32("w_f16", dst);
            Assert.Equal(F16Values, dst);   // chosen values are exact in F16
        }

        [Fact]
        public void LoadF32_ConvertsBF16()
        {
            using var reader = new SafetensorsReader(BuildStream(), ownsStream: true);
            var dst = new float[3];
            reader.LoadF32("w_bf16", dst);
            Assert.Equal(BF16Values, dst);  // chosen values are exact in BF16
        }

        [Fact]
        public void LoadF32_MissingTensor_Throws()
        {
            using var reader = new SafetensorsReader(BuildStream(), ownsStream: true);
            Assert.Throws<KeyNotFoundException>(() => reader.LoadF32("nope", new float[1]));
        }

        [Fact]
        public void LoadF32_DestinationTooSmall_Throws()
        {
            using var reader = new SafetensorsReader(BuildStream(), ownsStream: true);
            Assert.Throws<ArgumentException>(() => reader.LoadF32("w_f32", new float[5]));
        }

        [Fact]
        public void Constructor_BadHeaderLength_Throws()
        {
            var bytes = new byte[16];
            BinaryPrimitives.WriteUInt64LittleEndian(bytes, 9999);  // > file length
            Assert.Throws<InvalidDataException>(() => new SafetensorsReader(new MemoryStream(bytes), ownsStream: true));
        }

        // ── Builds an in-memory safetensors blob matching Header ────────────
        private static MemoryStream BuildStream()
        {
            var headerBytes = Encoding.UTF8.GetBytes(Header);

            using var ms = new MemoryStream();
            Span<byte> len = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)headerBytes.Length);
            ms.Write(len);
            ms.Write(headerBytes);

            // Data block, in offset order: f32, f16, bf16.
            Span<byte> tmp4 = stackalloc byte[4];
            foreach (var v in F32Values)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(tmp4, BitConverter.SingleToUInt32Bits(v));
                ms.Write(tmp4);
            }
            Span<byte> tmp2 = stackalloc byte[2];
            foreach (var v in F16Values)
            {
                BinaryPrimitives.WriteUInt16LittleEndian(tmp2, BitConverter.HalfToUInt16Bits((Half)v));
                ms.Write(tmp2);
            }
            foreach (var v in BF16Values)
            {
                var bf = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
                BinaryPrimitives.WriteUInt16LittleEndian(tmp2, bf);
                ms.Write(tmp2);
            }

            return new MemoryStream(ms.ToArray());
        }
    }
}
