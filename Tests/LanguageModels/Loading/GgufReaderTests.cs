// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    [Trait("Category", "Gguf")]
    public sealed class GgufReaderTests
    {
        [Fact]
        public void Constructor_ThrowsOnInvalidMagic()
        {
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write((uint)0xDEADBEEF);  // wrong magic
                bw.Write((uint)3);
            }
            ms.Position = 0;

            var ex = Assert.Throws<InvalidDataException>(() => new GgufReader(ms));
            Assert.Contains("Not a GGUF file", ex.Message);
        }

        [Fact]
        public void Constructor_ThrowsOnUnsupportedVersion()
        {
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(GgufFormat.Magic);
                bw.Write((uint)1);  // version 1 not supported
                bw.Write((ulong)0);  // tensor count
                bw.Write((ulong)0);  // meta count
            }
            ms.Position = 0;

            var ex = Assert.Throws<NotSupportedException>(() => new GgufReader(ms));
            Assert.Contains("version 1", ex.Message);
        }

        [Fact]
        public void Parses_HeaderAndEmptyMetadata()
        {
            using var ms = BuildMinimalGguf(metadata: null, tensors: null);
            using var reader = new GgufReader(ms);

            Assert.Equal(3u, reader.Version);
            Assert.Empty(reader.Metadata);
            Assert.Empty(reader.Tensors);
        }

        [Fact]
        public void Parses_StringAndIntMetadata()
        {
            var meta = new Dictionary<string, (GgufValueType, object)>
            {
                ["general.architecture"] = (GgufValueType.String, "qwen2"),
                ["qwen2.block_count"] = (GgufValueType.UInt32, (uint)24),
                ["qwen2.rope.freq_base"] = (GgufValueType.Float32, 1_000_000.0f),
            };
            using var ms = BuildMinimalGguf(meta, tensors: null);
            using var reader = new GgufReader(ms);

            Assert.Equal("qwen2", reader.GetMeta<string>("general.architecture", ""));
            Assert.Equal(24, reader.GetMeta<int>("qwen2.block_count", 0));  // widening uint→int
            Assert.Equal(1_000_000.0f, reader.GetMeta<float>("qwen2.rope.freq_base", 0f));
            Assert.Equal(99, reader.GetMeta<int>("nonexistent", 99));  // default fallback
        }

        [Fact]
        public void LoadTensorAsF32_ReadsF32Directly()
        {
            // Build GGUF with single F32 tensor [4] = {1, 2, 3, 4}
            var data = new float[] { 1f, 2f, 3f, 4f };
            using var ms = BuildGgufWithSingleTensor("test", new ulong[] { 4 }, GgmlType.F32, data);
            using var reader = new GgufReader(ms);

            var info = reader.Tensors["test"];
            Assert.Equal(GgmlType.F32, info.Type);
            Assert.Equal(4, info.ElementCount);

            var dst = new float[4];
            reader.LoadTensorAsF32(info, dst);
            Assert.Equal(data, dst);
        }

        [Fact]
        public void LoadTensorAsF32_ConvertsF16ToF32()
        {
            // F16 representation of {1.0, 2.0, -0.5, 100.0}
            var values = new float[] { 1.0f, 2.0f, -0.5f, 100.0f };
            var f16Bytes = new byte[values.Length * 2];
            for (var i = 0; i < values.Length; i++)
            {
                var half = (Half)values[i];
                var u16 = BitConverter.HalfToUInt16Bits(half);
                BinaryPrimitives.WriteUInt16LittleEndian(f16Bytes.AsSpan(i * 2), u16);
            }

            using var ms = BuildGgufWithTensorBytes("test_f16", new ulong[] { 4 }, GgmlType.F16, f16Bytes);
            using var reader = new GgufReader(ms);

            var info = reader.Tensors["test_f16"];
            var dst = new float[4];
            reader.LoadTensorAsF32(info, dst);

            // F16 has limited precision but these values are exactly representable
            Assert.Equal(1.0f, dst[0], precision: 4);
            Assert.Equal(2.0f, dst[1], precision: 4);
            Assert.Equal(-0.5f, dst[2], precision: 4);
            Assert.Equal(100.0f, dst[3], precision: 1);
        }

        [Fact]
        public void LoadTensorAsF32_ThrowsOnUnsupportedQuantizedType()
        {
            // Q4_K and Q6_K are now supported; use Q5_K (still unsupported) to keep
            // the assertion meaningful. Body bytes are irrelevant — dispatch throws
            // before reading any tensor data.
            using var ms = BuildGgufWithTensorBytes("test_q5k", new ulong[] { 256 }, GgmlType.Q5_K, new byte[176]);
            using var reader = new GgufReader(ms);

            var info = reader.Tensors["test_q5k"];
            var dst = new float[256];

            var ex = Assert.Throws<NotSupportedException>(() => reader.LoadTensorAsF32(info, dst));
            Assert.Contains("Q5_K", ex.Message);
        }

        // ─── Helpers to build minimal in-memory GGUF files ──────────────────

        private static MemoryStream BuildMinimalGguf(
            Dictionary<string, (GgufValueType, object)>? metadata,
            Dictionary<string, (ulong[] dims, GgmlType type, byte[] data)>? tensors)
        {
            var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(GgufFormat.Magic);
                bw.Write((uint)3);  // version
                bw.Write((ulong)(tensors?.Count ?? 0));
                bw.Write((ulong)(metadata?.Count ?? 0));

                if (metadata != null)
                {
                    foreach (var (key, (vtype, value)) in metadata)
                    {
                        WriteString(bw, key);
                        bw.Write((uint)vtype);
                        WriteValue(bw, vtype, value);
                    }
                }

                if (tensors != null)
                {
                    ulong offset = 0;
                    var sizes = new Dictionary<string, ulong>();
                    foreach (var (name, (dims, type, data)) in tensors)
                    {
                        WriteString(bw, name);
                        bw.Write((uint)dims.Length);
                        foreach (var d in dims) { bw.Write(d); }
                        bw.Write((uint)type);
                        bw.Write(offset);
                        sizes[name] = (ulong)data.Length;
                        offset += (ulong)data.Length;
                    }

                    // Align to 32
                    var pos = ms.Position;
                    var aligned = (pos + 31) & ~31L;
                    var padding = aligned - pos;
                    for (var i = 0; i < padding; i++) { bw.Write((byte)0); }

                    // Data section
                    foreach (var (_, (_, _, data)) in tensors)
                    {
                        bw.Write(data);
                    }
                }
            }
            ms.Position = 0;
            return ms;
        }

        private static MemoryStream BuildGgufWithSingleTensor(string name, ulong[] dims, GgmlType type, float[] data)
        {
            var bytes = new byte[data.Length * 4];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            return BuildGgufWithTensorBytes(name, dims, type, bytes);
        }

        private static MemoryStream BuildGgufWithTensorBytes(string name, ulong[] dims, GgmlType type, byte[] data)
        {
            return BuildMinimalGguf(
                metadata: null,
                tensors: new Dictionary<string, (ulong[], GgmlType, byte[])>
                {
                    [name] = (dims, type, data)
                });
        }

        private static void WriteString(BinaryWriter bw, string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            bw.Write((ulong)bytes.Length);
            bw.Write(bytes);
        }

        private static void WriteValue(BinaryWriter bw, GgufValueType vtype, object value)
        {
            switch (vtype)
            {
                case GgufValueType.UInt32:  bw.Write((uint)value); break;
                case GgufValueType.Int32:   bw.Write((int)value); break;
                case GgufValueType.Float32: bw.Write((float)value); break;
                case GgufValueType.String:  WriteString(bw, (string)value); break;
                case GgufValueType.UInt64:  bw.Write((ulong)value); break;
                case GgufValueType.Bool:    bw.Write((bool)value); break;
                default: throw new NotImplementedException($"WriteValue: {vtype}");
            }
        }
    }
}
