// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// A sidecar file holding Q4_K weight matrices pre-converted to the <c>block_q4_Kx8</c> repacked layout
    /// (the one <see cref="Runtime.Q4KGemvKernel"/> / <see cref="Runtime.Q4KGemvKernel.GemmTiled"/> consume).
    /// Produced offline once from a GGUF; at load the repacked bytes are memory-mapped straight into the
    /// kernels — so the fast decode-GEMV / tiled-prefill path costs <b>zero extra heap</b> (no runtime
    /// <c>EnsureRepacked</c> copy) and can be default-on. The bytes written here are byte-for-byte what
    /// <see cref="Runtime.Q4KRepack.RepackMatrix"/> builds at runtime, so nothing about the computation
    /// changes — it is purely a loading/RAM optimisation.
    ///
    /// <para>Format (little-endian): <c>"OVFRPK" + {1,0}</c> magic (8 B) · <c>int32</c> entry count · then per
    /// entry <c>int32</c> name length + UTF-8 name + <c>int32</c> inputSize + <c>int32</c> outputSize +
    /// <c>int64</c> blob offset + <c>int64</c> blob length; the repacked blobs follow, each 16-byte aligned.</para>
    /// </summary>
    public sealed class RepackedWeightsFile : IDisposable
    {
        private static readonly byte[] Magic = [(byte)'O', (byte)'V', (byte)'F', (byte)'R', (byte)'P', (byte)'K', 1, 0];
        private const int Alignment = 16;

        private readonly MemoryMappedModelFile _mapped;
        private readonly Dictionary<string, TensorRecord> _index;

        private RepackedWeightsFile(MemoryMappedModelFile mapped, Dictionary<string, TensorRecord> index)
        {
            _mapped = mapped;
            _index = index;
        }

        /// <summary>Number of repacked tensors in the file.</summary>
        public int Count => _index.Count;

        /// <summary>One repacked tensor to persist (its <paramref name="Repacked"/> bytes are the
        /// <c>block_q4_Kx8</c> layout from <see cref="Runtime.Q4KRepack.RepackMatrix"/>).</summary>
        public readonly record struct Entry(string Name, int InputSize, int OutputSize, ReadOnlyMemory<byte> Repacked);

        private readonly record struct TensorRecord(int InputSize, int OutputSize, long Offset, long Length);

        /// <summary>
        /// Offline tool: reads a GGUF and writes a sidecar with every repackable Q4_K matmul weight converted to
        /// <c>block_q4_Kx8</c>. Skips non-Q4_K tensors, non-2D tensors, the token embedding (consumed row-wise via
        /// <c>DecodeRow</c>, never through the repacked kernel), and shapes that can't repack (outputSize not a
        /// multiple of 8, or inputSize not a multiple of 256). Returns the number of tensors written.
        /// </summary>
        public static int BuildFromGguf(string ggufPath, string outPath)
        {
            ArgumentNullException.ThrowIfNull(ggufPath);
            ArgumentNullException.ThrowIfNull(outPath);

            using var reader = new GgufReader(ggufPath);
            var entries = new List<Entry>();

            foreach (var info in reader.Tensors.Values)
            {
                if (info.Type != GgmlType.Q4_K || info.Dims.Length != 2)
                {
                    continue;
                }
                if (info.Name.Contains("token_embd", StringComparison.Ordinal))
                {
                    continue;
                }

                var inputSize = checked((int)info.Dims[0]);
                var outputSize = checked((int)info.Dims[1]);
                if (inputSize % Q4KWeight.SuperBlockElements != 0 || outputSize % Q4KRepack.RowsInterleaved != 0)
                {
                    continue;
                }

                var byteCount = checked((int)((long)outputSize * (inputSize / Q4KWeight.SuperBlockElements) * Q4KRepack.SuperBlockBytes));
                var raw = new byte[byteCount];
                reader.LoadTensorQ4_KRaw(info, raw);
                entries.Add(new Entry(info.Name, inputSize, outputSize, Q4KRepack.RepackMatrix(raw, outputSize, inputSize)));
            }

            Write(outPath, entries);
            return entries.Count;
        }

        /// <summary>Writes the sidecar: header + index + 16-byte-aligned repacked blobs.</summary>
        public static void Write(string path, IReadOnlyList<Entry> entries)
        {
            ArgumentNullException.ThrowIfNull(path);
            ArgumentNullException.ThrowIfNull(entries);

            // Pass 1: index size, then each blob's aligned offset.
            long cursor = Magic.Length + sizeof(int);
            foreach (var e in entries)
            {
                cursor += sizeof(int) + Encoding.UTF8.GetByteCount(e.Name) + sizeof(int) + sizeof(int) + sizeof(long) + sizeof(long);
            }

            var offsets = new long[entries.Count];
            for (var i = 0; i < entries.Count; i++)
            {
                cursor = Align(cursor);
                offsets[i] = cursor;
                cursor += entries[i].Repacked.Length;
            }

            using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
            using var writer = new BinaryWriter(stream);
            writer.Write(Magic);
            writer.Write(entries.Count);
            for (var i = 0; i < entries.Count; i++)
            {
                var e = entries[i];
                var nameBytes = Encoding.UTF8.GetBytes(e.Name);
                writer.Write(nameBytes.Length);
                writer.Write(nameBytes);
                writer.Write(e.InputSize);
                writer.Write(e.OutputSize);
                writer.Write(offsets[i]);
                writer.Write((long)e.Repacked.Length);
            }

            for (var i = 0; i < entries.Count; i++)
            {
                Pad(writer, offsets[i] - writer.BaseStream.Position);
                writer.Write(entries[i].Repacked.Span);
            }
        }

        /// <summary>Opens the sidecar and memory-maps it. Blob slices from <see cref="TryGet"/> point straight
        /// into the mapping (zero-copy) and stay valid until this instance is disposed.</summary>
        public static RepackedWeightsFile Open(string path)
        {
            ArgumentNullException.ThrowIfNull(path);

            Dictionary<string, TensorRecord> index = new(StringComparer.Ordinal);
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var reader = new BinaryReader(stream))
            {
                var magic = reader.ReadBytes(Magic.Length);
                if (magic.Length != Magic.Length || !magic.AsSpan().SequenceEqual(Magic))
                {
                    throw new OverfitFormatException($"'{path}' is not a repacked-weights file (bad magic).");
                }

                var count = reader.ReadInt32();
                if (count < 0)
                {
                    throw new OverfitFormatException($"'{path}' has a negative entry count ({count}).");
                }

                for (var i = 0; i < count; i++)
                {
                    var nameLen = reader.ReadInt32();
                    var name = Encoding.UTF8.GetString(reader.ReadBytes(nameLen));
                    var inputSize = reader.ReadInt32();
                    var outputSize = reader.ReadInt32();
                    var offset = reader.ReadInt64();
                    var length = reader.ReadInt64();
                    index[name] = new TensorRecord(inputSize, outputSize, offset, length);
                }
            }

            var mapped = new MemoryMappedModelFile(path);
            return new RepackedWeightsFile(mapped, index);
        }

        /// <summary>Zero-copy mmap slice of tensor <paramref name="name"/>'s repacked bytes, or false if absent.</summary>
        public bool TryGet(string name, out int inputSize, out int outputSize, out ReadOnlyMemory<byte> repacked)
        {
            if (_index.TryGetValue(name, out var rec))
            {
                inputSize = rec.InputSize;
                outputSize = rec.OutputSize;
                repacked = _mapped.Slice(rec.Offset, checked((int)rec.Length));
                return true;
            }

            inputSize = 0;
            outputSize = 0;
            repacked = default;
            return false;
        }

        public void Dispose() => _mapped.Dispose();

        private static long Align(long x) => (x + (Alignment - 1)) & ~(long)(Alignment - 1);

        private static void Pad(BinaryWriter writer, long bytes)
        {
            for (var i = 0L; i < bytes; i++)
            {
                writer.Write((byte)0);
            }
        }
    }
}
