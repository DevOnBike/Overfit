// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Reads a GGUF file: header, metadata KVs, tensor info, and tensor data.
    /// Supports F32, F16, BF16, Q8_0, Q4_K, Q6_K. Other quantized formats throw NotSupportedException.
    ///
    /// Usage:
    ///   using var reader = new GgufReader("model.gguf");
    ///   var arch = reader.GetMeta&lt;string&gt;("general.architecture", "qwen2");
    ///   var info = reader.Tensors["token_embd.weight"];
    ///   var buffer = new float[info.ElementCount];
    ///   reader.LoadTensorAsF32(info, buffer);
    /// </summary>
    public sealed class GgufReader : IDisposable
    {
        private readonly Stream _stream;
        private readonly BinaryReader _reader;
        private readonly long _dataStart;
        private bool _disposed;

        public uint Version { get; }
        public IReadOnlyDictionary<string, object> Metadata { get; }
        public IReadOnlyDictionary<string, GgufTensorInfo> Tensors { get; }

        public GgufReader(string path)
            : this(File.OpenRead(path))
        {
        }

        public GgufReader(Stream stream)
        {
            if (stream is null) { throw new ArgumentNullException(nameof(stream)); }
            if (!stream.CanRead) { throw new ArgumentException("Stream must be readable.", nameof(stream)); }
            if (!stream.CanSeek) { throw new ArgumentException("Stream must be seekable.", nameof(stream)); }

            _stream = stream;
            _reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);

            // ─── Magic + version ───────────────────────────────────────────
            var magic = _reader.ReadUInt32();
            if (magic != GgufFormat.Magic)
            {
                throw new InvalidDataException(
                    $"Not a GGUF file. Expected magic 0x{GgufFormat.Magic:X8}, got 0x{magic:X8}.");
            }

            Version = _reader.ReadUInt32();
            if (Version < GgufFormat.SupportedVersionMin || Version > GgufFormat.SupportedVersionMax)
            {
                throw new NotSupportedException(
                    $"GGUF version {Version} not supported (range {GgufFormat.SupportedVersionMin}..{GgufFormat.SupportedVersionMax}).");
            }

            var tensorCount = _reader.ReadUInt64();
            var metaCount = _reader.ReadUInt64();

            // ─── Metadata KVs ──────────────────────────────────────────────
            var meta = new Dictionary<string, object>((int)metaCount);
            for (var i = 0UL; i < metaCount; i++)
            {
                var key = ReadString();
                var vtype = (GgufValueType)_reader.ReadUInt32();
                meta[key] = ReadValue(vtype);
            }
            Metadata = meta;

            // ─── Tensor info entries ───────────────────────────────────────
            var tensors = new Dictionary<string, GgufTensorInfo>((int)tensorCount);
            for (var i = 0UL; i < tensorCount; i++)
            {
                var name = ReadString();
                var nDims = _reader.ReadUInt32();
                var dims = new ulong[nDims];
                for (var d = 0; d < nDims; d++)
                {
                    dims[d] = _reader.ReadUInt64();
                }
                var type = (GgmlType)_reader.ReadUInt32();
                var offset = _reader.ReadUInt64();
                tensors[name] = new GgufTensorInfo(name, dims, type, offset);
            }
            Tensors = tensors;

            // ─── Align to 32-byte boundary ─────────────────────────────────
            var pos = _stream.Position;
            _dataStart = (pos + GgufFormat.DataAlignment - 1) & ~(long)(GgufFormat.DataAlignment - 1);
        }

        /// <summary>
        /// Loads tensor data as float32 into <paramref name="destination"/>.
        /// Performs F16 → F32 conversion if needed. F32 is copied directly.
        /// </summary>
        public void LoadTensorAsF32(GgufTensorInfo info, Span<float> destination)
        {
            if (info is null) { throw new ArgumentNullException(nameof(info)); }

            var elementCount = info.ElementCount;
            if (destination.Length < elementCount)
            {
                throw new ArgumentException(
                    $"Destination span too small: {destination.Length} < {elementCount}.",
                    nameof(destination));
            }

            _stream.Seek(_dataStart + (long)info.Offset, SeekOrigin.Begin);

            switch (info.Type)
            {
                case GgmlType.F32:
                    ReadF32(destination[..(int)elementCount]);
                    return;

                case GgmlType.F16:
                    ReadF16ToF32(destination[..(int)elementCount]);
                    return;

                case GgmlType.BF16:
                    ReadBF16ToF32(destination[..(int)elementCount]);
                    return;

                case GgmlType.Q8_0:
                    ReadQ8_0ToF32(destination[..(int)elementCount]);
                    return;

                case GgmlType.Q4_K:
                    ReadQ4_KToF32(destination[..(int)elementCount]);
                    return;

                case GgmlType.Q6_K:
                    ReadQ6_KToF32(destination[..(int)elementCount]);
                    return;

                default:
                    throw new NotSupportedException(
                        $"Tensor type {info.Type} is not supported yet. " +
                        $"Supported: F32, F16, BF16, Q8_0, Q4_K, Q6_K.");
            }
        }

        /// <summary>Reads a metadata value or returns the default if the key is absent.</summary>
        public T GetMeta<T>(string key, T defaultValue)
        {
            if (!Metadata.TryGetValue(key, out var value)) { return defaultValue; }
            if (value is T typed) { return typed; }

            // Common widening: GGUF may store small ints as uint32, but C# default int is int32.
            try { return (T)Convert.ChangeType(value, typeof(T))!; }
            catch { return defaultValue; }
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _reader.Dispose();
        }

        // ─── Private I/O helpers ─────────────────────────────────────────────

        private string ReadString()
        {
            var n = _reader.ReadUInt64();
            if (n > int.MaxValue) { throw new InvalidDataException($"String length {n} exceeds int.MaxValue."); }
            var bytes = _reader.ReadBytes((int)n);
            return Encoding.UTF8.GetString(bytes);
        }

        private object ReadValue(GgufValueType vtype)
        {
            switch (vtype)
            {
                case GgufValueType.UInt8:   return _reader.ReadByte();
                case GgufValueType.Int8:    return _reader.ReadSByte();
                case GgufValueType.UInt16:  return _reader.ReadUInt16();
                case GgufValueType.Int16:   return _reader.ReadInt16();
                case GgufValueType.UInt32:  return _reader.ReadUInt32();
                case GgufValueType.Int32:   return _reader.ReadInt32();
                case GgufValueType.Float32: return _reader.ReadSingle();
                case GgufValueType.Bool:    return _reader.ReadBoolean();
                case GgufValueType.String:  return ReadString();
                case GgufValueType.UInt64:  return _reader.ReadUInt64();
                case GgufValueType.Int64:   return _reader.ReadInt64();
                case GgufValueType.Float64: return _reader.ReadDouble();
                case GgufValueType.Array:
                {
                    var elemType = (GgufValueType)_reader.ReadUInt32();
                    var count = _reader.ReadUInt64();
                    if (count > int.MaxValue) { throw new InvalidDataException("Array too large."); }
                    var arr = new object[(int)count];
                    for (var i = 0; i < arr.Length; i++) { arr[i] = ReadValue(elemType); }
                    return arr;
                }
                default:
                    throw new NotSupportedException($"GGUF value type {vtype} not supported.");
            }
        }

        private void ReadF32(Span<float> dst)
        {
            var bytes = MemoryMarshal.AsBytes(dst);
            var read = 0;
            while (read < bytes.Length)
            {
                var n = _stream.Read(bytes[read..]);
                if (n == 0) { throw new EndOfStreamException("Unexpected EOF reading tensor data."); }
                read += n;
            }
        }

        private void ReadF16ToF32(Span<float> dst)
        {
            // Read raw 2 bytes per element, convert to float
            const int batch = 8192;
            var buf = new byte[batch * 2];
            var i = 0;
            while (i < dst.Length)
            {
                var take = Math.Min(batch, dst.Length - i);
                var bytesToRead = take * 2;
                var read = 0;
                while (read < bytesToRead)
                {
                    var n = _stream.Read(buf, read, bytesToRead - read);
                    if (n == 0) { throw new EndOfStreamException("Unexpected EOF reading F16 tensor."); }
                    read += n;
                }
                for (var k = 0; k < take; k++)
                {
                    var u16 = BinaryPrimitives.ReadUInt16LittleEndian(buf.AsSpan(k * 2));
                    dst[i + k] = (float)BitConverter.UInt16BitsToHalf(u16);
                }
                i += take;
            }
        }

        private void ReadBF16ToF32(Span<float> dst)
        {
            // BF16: top 16 bits of F32. Shift left by 16 to reconstruct.
            const int batch = 8192;
            var buf = new byte[batch * 2];
            var i = 0;
            while (i < dst.Length)
            {
                var take = Math.Min(batch, dst.Length - i);
                var bytesToRead = take * 2;
                var read = 0;
                while (read < bytesToRead)
                {
                    var n = _stream.Read(buf, read, bytesToRead - read);
                    if (n == 0) { throw new EndOfStreamException("Unexpected EOF reading BF16 tensor."); }
                    read += n;
                }
                for (var k = 0; k < take; k++)
                {
                    var u16 = BinaryPrimitives.ReadUInt16LittleEndian(buf.AsSpan(k * 2));
                    var u32 = (uint)u16 << 16;
                    dst[i + k] = BitConverter.UInt32BitsToSingle(u32);
                }
                i += take;
            }
        }

        /// <summary>
        /// Dequantize a Q4_K tensor block-by-block (144 bytes per 256 elements).
        /// Decode math lives in <see cref="GgmlDequant.DecodeQ4_KBlock"/>; this method
        /// only handles streaming. Scratch buffer is stack-allocated — zero managed
        /// allocations regardless of tensor size.
        /// </summary>
        private void ReadQ4_KToF32(Span<float> dst)
        {
            const int BlockElements = GgmlDequant.SuperBlockElements;
            const int BlockBytes = GgmlDequant.Q4_K_BlockBytes;

            if (dst.Length % BlockElements != 0)
            {
                throw new InvalidDataException(
                    $"Q4_K tensor element count {dst.Length} is not divisible by {BlockElements}.");
            }

            var nBlocks = dst.Length / BlockElements;
            Span<byte> buf = stackalloc byte[BlockBytes];

            for (var b = 0; b < nBlocks; b++)
            {
                _stream.ReadExactly(buf);
                GgmlDequant.DecodeQ4_KBlock(buf, dst.Slice(b * BlockElements, BlockElements));
            }
        }

        /// <summary>
        /// Dequantize a Q6_K tensor block-by-block (210 bytes per 256 elements).
        /// Used in Q4_K_M mixed-quant files for token_embd / output tensors.
        /// </summary>
        private void ReadQ6_KToF32(Span<float> dst)
        {
            const int BlockElements = GgmlDequant.SuperBlockElements;
            const int BlockBytes = GgmlDequant.Q6_K_BlockBytes;

            if (dst.Length % BlockElements != 0)
            {
                throw new InvalidDataException(
                    $"Q6_K tensor element count {dst.Length} is not divisible by {BlockElements}.");
            }

            var nBlocks = dst.Length / BlockElements;
            Span<byte> buf = stackalloc byte[BlockBytes];

            for (var b = 0; b < nBlocks; b++)
            {
                _stream.ReadExactly(buf);
                GgmlDequant.DecodeQ6_KBlock(buf, dst.Slice(b * BlockElements, BlockElements));
            }
        }

        /// <summary>
        /// Dequantize Q8_0 tensor block-by-block.
        ///
        /// Q8_0 layout (34 bytes per block, 32 elements):
        ///   [2 bytes FP16 scale]
        ///   [32 signed int8 quants]
        /// Value: dst[i] = scale * (int8)quants[i]
        ///
        /// Element count MUST be divisible by 32 (true for all real model weights).
        /// </summary>
        private void ReadQ8_0ToF32(Span<float> dst)
        {
            const int BlockElements = 32;
            const int BlockBytes = 2 + 32;  // FP16 scale + 32 int8 quants

            if (dst.Length % BlockElements != 0)
            {
                throw new InvalidDataException(
                    $"Q8_0 tensor element count {dst.Length} is not divisible by {BlockElements}.");
            }

            var nBlocks = dst.Length / BlockElements;
            var buf = new byte[BlockBytes];
            var dstPos = 0;

            for (var b = 0; b < nBlocks; b++)
            {
                // Read one block atomically
                var read = 0;
                while (read < BlockBytes)
                {
                    var n = _stream.Read(buf, read, BlockBytes - read);
                    if (n == 0) { throw new EndOfStreamException("Unexpected EOF reading Q8_0 block."); }
                    read += n;
                }

                // Dequantize
                var scaleU16 = BinaryPrimitives.ReadUInt16LittleEndian(buf.AsSpan(0, 2));
                var scale = (float)BitConverter.UInt16BitsToHalf(scaleU16);
                for (var i = 0; i < BlockElements; i++)
                {
                    var q = (sbyte)buf[2 + i];
                    dst[dstPos + i] = scale * q;
                }
                dstPos += BlockElements;
            }
        }
    }
}
