// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Native reader for the HuggingFace <c>safetensors</c> format — no Python, no
    /// torch, no conversion step. Closes the last spot where loading a model could
    /// require a Python toolchain: a raw HF repo with no GGUF variant.
    ///
    /// Layout (little-endian, as the spec mandates):
    /// <code>
    ///   [0..8)            uint64  header length N
    ///   [8..8+N)          UTF-8 JSON header: name -> { dtype, shape, data_offsets:[begin,end] }
    ///                     plus an optional "__metadata__" string map
    ///   [8+N..EOF)        raw tensor bytes; data_offsets are relative to this block
    /// </code>
    ///
    /// The JSON header is parsed with the reflection-free, allocation-light
    /// <see cref="Utf8JsonReader"/> so the type stays Native-AOT clean (no
    /// <c>JsonSerializer.Deserialize</c> reflection). Float tensors are dequantized
    /// to F32 on read via <see cref="LoadF32"/>; F32 / F16 / BF16 are supported
    /// (the dtypes LLM weights actually use). Sharded repos
    /// (<c>model-00001-of-0000N.safetensors</c> + <c>model.safetensors.index.json</c>)
    /// are a follow-on — this reads one file.
    /// </summary>
    public sealed class SafetensorsReader : ISafetensorsSource
    {
        private readonly Stream _stream;
        private readonly bool _ownsStream;
        private readonly long _dataStart;
        private bool _disposed;

        public SafetensorsReader(string path)
            : this(File.OpenRead(path), ownsStream: true)
        {
        }

        public SafetensorsReader(Stream stream, bool ownsStream = false)
        {
            if (stream is null) { throw new ArgumentNullException(nameof(stream)); }
            if (!stream.CanRead) { throw new ArgumentException("Stream must be readable.", nameof(stream)); }
            if (!stream.CanSeek) { throw new ArgumentException("Stream must be seekable.", nameof(stream)); }

            _stream = stream;
            _ownsStream = ownsStream;

            // ─── Header length (8-byte LE uint64) ──────────────────────────
            Span<byte> lenBytes = stackalloc byte[8];
            stream.ReadExactly(lenBytes);
            var headerLen = BinaryPrimitives.ReadUInt64LittleEndian(lenBytes);

            if (headerLen == 0 || headerLen > (ulong)(stream.Length - 8))
            {
                throw new InvalidDataException(
                    $"safetensors header length {headerLen} is invalid for a {stream.Length}-byte file.");
            }

            _dataStart = 8 + (long)headerLen;

            // ─── JSON header ───────────────────────────────────────────────
            var headerBytes = new byte[headerLen];
            stream.ReadExactly(headerBytes);

            var (tensors, metadata) = ParseHeader(headerBytes);
            Tensors = tensors;
            Metadata = metadata;
        }

        /// <summary>Tensor name → dtype / shape / byte range (offsets relative to the data block).</summary>
        public IReadOnlyDictionary<string, SafetensorsTensorInfo> Tensors { get; }

        /// <summary>The optional <c>__metadata__</c> string map (empty if absent).</summary>
        public IReadOnlyDictionary<string, string> Metadata { get; }

        /// <summary>Element count (product of shape dims) for a named tensor.</summary>
        public long ElementCount(string name) => Info(name).ElementCount;

        /// <summary>
        /// Reads <paramref name="name"/> and dequantizes it to F32 into
        /// <paramref name="destination"/> (which must be at least
        /// <see cref="ElementCount"/> long). Source dtype may be F32, F16 or BF16.
        /// </summary>
        public void LoadF32(string name, Span<float> destination)
        {
            ThrowIfDisposed();

            var info = Info(name);
            var count = info.ElementCount;
            if (destination.Length < count)
            {
                throw new ArgumentException(
                    $"Destination ({destination.Length}) is smaller than tensor '{name}' element count ({count}).",
                    nameof(destination));
            }

            var byteLen = info.End - info.Begin;
            var expectedBytes = count * BytesPerElement(info.DType);
            if (byteLen != expectedBytes)
            {
                throw new InvalidDataException(
                    $"Tensor '{name}' byte range {byteLen} != expected {expectedBytes} for {info.DType} × {count}.");
            }

            _stream.Seek(_dataStart + info.Begin, SeekOrigin.Begin);

            switch (info.DType)
            {
                case SafetensorsDType.F32:
                    ReadF32(destination, count);
                    break;
                case SafetensorsDType.F16:
                    ReadF16(destination, count);
                    break;
                case SafetensorsDType.BF16:
                    ReadBF16(destination, count);
                    break;
                default:
                    throw new NotSupportedException(
                        $"Tensor '{name}' dtype {info.DType} is not supported by LoadF32 (F32/F16/BF16 only).");
            }
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            if (_ownsStream) { _stream.Dispose(); }
        }

        private SafetensorsTensorInfo Info(string name)
        {
            if (!Tensors.TryGetValue(name, out var info))
            {
                throw new KeyNotFoundException($"Tensor '{name}' not found in safetensors header.");
            }
            return info;
        }

        // ─── F32 / F16 / BF16 streaming dequant (little-endian) ────────────
        private void ReadF32(Span<float> dst, long count)
        {
            // 4 KiB-element chunks keep the scratch small and AOT-friendly.
            const int chunk = 1024;
            Span<byte> buf = stackalloc byte[chunk * 4];
            var done = 0L;
            while (done < count)
            {
                var n = (int)Math.Min(chunk, count - done);
                var slice = buf[..(n * 4)];
                _stream.ReadExactly(slice);
                for (var i = 0; i < n; i++)
                {
                    var bits = BinaryPrimitives.ReadUInt32LittleEndian(slice.Slice(i * 4, 4));
                    dst[(int)(done + i)] = BitConverter.UInt32BitsToSingle(bits);
                }
                done += n;
            }
        }

        private void ReadF16(Span<float> dst, long count)
        {
            const int chunk = 2048;
            Span<byte> buf = stackalloc byte[chunk * 2];
            var done = 0L;
            while (done < count)
            {
                var n = (int)Math.Min(chunk, count - done);
                var slice = buf[..(n * 2)];
                _stream.ReadExactly(slice);
                for (var i = 0; i < n; i++)
                {
                    var bits = BinaryPrimitives.ReadUInt16LittleEndian(slice.Slice(i * 2, 2));
                    dst[(int)(done + i)] = (float)BitConverter.UInt16BitsToHalf(bits);
                }
                done += n;
            }
        }

        private void ReadBF16(Span<float> dst, long count)
        {
            // bfloat16 is the upper 16 bits of an IEEE-754 float32.
            const int chunk = 2048;
            Span<byte> buf = stackalloc byte[chunk * 2];
            var done = 0L;
            while (done < count)
            {
                var n = (int)Math.Min(chunk, count - done);
                var slice = buf[..(n * 2)];
                _stream.ReadExactly(slice);
                for (var i = 0; i < n; i++)
                {
                    var bf = BinaryPrimitives.ReadUInt16LittleEndian(slice.Slice(i * 2, 2));
                    dst[(int)(done + i)] = BitConverter.UInt32BitsToSingle((uint)bf << 16);
                }
                done += n;
            }
        }

        private static long BytesPerElement(SafetensorsDType dtype) => dtype switch
        {
            SafetensorsDType.F64 or SafetensorsDType.I64 or SafetensorsDType.U64 => 8,
            SafetensorsDType.F32 or SafetensorsDType.I32 or SafetensorsDType.U32 => 4,
            SafetensorsDType.F16 or SafetensorsDType.BF16 or SafetensorsDType.I16 or SafetensorsDType.U16 => 2,
            SafetensorsDType.I8 or SafetensorsDType.U8 or SafetensorsDType.Bool => 1,
            _ => throw new NotSupportedException($"Unknown safetensors dtype {dtype}."),
        };

        // ─── Header JSON parse (Utf8JsonReader — reflection-free) ──────────
        private static (Dictionary<string, SafetensorsTensorInfo> tensors, Dictionary<string, string> metadata)
            ParseHeader(byte[] headerBytes)
        {
            var tensors = new Dictionary<string, SafetensorsTensorInfo>();
            var metadata = new Dictionary<string, string>();

            var reader = new Utf8JsonReader(headerBytes, isFinalBlock: true, state: default);

            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            {
                throw new InvalidDataException("safetensors header is not a JSON object.");
            }

            while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
            {
                var name = reader.GetString()!;
                reader.Read(); // step onto the value

                if (name == "__metadata__")
                {
                    ReadStringMap(ref reader, metadata);
                    continue;
                }

                tensors[name] = ReadTensorEntry(ref reader, name);
            }

            return (tensors, metadata);
        }

        private static void ReadStringMap(ref Utf8JsonReader reader, Dictionary<string, string> into)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
            {
                reader.Skip();
                return;
            }
            while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
            {
                var key = reader.GetString()!;
                reader.Read();
                into[key] = reader.TokenType == JsonTokenType.String ? reader.GetString()! : string.Empty;
            }
        }

        private static SafetensorsTensorInfo ReadTensorEntry(ref Utf8JsonReader reader, string name)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
            {
                throw new InvalidDataException($"Tensor '{name}' entry is not a JSON object.");
            }

            var dtype = SafetensorsDType.F32;
            long[] shape = [];
            long begin = 0, end = 0;

            while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
            {
                var field = reader.GetString()!;
                reader.Read();

                switch (field)
                {
                    case "dtype":
                        dtype = ParseDType(reader.GetString()!);
                        break;
                    case "shape":
                        shape = ReadLongArray(ref reader);
                        break;
                    case "data_offsets":
                        var off = ReadLongArray(ref reader);
                        if (off.Length != 2)
                        {
                            throw new InvalidDataException($"Tensor '{name}' data_offsets must have 2 entries.");
                        }
                        begin = off[0];
                        end = off[1];
                        break;
                    default:
                        reader.Skip();
                        break;
                }
            }

            return new SafetensorsTensorInfo(name, dtype, shape, begin, end);
        }

        private static long[] ReadLongArray(ref Utf8JsonReader reader)
        {
            if (reader.TokenType != JsonTokenType.StartArray)
            {
                throw new InvalidDataException("Expected a JSON array.");
            }
            var values = new List<long>(4);
            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
            {
                values.Add(reader.GetInt64());
            }
            return values.ToArray();
        }

        private static SafetensorsDType ParseDType(string s) => s switch
        {
            "F64" => SafetensorsDType.F64,
            "F32" => SafetensorsDType.F32,
            "F16" => SafetensorsDType.F16,
            "BF16" => SafetensorsDType.BF16,
            "I64" => SafetensorsDType.I64,
            "U64" => SafetensorsDType.U64,
            "I32" => SafetensorsDType.I32,
            "U32" => SafetensorsDType.U32,
            "I16" => SafetensorsDType.I16,
            "U16" => SafetensorsDType.U16,
            "I8" => SafetensorsDType.I8,
            "U8" => SafetensorsDType.U8,
            "BOOL" => SafetensorsDType.Bool,
            _ => throw new NotSupportedException($"Unknown safetensors dtype '{s}'."),
        };

        private void ThrowIfDisposed()
        {
            if (_disposed) { throw new ObjectDisposedException(nameof(SafetensorsReader)); }
        }
    }
}
