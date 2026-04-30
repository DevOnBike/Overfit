// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Text;

namespace DevOnBike.Overfit.Onnx.Protobuf
{
    /// <summary>
    /// Minimal Protocol Buffers wire format reader, sufficient for parsing ONNX models.
    /// Supports: varint, fixed32, fixed64, length-delimited.
    /// Does NOT support: groups (deprecated), packed repeated fields beyond basic cases.
    /// 
    /// Wire format reference: https://protobuf.dev/programming-guides/encoding/
    /// </summary>
    internal ref struct ProtoReader
    {
        private ReadOnlySpan<byte> _data;
        private int _pos;

        public ProtoReader(ReadOnlySpan<byte> data)
        {
            _data = data;
            _pos = 0;
        }

        public bool IsEnd
        {
            get
            {
                return _pos >= _data.Length;
            }
        }

        public int Position
        {
            get
            {
                return _pos;
            }
        }

        public int Length
        {
            get
            {
                return _data.Length;
            }
        }

        /// <summary>
        /// Reads a tag (field number + wire type combined). Returns -1 at end of stream.
        /// </summary>
        public int ReadTag()
        {
            if (IsEnd)
            {
                return -1;
            }

            return (int)ReadVarint();
        }

        public static int GetFieldNumber(int tag)
        {
            return tag >> 3;
        }
        public static WireType GetWireType(int tag)
        {
            return (WireType)(tag & 0x07);
        }

        public ulong ReadVarint()
        {
            ulong result = 0;
            var shift = 0;

            while (true)
            {
                if (_pos >= _data.Length)
                {
                    throw new InvalidDataException("Unexpected end of varint.");
                }

                var b = _data[_pos++];
                result |= (ulong)(b & 0x7F) << shift;

                if ((b & 0x80) == 0)
                {
                    return result;
                }

                shift += 7;

                if (shift >= 64)
                {
                    throw new InvalidDataException("Varint exceeds 64 bits.");
                }
            }
        }

        public int ReadInt32()
        {
            return (int)ReadVarint();
        }
        
        public long ReadInt64()
        {
            return (long)ReadVarint();
        }

        public uint ReadFixed32()
        {
            if (_pos + 4 > _data.Length)
            {
                throw new InvalidDataException("Unexpected end of fixed32.");
            }

            var value = BinaryPrimitives.ReadUInt32LittleEndian(_data.Slice(_pos));
            _pos += 4;
            return value;
        }

        public ulong ReadFixed64()
        {
            if (_pos + 8 > _data.Length)
            {
                throw new InvalidDataException("Unexpected end of fixed64.");
            }

            var value = BinaryPrimitives.ReadUInt64LittleEndian(_data.Slice(_pos));
            _pos += 8;
            return value;
        }

        public float ReadFloat()
        {
            return BitConverter.UInt32BitsToSingle(ReadFixed32());
        }

        public double ReadDouble()
        {
            return BitConverter.Int64BitsToDouble((long)ReadFixed64());
        }

        /// <summary>
        /// Reads length-delimited bytes (returns slice, no copy).
        /// </summary>
        public ReadOnlySpan<byte> ReadBytes()
        {
            var len = (int)ReadVarint();

            if (_pos + len > _data.Length)
            {
                throw new InvalidDataException($"Length-delimited field exceeds buffer ({len} bytes at pos {_pos}, total {_data.Length}).");
            }

            var slice = _data.Slice(_pos, len);
            _pos += len;
            return slice;
        }

        public string ReadString()
        {
            return Encoding.UTF8.GetString(ReadBytes());
        }

        /// <summary>
        /// Skips a field of given wire type, advancing position past it.
        /// Used to ignore unknown/unused fields.
        /// </summary>
        public void SkipField(WireType wireType)
        {
            switch (wireType)
            {
                case WireType.Varint:
                    ReadVarint();
                    break;
                case WireType.Fixed64:
                    if (_pos + 8 > _data.Length)
                    {
                        throw new InvalidDataException("Unexpected end skipping fixed64.");
                    }
                    _pos += 8;
                    break;
                case WireType.LengthDelimited:
                    var len = (int)ReadVarint();
                    if (_pos + len > _data.Length)
                    {
                        throw new InvalidDataException("Unexpected end skipping length-delimited.");
                    }
                    _pos += len;
                    break;
                case WireType.Fixed32:
                    if (_pos + 4 > _data.Length)
                    {
                        throw new InvalidDataException("Unexpected end skipping fixed32.");
                    }
                    _pos += 4;
                    break;
                default:
                    throw new InvalidDataException($"Unsupported wire type: {wireType}");
            }
        }

        /// <summary>
        /// Reads a length-delimited sub-message and returns a reader scoped to it.
        /// </summary>
        public ProtoReader ReadSubMessage()
        {
            return new ProtoReader(ReadBytes());
        }

        /// <summary>
        /// Reads a packed repeated int64 field (length-delimited block of varints).
        /// </summary>
        public List<long> ReadPackedInt64()
        {
            var result = new List<long>();
            var bytes = ReadBytes();
            var sub = new ProtoReader(bytes);

            while (!sub.IsEnd)
            {
                result.Add((long)sub.ReadVarint());
            }

            return result;
        }

        /// <summary>
        /// Reads a packed repeated float field.
        /// </summary>
        public float[] ReadPackedFloat()
        {
            var bytes = ReadBytes();

            if (bytes.Length % 4 != 0)
            {
                throw new InvalidDataException($"Packed float data length {bytes.Length} not divisible by 4.");
            }

            var result = new float[bytes.Length / 4];

            for (var i = 0; i < result.Length; i++)
            {
                result[i] = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(i * 4, 4)));
            }

            return result;
        }
    }

}