// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>
    /// MSB-first bit reader over a byte buffer — the bit granularity MPEG audio side-info and main-data
    /// (Huffman) need. Reads proceed from the most-significant bit of each byte. No allocation; the buffer
    /// is borrowed (not copied).
    /// </summary>
    internal struct Mp3BitReader
    {
        private readonly byte[] _data;
        private int _bitPos; // absolute bit position from the start of _data

        public Mp3BitReader(byte[] data, int byteOffset = 0)
        {
            _data = data;
            _bitPos = byteOffset * 8;
        }

        /// <summary>Current absolute bit position.</summary>
        public int BitPosition => _bitPos;

        /// <summary>Sets the absolute bit position (used by the bit-reservoir to seek into main data).</summary>
        public void SetBitPosition(int bitPos) => _bitPos = bitPos;

        /// <summary>Reads <paramref name="n"/> bits (0..32) MSB-first as an unsigned value.</summary>
        public uint ReadBits(int n)
        {
            uint result = 0;
            for (var i = 0; i < n; i++)
            {
                var byteIndex = _bitPos >> 3;
                var bitIndex = 7 - (_bitPos & 7);
                var bit = byteIndex < _data.Length ? (_data[byteIndex] >> bitIndex) & 1 : 0;
                result = (result << 1) | (uint)bit;
                _bitPos++;
            }
            return result;
        }

        /// <summary>Reads a single bit as a bool.</summary>
        public bool ReadBit() => ReadBits(1) != 0;

        /// <summary>Peeks <paramref name="n"/> bits without advancing.</summary>
        public uint PeekBits(int n)
        {
            var save = _bitPos;
            var v = ReadBits(n);
            _bitPos = save;
            return v;
        }
    }
}
