// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>
    /// MPEG Layer III Huffman decoder. Walks the packed binary-tree tables in <see cref="Mp3HuffmanData"/>
    /// (ISO/IEC 11172-3 Table B.7): each node entry is 0xHHLL — a matched code word when the high byte is zero
    /// (x=(e&gt;&gt;4)&amp;0xF, y=e&amp;0xF), otherwise step to the left/right child by adding the high/low byte
    /// (chained while a byte is ≥250 to encode skips that don't fit in 8 bits). The 32 big_values table-selects
    /// share 15 codeword blocks (16..23 reuse block 16 with different linbits, 24..31 reuse block 24); table
    /// selects 32/33 are the count1 (quadruple) tables. Stateless &amp; allocation-free — the bit source is the
    /// caller-owned <see cref="Mp3BitReader"/>.
    /// </summary>
    internal static class Mp3Huffman
    {
        // Per table_select 0..33: base offset into Mp3HuffmanData.Table, tree length, and linbits.
        private static readonly int[] Offset =
        {
            0,      // 0  (empty)
            0,      // 1
            7,      // 2
            24,     // 3
            0,      // 4  (empty)
            41,     // 5
            72,     // 6
            103,    // 7
            174,    // 8
            245,    // 9
            316,    // 10
            443,    // 11
            570,    // 12
            697,    // 13
            0,      // 14 (empty)
            1208,   // 15
            1719, 1719, 1719, 1719, 1719, 1719, 1719, 1719, // 16..23 share block 16
            2230, 2230, 2230, 2230, 2230, 2230, 2230, 2230, // 24..31 share block 24
            2742,   // 32 (count1 A)
            2773,   // 33 (count1 B)
        };

        private static readonly int[] TreeLen =
        {
            0, 7, 17, 17, 0, 31, 31, 71, 71, 71, 127, 127, 127, 511, 0, 511,
            511, 511, 511, 511, 511, 511, 511, 511,
            512, 512, 512, 512, 512, 512, 512, 512,
            31, 31,
        };

        private static readonly int[] LinBits =
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 2, 3, 4, 6, 8, 10, 13,
            4, 5, 6, 7, 8, 9, 11, 13,
            0, 0,
        };

        /// <summary>Decodes one big_values pair (x, y) for <paramref name="tableSelect"/> (0..31), applying the
        /// linbits ESC extension and sign bits.</summary>
        public static void DecodeBigValue(ref Mp3BitReader br, int tableSelect, out int x, out int y)
        {
            if (!Walk(ref br, tableSelect, out x, out y))
            {
                return;
            }

            var linbits = LinBits[tableSelect];
            if (linbits > 0 && x == 15)
            {
                x += (int)br.ReadBits(linbits);
            }
            if (x > 0 && br.ReadBit())
            {
                x = -x;
            }
            if (linbits > 0 && y == 15)
            {
                y += (int)br.ReadBits(linbits);
            }
            if (y > 0 && br.ReadBit())
            {
                y = -y;
            }
        }

        /// <summary>Decodes one count1 quadruple (v, w, x, y) for table A (<paramref name="tableSelect"/>=32)
        /// or B (33), with sign bits.</summary>
        public static void DecodeQuad(ref Mp3BitReader br, int tableSelect, out int v, out int w, out int x, out int y)
        {
            Walk(ref br, tableSelect, out _, out var q); // count1 packs the 4-bit value in the low nibble (y)
            v = (q >> 3) & 1;
            w = (q >> 2) & 1;
            x = (q >> 1) & 1;
            y = q & 1;
            if (v > 0 && br.ReadBit())
            {
                v = -v;
            }
            if (w > 0 && br.ReadBit())
            {
                w = -w;
            }
            if (x > 0 && br.ReadBit())
            {
                x = -x;
            }
            if (y > 0 && br.ReadBit())
            {
                y = -y;
            }
        }

        private static bool Walk(ref Mp3BitReader br, int tableSelect, out int x, out int y)
        {
            var treelen = TreeLen[tableSelect];
            if (treelen == 0)
            {
                x = 0;
                y = 0;
                return false;
            }

            var ht = Mp3HuffmanData.Table;
            var off = Offset[tableSelect];
            var point = 0;
            var bitsleft = 32;
            do
            {
                var e = ht[off + point];
                if ((e & 0xff00) == 0)
                {
                    x = (e >> 4) & 0xf;
                    y = e & 0xf;
                    return true;
                }
                if (br.ReadBit())
                {
                    while ((ht[off + point] & 0xff) >= 250)
                    {
                        point += ht[off + point] & 0xff;
                    }
                    point += ht[off + point] & 0xff;
                }
                else
                {
                    while ((ht[off + point] >> 8) >= 250)
                    {
                        point += ht[off + point] >> 8;
                    }
                    point += ht[off + point] >> 8;
                }
            }
            while (--bitsleft > 0 && point < treelen);

            x = 0; // illegal code in data
            y = 0;
            return false;
        }
    }
}
