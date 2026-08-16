// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Mp3;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>
    /// A syntactically valid MPEG-1 Layer III frame whose side info is internally inconsistent — <c>NR-5</c>.
    ///
    /// <para><b>The finding, and it is why a header check is not enough: no cross-field validation exists
    /// between <c>part2_3_length</c>, <c>big_values</c> and <c>scalefac_compress</c>.</b> Each is in range on
    /// its own, so a frame carrying this combination clears <c>Mp3FrameHeader.TryParse</c> and the side-info
    /// read without complaint, and only the interaction is impossible.</para>
    ///
    /// <para>The arithmetic that fires it, in the affected granule:</para>
    /// <list type="bullet">
    /// <item><description><c>scalefac_compress = 4</c> gives <c>slen1 = 3</c>, <c>slen2 = 0</c>, and granule 0
    /// always reads fresh scalefactors regardless of <c>scfsi</c> — so <c>(6+5)*3 + (5+5)*0 = 33</c> bits are
    /// consumed before any Huffman data.</description></item>
    /// <item><description><c>part2_3_length = 1</c> gives
    /// <c>bitPosEnd = part2Start + part23 - 1 = 0</c>.</description></item>
    /// <item><description><c>big_values = 0</c> gives <c>bigEnd = 0</c>, so the big-values loop never runs;
    /// the quad loop needs <c>BitPosition &lt;= bitPosEnd</c>, i.e. <c>33 &lt;= 0</c>, so it never runs
    /// either. <c>pos</c> is still 0.</description></item>
    /// <item><description>The overshoot test <c>33 &gt; 0 + 1</c> then holds and subtracts a quad that was
    /// never read, taking <c>pos</c> to <b>-4</b>. The zero-fill loop that follows writes from there.
    /// </description></item>
    /// </list>
    ///
    /// <para><b>Two defects, one cause, and which slot is hit depends on <c>g = gr * 2 + ch</c>.</b> For
    /// <c>g = 0</c>, <c>isBase</c> is 0 and the write is <c>_is[-4]</c> — an
    /// <see cref="IndexOutOfRangeException"/> out of a public read of a user-supplied file. For any later
    /// <c>g</c> it lands <i>inside</i> the array, in the preceding slot, and silently zeroes four samples of
    /// other audio. Note the indexing: a MONO stream only ever uses <c>ch = 0</c>, so its granule 1 is
    /// <c>g = 2</c> and its underflow writes into <c>g = 1</c>, a channel a mono decode never reads. The
    /// silent case therefore needs STEREO, where granule 0 / channel 1 (<c>g = 1</c>) writes back into
    /// channel 0's live region.</para>
    ///
    /// <para><b>What these tests do and do not cover.</b> The crash is pinned directly. The stereo case
    /// exercises the real mechanism — an in-bounds write into a live slot — but still cannot assert the
    /// corruption: channel 0 there carries <c>part2_3_length = 0</c>, so the four samples it clobbers are
    /// already zero and no assertion can see zero overwritten with zero. Showing a value change needs a
    /// channel holding real Huffman-decoded data, which needs valid codewords from
    /// <c>Mp3HuffmanData.Table</c> rather than a hand-built frame. The clamp fixes both by construction,
    /// because <c>isBase + pos</c> can no longer go below <c>isBase</c>; only the crash is pinned by a test.
    /// </para>
    ///
    /// <para>Both frames were derived twice and independently — here from the side-info field table, and by
    /// <c>nasa-mp3</c> from a Python re-implementation of the header, side-info and scalefactor reads. The
    /// two derivations agree byte for byte, which is what <see cref="TheBuiltFramesMatchTheVerifiedBytes"/>
    /// keeps true.</para>
    /// </summary>
    public sealed class Mp3MalformedFrameTests
    {
        /// <summary>
        /// Granule 0 claims one bit for a Huffman region its own scalefactors have already overrun.
        /// Before the clamp this threw <see cref="IndexOutOfRangeException"/> from
        /// <see cref="Mp3Reader.ReadMono(Stream, out int)"/> — an uncaught crash on an untrusted file.
        /// </summary>
        [Fact]
        public void AGranuleClaimingFewerBitsThanItsScalefactors_IsDecodedRatherThanCrashing()
        {
            using var stream = new MemoryStream(MonoCrashFrame());
            var samples = Mp3Reader.ReadMono(stream, out var sampleRate);

            Assert.Equal(44100, sampleRate);
            Assert.Equal(1152, samples.Length);          // two granules of 576

            // Exact silence, and it pins the CHOICE rather than merely the absence of a crash: the clamp
            // sets _count1 to 0, meaning the granule decoded no coefficients, so the spectrum is all zero
            // and a linear synthesis of it with a zero-initialised overlap buffer is exactly zero. A clamp
            // that guarded only the array write while leaving _count1 negative would not owe this.
            Assert.All(samples, s => Assert.Equal(0f, s));
        }

        /// <summary>
        /// The same inconsistency in granule 0 / channel 1 of a stereo frame, where the underflow stays in
        /// bounds and reaches back into channel 0. This pins that the frame decodes; it cannot observe the
        /// corruption itself, for the reason given on the class.
        /// </summary>
        [Fact]
        public void TheSameInconsistencyInASecondChannel_IsDecodedRatherThanReachingIntoTheFirst()
        {
            using var stream = new MemoryStream(StereoSilentFrame());
            var samples = Mp3Reader.ReadMono(stream, out var sampleRate);

            Assert.Equal(44100, sampleRate);
            Assert.Equal(1152, samples.Length);
            Assert.All(samples, s => Assert.Equal(0f, s));
        }

        /// <summary>
        /// The control. A frame built the same way but internally consistent must decode, or the two tests
        /// above would pass on a frame the decoder rejected for some unrelated reason.
        /// </summary>
        [Fact]
        public void AConsistentFrameBuiltTheSameWay_Decodes()
        {
            var frame = Frame(0xC0, privateBits: 5, channels: 1, Granule(), Granule());

            using var stream = new MemoryStream(frame);
            var samples = Mp3Reader.ReadMono(stream, out var sampleRate);

            Assert.Equal(44100, sampleRate);
            Assert.Equal(1152, samples.Length);
        }

        /// <summary>
        /// Pins the frame builder against the same bytes derived independently from a Python model of the
        /// header / side-info / scalefactor reads. Without this, a mistake in the bit packing below turns
        /// the three tests above into tests of some other frame — passing, and meaningless.
        /// </summary>
        [Fact]
        public void TheBuiltFramesMatchTheVerifiedBytes()
        {
            byte[] mono =
            [
                0xFF, 0xFB, 0x10, 0xC0, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ];
            byte[] stereo =
            [
                0xFF, 0xFB, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x20, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ];

            AssertFrame(mono, MonoCrashFrame());
            AssertFrame(stereo, StereoSilentFrame());
        }

        /// <summary>Header plus side info must match exactly; everything after it is 0xAA filler.</summary>
        private static void AssertFrame(byte[] expectedPrefix, byte[] actual)
        {
            Assert.Equal(104, actual.Length);
            Assert.Equal(expectedPrefix, actual.AsSpan(0, expectedPrefix.Length).ToArray());
            Assert.All(
                actual.AsSpan(expectedPrefix.Length).ToArray(), b => Assert.Equal(0xAA, b));
        }

        // ── frame construction ──

        private static byte[] MonoCrashFrame()
        {
            return Frame(
                0xC0, privateBits: 5, channels: 1,
                Granule(part23: 1, scalefacCompress: 4), Granule());
        }

        /// <summary>
        /// Stereo, so that the malicious granule is <c>g = 1</c> and its underflow lands in channel 0's
        /// live region rather than in an unused slot. Side info is 32 bytes here against 17 for mono:
        /// <c>private_bits</c> is 3 rather than 5, <c>scfsi</c> is 4 bits per channel, and there are four
        /// granule blocks in <c>gr0ch0, gr0ch1, gr1ch0, gr1ch1</c> order.
        /// </summary>
        private static byte[] StereoSilentFrame()
        {
            return Frame(
                0x00, privateBits: 3, channels: 2,
                Granule(), Granule(part23: 1, scalefacCompress: 4), Granule(), Granule());
        }

        /// <summary>
        /// One MPEG-1 Layer III frame: 4-byte header, side info, filler to 104 bytes.
        ///
        /// <para>104 is not a chosen number — it is <c>144 * 32000 / 44100</c> for the bitrate and sample
        /// rate in the header. The filler is never read: <c>part2_3_length</c> bounds the main data, and
        /// every granule here claims at most one bit of it.</para>
        /// </summary>
        private static byte[] Frame(byte modeByte, int privateBits, int channels, params List<byte>[] granules)
        {
            var side = new List<byte>();
            Put(side, 0, 9);                       // main_data_begin = 0 — decode now, not via the reservoir
            Put(side, 0, privateBits);
            Put(side, 0, 4 * channels);            // scfsi

            foreach (var granule in granules)
            {
                side.AddRange(granule);
            }

            // The bit total is what proves the layout: 17 bytes for mono, 32 for stereo.
            Assert.Equal(0, side.Count % 8);
            Assert.Equal(9 + privateBits + (4 * channels) + (59 * granules.Length), side.Count);

            // FF FB: sync, MPEG-1, Layer III, no CRC. 10: 32 kbps at 44100 Hz. C0 mono / 00 stereo.
            var frame = new List<byte> { 0xFF, 0xFB, 0x10, modeByte };
            frame.AddRange(Pack(side));

            while (frame.Count < 104)
            {
                frame.Add(0xAA);
            }

            return frame.ToArray();
        }

        /// <summary>
        /// One granule's side info with <c>window_switching_flag = 0</c> — 59 bits.
        /// </summary>
        private static List<byte> Granule(int part23 = 0, int bigValues = 0, int scalefacCompress = 0)
        {
            var bits = new List<byte>();
            Put(bits, part23, 12);
            Put(bits, bigValues, 9);
            Put(bits, 0, 8);                       // global_gain
            Put(bits, scalefacCompress, 4);
            Put(bits, 0, 1);                       // window_switching_flag = 0 → long block
            Put(bits, 0, 15);                      // table_select ×3
            Put(bits, 0, 4);                       // region0_count
            Put(bits, 0, 3);                       // region1_count
            Put(bits, 0, 1);                       // preflag
            Put(bits, 0, 1);                       // scalefac_scale
            Put(bits, 0, 1);                       // count1table_select

            Assert.Equal(59, bits.Count);

            return bits;
        }

        private static void Put(List<byte> bits, int value, int width)
        {
            for (var i = width - 1; i >= 0; i--)
            {
                bits.Add((byte)((value >> i) & 1));
            }
        }

        private static byte[] Pack(IReadOnlyList<byte> bits)
        {
            var bytes = new byte[bits.Count / 8];

            for (var i = 0; i < bits.Count; i++)
            {
                bytes[i / 8] = (byte)((bytes[i / 8] << 1) | bits[i]);
            }

            return bytes;
        }
    }
}
