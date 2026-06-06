// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Orpheus;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>The Orpheus→SNAC token bridge — the off-by-one-critical glue. Validates the custom-token parse, the
    /// per-position offset formula, and the 7→(1,2,4) redistribute, capped by a full round trip: known code levels
    /// → interleave → encode as custom-token numbers → decode → redistribute reproduces the levels exactly.
    /// Model-free.</summary>
    public sealed class OrpheusSnacBridgeTests
    {
        [Fact]
        public void DecodeCustomToken_RemovesPerPositionOffset()
        {
            // The model emits N = code + 10 + (pos%7)*4096; decoding must recover `code` at every frame position.
            for (var pos = 0; pos < 7; pos++)
            {
                var code = 1234;
                var n = code + 10 + (pos * 4096);
                Assert.Equal(code, OrpheusSnacBridge.DecodeCustomToken(n, pos));
                // index wraps mod 7
                Assert.Equal(code, OrpheusSnacBridge.DecodeCustomToken(n, pos + 7));
            }
        }

        [Theory]
        [InlineData("<custom_token_1234>", true, 1234)]
        [InlineData("blah <custom_token_5>", true, 5)]
        [InlineData("<custom_token_1><custom_token_42>", true, 42)] // reference scans from the right
        [InlineData("hello world", false, 0)]
        [InlineData("<custom_token_>", false, 0)]
        public void TryReadCustomTokenNumber_ParsesLastToken(string text, bool ok, int expected)
        {
            var parsed = OrpheusSnacBridge.TryReadCustomTokenNumber(text, out var number);
            Assert.Equal(ok, parsed);
            if (ok)
            {
                Assert.Equal(expected, number);
            }
        }

        [Fact]
        public void Redistribute_FansFrameInto_OneTwoFourLevels()
        {
            // One frame: positions 0..6 → L0:[0]  L1:[1,4]  L2:[2,3,5,6]
            int[] frame = [100, 101, 102, 103, 104, 105, 106];

            var levels = OrpheusSnacBridge.Redistribute(frame);

            Assert.Equal([100], levels[0]);
            Assert.Equal([101, 104], levels[1]);
            Assert.Equal([102, 103, 105, 106], levels[2]);
        }

        [Fact]
        public void Redistribute_OutOfRangeCode_Throws()
        {
            int[] frame = [0, 1, 2, 3, 4, 5, 9000];
            Assert.Throws<OverfitRuntimeException>(() => OrpheusSnacBridge.Redistribute(frame));
        }

        [Fact]
        public void FullBridge_RoundTrips_LevelsThroughTokensAndBack()
        {
            // F=3 frames → SNAC level lengths 3 / 6 / 12 (the 1:2:4 structure).
            int[] level0 = [11, 12, 13];
            int[] level1 = [21, 22, 23, 24, 25, 26];
            int[] level2 = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42];

            // Interleave back to the flat 7-per-frame stream (inverse of Redistribute).
            var flat = Interleave(level0, level1, level2);

            // Encode each code as the model would (custom-token number), then decode with the running index.
            var decoded = new int[flat.Length];
            for (var i = 0; i < flat.Length; i++)
            {
                var n = flat[i] + 10 + ((i % 7) * 4096);
                decoded[i] = OrpheusSnacBridge.DecodeCustomToken(n, i);
            }

            var levels = OrpheusSnacBridge.Redistribute(decoded);

            Assert.Equal(level0, levels[0]);
            Assert.Equal(level1, levels[1]);
            Assert.Equal(level2, levels[2]);
        }

        // Inverse of Redistribute: 3 levels → flat 7-per-frame stream.
        private static int[] Interleave(int[] l0, int[] l1, int[] l2)
        {
            var frames = l0.Length;
            var flat = new int[frames * 7];
            for (var j = 0; j < frames; j++)
            {
                var i = 7 * j;
                flat[i] = l0[j];
                flat[i + 1] = l1[2 * j];
                flat[i + 4] = l1[(2 * j) + 1];
                flat[i + 2] = l2[4 * j];
                flat[i + 3] = l2[(4 * j) + 1];
                flat[i + 5] = l2[(4 * j) + 2];
                flat[i + 6] = l2[(4 * j) + 3];
            }
            return flat;
        }
    }
}
