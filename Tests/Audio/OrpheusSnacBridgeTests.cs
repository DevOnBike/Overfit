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
        public void CustomTokenNumber_IsInverseOf_DecodeCustomToken()
        {
            for (var pos = 0; pos < 9; pos++)
            {
                var n = OrpheusSnacBridge.CustomTokenNumber(1234, pos);
                Assert.Equal(1234, OrpheusSnacBridge.DecodeCustomToken(n, pos));
            }
        }

        [Fact]
        public void Interleave_IsInverseOf_Redistribute()
        {
            int[] flat = [100, 101, 102, 103, 104, 105, 106, 200, 201, 202, 203, 204, 205, 206];
            var levels = OrpheusSnacBridge.Redistribute(flat);

            Assert.Equal(flat, OrpheusSnacBridge.Interleave(levels));
        }

        [Fact]
        public void FullBridge_RoundTrips_LevelsThroughTokensAndBack()
        {
            // F=3 frames → SNAC level lengths 3 / 6 / 12 (the 1:2:4 structure).
            int[][] original =
            [
                [11, 12, 13],
                [21, 22, 23, 24, 25, 26],
                [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
            ];

            // Encode side: levels → flat → custom-token numbers; decode side: numbers → codes → levels.
            var flat = OrpheusSnacBridge.Interleave(original);
            var decoded = new int[flat.Length];
            for (var i = 0; i < flat.Length; i++)
            {
                var n = OrpheusSnacBridge.CustomTokenNumber(flat[i], i);
                decoded[i] = OrpheusSnacBridge.DecodeCustomToken(n, i);
            }

            var levels = OrpheusSnacBridge.Redistribute(decoded);
            Assert.Equal(original[0], levels[0]);
            Assert.Equal(original[1], levels[1]);
            Assert.Equal(original[2], levels[2]);
        }
    }
}
