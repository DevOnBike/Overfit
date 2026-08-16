// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Onnx;

namespace DevOnBike.Overfit.Tests.Integrations.Onnx
{
    /// <summary>
    /// Pins the contract for a corrupt or hostile <c>.onnx</c> file. Protobuf parsing here is hand-rolled
    /// (there is no <c>Google.Protobuf</c> dependency), so every bound is ours to enforce, and an ONNX model
    /// is something a user downloads.
    ///
    /// <para>The case that matters most is not a bad allocation — it is <b>non-termination</b>. Protobuf
    /// length-delimited fields carry a varint length; <c>ProtoReader</c> reads it as a 64-bit varint and casts
    /// to <c>int</c>. A crafted varint whose low 32 bits are negative therefore produces a <i>negative
    /// length</i>, which slips past a <c>_pos + len &gt; _data.Length</c> bounds check (adding a negative
    /// number never exceeds the limit) and then <b>moves the read position backwards</b>. Choose the value so
    /// the rewind exactly cancels the bytes just consumed and the parse loop reads the same field forever:
    /// no exception, no crash, no log line — the host simply stops responding. That is the hardest failure of
    /// all to diagnose, and it is the reason this file exists.</para>
    ///
    /// <para>Every test here is wrapped in a timeout rather than called directly: a regression must fail the
    /// test, not hang the suite.</para>
    /// </summary>
    public sealed class OnnxMalformedModelTests
    {
        private static readonly TimeSpan ParseBudget = TimeSpan.FromSeconds(5);

        /// <summary>
        /// Runs the parser under a wall-clock budget. Returns the exception it threw, or reports
        /// non-termination — the outcome this suite is really testing for.
        /// </summary>
        private static Exception? ParseWithinBudget(byte[] model)
        {
            var work = Task.Run(() => Record.Exception(() => OnnxImporter.LoadFromBytes(model)));

            if (!work.Wait(ParseBudget))
            {
                Assert.Fail(
                    $"The ONNX parser did not terminate within {ParseBudget.TotalSeconds:F0}s on a "
                    + $"{model.Length}-byte input. A malformed model must be refused, not looped on.");
            }

            return work.Result;
        }

        [Fact]
        public void NegativeLengthField_TerminatesInsteadOfLoopingForever()
        {
            // Tag 0x7A = field 15, wire type 2 (length-delimited) — an unknown ModelProto field, so the parser
            // takes its skip path. The five varint bytes decode to 4294967290, whose low 32 bits as a signed
            // int are -6: exactly the number of bytes consumed by this tag plus its length varint. An
            // unguarded `_pos += len` therefore lands back on the tag it just read, forever.
            byte[] model = [0x7A, 0xFA, 0xFF, 0xFF, 0xFF, 0x0F];

            var thrown = ParseWithinBudget(model);

            Assert.NotNull(thrown);
            Assert.IsNotType<OutOfMemoryException>(thrown);
        }

        [Fact]
        public void NegativeLengthField_InAReadRatherThanASkip_IsAlsoRefused()
        {
            // Same trick against a field the parser reads rather than skips: tag 0x0A = field 1, wire type 2.
            byte[] model = [0x0A, 0xFA, 0xFF, 0xFF, 0xFF, 0x0F];

            var thrown = ParseWithinBudget(model);

            Assert.NotNull(thrown);
            Assert.IsNotType<OutOfMemoryException>(thrown);
        }

        [Fact]
        public void LengthBeyondTheBuffer_IsRefused()
        {
            // An honest-looking length that simply exceeds what the file holds — the ordinary truncation case.
            byte[] model = [0x0A, 0xFF, 0x7F];   // field 1, length 16383, three bytes of file

            var thrown = ParseWithinBudget(model);

            Assert.NotNull(thrown);
            Assert.IsNotType<OutOfMemoryException>(thrown);
        }

        [Fact]
        public void GarbageAndStubFiles_AreRefused_WithoutHangingOrExhaustingMemory()
        {
            var inputs = new[]
            {
                Array.Empty<byte>(),
                new byte[] { 0xFF },
                new byte[] { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF },  // varint that never ends
                "not an onnx model at all"u8.ToArray(),
            };

            foreach (var model in inputs)
            {
                var thrown = ParseWithinBudget(model);

                Assert.NotNull(thrown);
                Assert.IsNotType<OutOfMemoryException>(thrown);
            }
        }
    }
}
