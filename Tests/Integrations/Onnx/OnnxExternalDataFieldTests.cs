// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.Onnx;

namespace DevOnBike.Overfit.Tests.Integrations.Onnx
{
    /// <summary>
    /// The <c>offset</c> and <c>length</c> of an externally-stored initializer, which arrive as free text.
    ///
    /// <para><b>Why this is a silent failure rather than a loud one.</b> Both fields are decimal strings
    /// inside the model file. When the parse of one fails it leaves the value at <c>0</c> — and <c>0</c> is
    /// a legal value for both: an offset of 0 is the start of the sidecar, and a length of 0 is the
    /// <i>read to the end</i> sentinel that <c>OnnxExternalData.ResolveOne</c> acts on. So every bound
    /// downstream passes and the loader returns weights read from the wrong region of the <c>.data</c> file.
    /// A model that is refused costs an operator a download; a model that loads and is quietly wrong costs
    /// them whatever they concluded from it.</para>
    ///
    /// <para>The parser is exercised directly rather than through <c>OnnxImporter</c>: the defect is in the
    /// parse, and a graph carrying one initializer and no nodes would be rejected by the importer for an
    /// unrelated reason before reaching it.</para>
    /// </summary>
    public sealed class OnnxExternalDataFieldTests
    {
        private const string TensorName = "weight";

        /// <summary>
        /// <b>The control, and it is what stops the rest of this file being vacuous.</b> Every other test
        /// here asserts that a hand-built model is refused, and a hand-built model with a mistake in the
        /// framing is refused too — for a reason that has nothing to do with the field under test. This one
        /// proves the builder emits a model the parser accepts, and that the two fields survive it intact.
        /// </summary>
        [Fact]
        public void WellFormedOffsetAndLength_AreParsed()
        {
            var model = ModelWithExternalInitializer(
                ("location", "weights.data"), ("offset", "128"), ("length", "16"));

            var parsed = OnnxProtoParser.ParseModel(model);

            var external = Assert.Single(parsed.Graph.Initializers).ExternalData;
            Assert.NotNull(external);
            Assert.Equal("weights.data", external.Location);
            Assert.Equal(128, external.Offset);
            Assert.Equal(16, external.Length);
        }

        /// <summary>
        /// An absent key is not a malformed one. ONNX permits both to be omitted, and the loader reads the
        /// whole sidecar in that case — so making them mandatory would refuse models that are legal today.
        /// </summary>
        [Fact]
        public void AbsentOffsetAndLength_RemainLegal()
        {
            var model = ModelWithExternalInitializer(("location", "weights.data"));

            var parsed = OnnxProtoParser.ParseModel(model);

            var external = Assert.Single(parsed.Graph.Initializers).ExternalData;
            Assert.NotNull(external);
            Assert.Equal(0, external.Offset);
            Assert.Equal(0, external.Length);
        }

        /// <summary>
        /// A non-numeric offset must refuse the file. Left unchecked it reads from byte 0 of the sidecar and
        /// returns those bytes as this tensor's weights.
        /// </summary>
        [Fact]
        public void NonNumericOffset_IsRefused()
        {
            var model = ModelWithExternalInitializer(
                ("location", "weights.data"), ("offset", "not-a-number"), ("length", "16"));

            var thrown = Assert.Throws<OverfitFormatException>(() => OnnxProtoParser.ParseModel(model));

            // The operator holding a multi-gigabyte model needs to know WHICH key is malformed and what it
            // said, or the message sends them reading the whole file by hand.
            Assert.Contains("offset", thrown.Message, StringComparison.Ordinal);
            Assert.Contains("not-a-number", thrown.Message, StringComparison.Ordinal);
            Assert.Contains(TensorName, thrown.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// The same for length, and this is the more dangerous of the two: 0 is not merely a legal length,
        /// it is the sentinel meaning "read to the end of the file", so an unparsed length is acted on as an
        /// instruction rather than ignored.
        /// </summary>
        [Fact]
        public void NonNumericLength_IsRefused()
        {
            var model = ModelWithExternalInitializer(
                ("location", "weights.data"), ("offset", "0"), ("length", "16 bytes"));

            var thrown = Assert.Throws<OverfitFormatException>(() => OnnxProtoParser.ParseModel(model));

            Assert.Contains("length", thrown.Message, StringComparison.Ordinal);
            Assert.Contains("16 bytes", thrown.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// A parsed-but-negative value is the same defect wearing a different hat, so it is refused at the
        /// same place. It would be caught downstream by <c>CheckedToInt64</c> as well, but only once the
        /// sidecar is resolved, and only with a message that has lost the text the model actually carried.
        /// </summary>
        [Theory]
        [InlineData("offset", "-1")]
        [InlineData("length", "-4096")]
        public void NegativeOffsetOrLength_IsRefused(string key, string value)
        {
            var model = ModelWithExternalInitializer(("location", "weights.data"), (key, value));

            var thrown = Assert.Throws<OverfitFormatException>(() => OnnxProtoParser.ParseModel(model));

            Assert.Contains(key, thrown.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// Locale independence. These are decimal strings in a file format, and this repository's dev box
        /// runs under a culture that groups digits with a space and would read "1 024" as 1024.
        /// </summary>
        [Fact]
        public void ACultureFormattedNumber_IsRefusedRatherThanReinterpreted()
        {
            var model = ModelWithExternalInitializer(
                ("location", "weights.data"), ("offset", "1 024"));

            Assert.Throws<OverfitFormatException>(() => OnnxProtoParser.ParseModel(model));
        }

        // ── a hand-built ONNX model, because the defect is in how bytes are read ──

        /// <summary>
        /// ModelProto { 7: GraphProto { 5: TensorProto { 8: name, 13: external_data*, 14: data_location } } }
        /// with <c>data_location = 1</c> (EXTERNAL), which is what makes the parser build the external-data
        /// record at all.
        /// </summary>
        private static byte[] ModelWithExternalInitializer(params (string Key, string Value)[] entries)
        {
            var tensor = new List<byte>();
            tensor.AddRange(StringField(8, TensorName));

            foreach (var (key, value) in entries)
            {
                var entry = new List<byte>();
                entry.AddRange(StringField(1, key));
                entry.AddRange(StringField(2, value));
                tensor.AddRange(LengthDelimited(13, entry));
            }

            tensor.AddRange(VarintField(14, 1));

            var graph = LengthDelimited(5, tensor);

            return LengthDelimited(7, graph).ToArray();
        }

        private static List<byte> Varint(ulong value)
        {
            var bytes = new List<byte>();

            while (value >= 0x80)
            {
                bytes.Add((byte)((value & 0x7F) | 0x80));
                value >>= 7;
            }

            bytes.Add((byte)value);

            return bytes;
        }

        private static List<byte> LengthDelimited(int fieldNumber, IReadOnlyCollection<byte> payload)
        {
            var bytes = Varint((ulong)((fieldNumber << 3) | 2));
            bytes.AddRange(Varint((ulong)payload.Count));
            bytes.AddRange(payload);

            return bytes;
        }

        private static List<byte> VarintField(int fieldNumber, ulong value)
        {
            var bytes = Varint((ulong)(fieldNumber << 3));
            bytes.AddRange(Varint(value));

            return bytes;
        }

        private static List<byte> StringField(int fieldNumber, string text)
        {
            return LengthDelimited(fieldNumber, System.Text.Encoding.UTF8.GetBytes(text));
        }
    }
}
