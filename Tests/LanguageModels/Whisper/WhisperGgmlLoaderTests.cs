// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>
    /// Validates <see cref="WhisperGgmlLoader"/> against the exact whisper.cpp ggml write order (magic →
    /// 10 hparams + f16 flag → mel filters → vocab → tensors, dims reversed). A hand-built file (written to
    /// that spec) round-trips: config, mel filterbank, vocab and an F32 + an F16 tensor all read back, with
    /// the reversed ggml dimensions un-reversed to logical shape.
    /// </summary>
    public sealed class WhisperGgmlLoaderTests
    {
        [Fact]
        public void Load_SyntheticGgml_RoundTripsAllSections()
        {
            var bytes = BuildGgml();
            using var ms = new MemoryStream(bytes);
            var model = WhisperGgmlLoader.Load(ms);

            // hparams
            var c = model.Config;
            Assert.Equal(51865, c.NVocab);
            Assert.Equal(1500, c.NAudioCtx);
            Assert.Equal(384, c.NAudioState);
            Assert.Equal(6, c.NAudioHead);
            Assert.Equal(4, c.NAudioLayer);
            Assert.Equal(448, c.NTextCtx);
            Assert.Equal(384, c.NTextState);
            Assert.Equal(6, c.NTextHead);
            Assert.Equal(4, c.NTextLayer);
            Assert.Equal(80, c.NMels);
            Assert.True(c.F16);
            Assert.True(c.IsMultilingual);

            // mel filters [2 × 3]
            Assert.Equal(2, model.MelFilterRows);
            Assert.Equal(3, model.MelFilterCols);
            Assert.Equal(new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f }, model.MelFilters);

            // vocab
            Assert.Equal(new[] { "hello", "world" }, model.Vocab);

            // F32 tensor "a.weight" [2,3] — logical shape (ggml stored reversed [3,2]).
            var a = model.Tensors["a.weight"];
            Assert.Equal(new[] { 2, 3 }, a.Shape);
            Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, a.Data);

            // F16 tensor "b.weight" [2] — dequantized to F32.
            var b = model.Tensors["b.weight"];
            Assert.Equal(new[] { 2 }, b.Shape);
            Assert.True(Math.Abs(b.Data[0] - 1.5f) < 1e-3 && Math.Abs(b.Data[1] - (-2.5f)) < 1e-3);
        }

        private static byte[] BuildGgml()
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            bw.Write(0x67676d6c);                 // magic
            // 10 hparams
            foreach (var h in new[] { 51865, 1500, 384, 6, 4, 448, 384, 6, 4, 80 })
            {
                bw.Write(h);
            }
            bw.Write(1);                          // f16 flag

            // mel filters
            bw.Write(2);
            bw.Write(3);
            foreach (var f in new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f })
            {
                bw.Write(f);
            }

            // vocab
            bw.Write(2);
            WriteToken(bw, "hello");
            WriteToken(bw, "world");

            // tensor a.weight [2,3] f32 — dims written reversed (ne = [3,2])
            WriteTensorHeader(bw, dimsReversed: new[] { 3, 2 }, name: "a.weight", ftype: 0);
            foreach (var v in new[] { 1f, 2f, 3f, 4f, 5f, 6f })
            {
                bw.Write(v);
            }

            // tensor b.weight [2] f16
            WriteTensorHeader(bw, dimsReversed: new[] { 2 }, name: "b.weight", ftype: 1);
            bw.Write(BitConverter.HalfToUInt16Bits((Half)1.5f));
            bw.Write(BitConverter.HalfToUInt16Bits((Half)(-2.5f)));

            bw.Flush();
            return ms.ToArray();
        }

        private static void WriteToken(BinaryWriter bw, string token)
        {
            var b = Encoding.UTF8.GetBytes(token);
            bw.Write(b.Length);
            bw.Write(b);
        }

        private static void WriteTensorHeader(BinaryWriter bw, int[] dimsReversed, string name, int ftype)
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            bw.Write(dimsReversed.Length); // n_dims
            bw.Write(nameBytes.Length);    // name length
            bw.Write(ftype);
            foreach (var d in dimsReversed)
            {
                bw.Write(d);
            }
            bw.Write(nameBytes);
        }
    }
}
