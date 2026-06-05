// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
using System.Text;

namespace DevOnBike.Overfit.LanguageModels.Whisper
{
    /// <summary>
    /// Loads a whisper.cpp <c>ggml-*.bin</c> model (the de-facto Whisper format, e.g. <c>ggml-tiny.bin</c>) —
    /// pure managed, no native deps. Parses the exact write order of whisper.cpp's
    /// <c>convert-pt-to-ggml.py</c>: magic <c>0x67676d6c</c> → 10 hparams + f16 flag → mel filters → vocab →
    /// tensors. F16 tensors are dequantized to F32 on load; ggml's reversed dimension order is un-reversed
    /// to logical shape.
    /// </summary>
    public static class WhisperGgmlLoader
    {
        private const int Magic = 0x67676d6c; // "ggml"

        public static WhisperModel Load(string path)
        {
            using var fs = File.OpenRead(path);
            return Load(fs);
        }

        public static WhisperModel Load(Stream stream)
        {
            using var br = new BinaryReader(stream);

            if (br.ReadInt32() != Magic)
            {
                throw new OverfitFormatException("Not a whisper ggml file (bad magic). Expected a whisper.cpp ggml-*.bin.");
            }

            var config = new WhisperConfig(
                NVocab: br.ReadInt32(),
                NAudioCtx: br.ReadInt32(),
                NAudioState: br.ReadInt32(),
                NAudioHead: br.ReadInt32(),
                NAudioLayer: br.ReadInt32(),
                NTextCtx: br.ReadInt32(),
                NTextState: br.ReadInt32(),
                NTextHead: br.ReadInt32(),
                NTextLayer: br.ReadInt32(),
                NMels: br.ReadInt32(),
                F16: br.ReadInt32() != 0);

            // ── mel filters ──
            var melRows = br.ReadInt32();
            var melCols = br.ReadInt32();
            var melFilters = new float[(long)melRows * melCols];
            for (var i = 0; i < melFilters.Length; i++) { melFilters[i] = br.ReadSingle(); }

            // ── vocab (byte-level BPE strings; specials are computed at use, not stored) ──
            var nTokens = br.ReadInt32();
            var vocab = new string[nTokens];
            for (var i = 0; i < nTokens; i++)
            {
                var len = br.ReadInt32();
                var bytes = br.ReadBytes(len);
                vocab[i] = Encoding.UTF8.GetString(bytes);
            }

            // ── tensors (until EOF) ──
            var tensors = new Dictionary<string, WhisperTensor>();
            while (stream.Position < stream.Length)
            {
                var nDims = br.ReadInt32();
                var nameLen = br.ReadInt32();
                var ftype = br.ReadInt32(); // 0 = f32, 1 = f16

                // Dimensions are written reversed (ggml ne[]); un-reverse to logical shape.
                var ne = new int[nDims];
                for (var i = 0; i < nDims; i++) { ne[i] = br.ReadInt32(); }
                var shape = new int[nDims];
                for (var i = 0; i < nDims; i++) { shape[i] = ne[nDims - 1 - i]; }

                var name = Encoding.UTF8.GetString(br.ReadBytes(nameLen));

                long count = 1;
                for (var i = 0; i < nDims; i++) { count *= shape[i]; }

                var data = new float[count];
                if (ftype == 0)
                {
                    for (var i = 0L; i < count; i++) { data[i] = br.ReadSingle(); }
                }
                else if (ftype == 1)
                {
                    for (var i = 0L; i < count; i++) { data[i] = (float)BitConverter.UInt16BitsToHalf(br.ReadUInt16()); }
                }
                else
                {
                    throw new OverfitRuntimeException($"Tensor '{name}' has unsupported ftype {ftype} (only F32/F16 supported so far).");
                }

                tensors[name] = new WhisperTensor(shape, data);
            }

            return new WhisperModel(config, melRows, melCols, melFilters, vocab, tensors);
        }
    }
}
