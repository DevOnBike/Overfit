// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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

#pragma warning disable OVERFIT001 // Load-time model parse — runs once per model load (mel filterbank, vocab, per-tensor shape + data), not on any serving path.
            // Every count below comes out of the file and sizes an allocation. Each is checked against the
            // bytes actually remaining, because a declared count that cannot fit is a malformed header
            // rather than a very large model — and taken at face value it is an allocation the file gets
            // to choose the size of. This is OVERFIT024's shape, four times in one method.
            // ── mel filters ──
            var melRows = br.ReadInt32();
            var melCols = br.ReadInt32();
            var melCount = (long)melRows * melCols;

            RequireFits(melRows >= 0 && melCols >= 0 && melCount * sizeof(float) <= Remaining(stream),
                $"mel filterbank {melRows}x{melCols}", stream);

            var melFilters = new float[melCount];
            for (var i = 0; i < melFilters.Length; i++)
            {
                melFilters[i] = br.ReadSingle();
            }

            // ── vocab (byte-level BPE strings; specials are computed at use, not stored) ──
            var nTokens = br.ReadInt32();

            // Four bytes is the smallest a token can be: its own length prefix and an empty string.
            RequireFits(nTokens >= 0 && (long)nTokens * sizeof(int) <= Remaining(stream),
                $"{nTokens} vocabulary entries", stream);

            var vocab = new string[nTokens];
            for (var i = 0; i < nTokens; i++)
            {
                var len = br.ReadInt32();

                RequireFits(len >= 0 && len <= Remaining(stream), $"token {i} of length {len}", stream);

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

                // ggml tensors are at most 4-dimensional. An unbounded nDims sizes two arrays and a read
                // loop straight from the file.
                RequireFits(nDims is >= 1 and <= 4, $"tensor rank {nDims}", stream);
                RequireFits(nameLen >= 0 && nameLen <= Remaining(stream),
                    $"tensor name of length {nameLen}", stream);

                // Validated HERE rather than after the read loop below, which is where it used to sit.
                // The element width is what the size bound is expressed in, so an unknown ftype has to be
                // rejected before anything is sized against it — and rejecting it after the data has been
                // read is a check standing behind the thing it guards.
                RequireFits(ftype is 0 or 1, $"tensor '{nameLen}-byte name' with ftype {ftype}", stream);

                var bytesPerElement = ftype == 0 ? sizeof(float) : sizeof(ushort);

                // Dimensions are written reversed (ggml ne[]); un-reverse to logical shape.
                var ne = new int[nDims];
                for (var i = 0; i < nDims; i++)
                {
                    ne[i] = br.ReadInt32();
                }
                var shape = new int[nDims];
                for (var i = 0; i < nDims; i++)
                {
                    shape[i] = ne[nDims - 1 - i];
                }

                var name = Encoding.UTF8.GetString(br.ReadBytes(nameLen));

                long count = 1;
                for (var i = 0; i < nDims; i++)
                {
                    RequireFits(shape[i] >= 0, $"tensor '{name}' dimension {i} is {shape[i]}", stream);

                    count *= shape[i];

                    // Bounded as it grows rather than after: the product is what overflows, and an
                    // overflowed product is small, positive and plausible.
                    //
                    // Multiplied by the element width, like the mel and vocab guards forty lines above.
                    // It was not, and comparing an element COUNT against remaining BYTES assumes one byte
                    // per element where the real minimum is two — the same "one sibling has the guard, its
                    // twin does not" shape this whole sweep was hunting, planted by hand inside a method
                    // that already contained two correct examples.
                    RequireFits(
                        count * bytesPerElement <= Remaining(stream),
                        $"tensor '{name}' with {count} elements of {bytesPerElement} bytes",
                        stream);
                }

                var data = new float[count];
                if (ftype == 0)
                {
                    for (var i = 0L; i < count; i++)
                    {
                        data[i] = br.ReadSingle();
                    }
                }
                if (ftype == 1)
                {
                    for (var i = 0L; i < count; i++)
                    {
                        data[i] = (float)BitConverter.UInt16BitsToHalf(br.ReadUInt16());
                    }
                }

                tensors[name] = new WhisperTensor(shape, data);
            }

            return new WhisperModel(config, melRows, melCols, melFilters, vocab, tensors);
#pragma warning restore OVERFIT001
        }

        /// <summary>Bytes left after the reader's current position.</summary>
        private static long Remaining(Stream stream) => stream.Length - stream.Position;

        /// <summary>
        /// Refuses a file-declared count that the file cannot back.
        ///
        /// <para>Deliberately compares against <b>bytes remaining</b> rather than a fixed ceiling. A
        /// constant would have to be either large enough to be useless or small enough to reject a real
        /// model one day; the file's own size cannot be either, and it is the bound that actually matters —
        /// no honest header asks for more data than it shipped.</para>
        /// </summary>
        private static void RequireFits(bool ok, string what, Stream stream)
        {
            if (!ok)
            {
                throw new OverfitFormatException(
                    $"whisper ggml file declares {what}, which does not fit the {Remaining(stream)} bytes "
                    + "remaining. The file is truncated or not a whisper ggml model.");
            }
        }
    }
}
