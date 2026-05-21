// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Loads a HuggingFace GPT-2 <c>safetensors</c> file straight into a
    /// <see cref="GPT1Model"/> — no Python, no <c>convert_gpt2.py</c>, no intermediate
    /// <c>.bin</c> on disk. The C# port of that script's weight mapping:
    /// <list type="bullet">
    ///   <item>HF Conv1D weights are stored <c>[in, out]</c> (Overfit's Linear layout) — used as-is.</item>
    ///   <item><c>attn.c_attn.weight [d, 3d]</c> is split into per-head Q/K/V <c>[d, dHead]</c>.</item>
    ///   <item><c>attn.c_proj.weight [d, d]</c> is split into per-head output <c>[dHead, d]</c>.</item>
    ///   <item><c>attn.c_attn.bias [3d]</c> is split into per-head Bq/Bk/Bv <c>[dHead]</c>.</item>
    ///   <item>the LM head is <c>wte.T → [d, vocab]</c> (Overfit's GPT-2 configs are untied).</item>
    /// </list>
    ///
    /// Rather than re-implement the parameter layout, the loader serialises the mapped
    /// weights into the exact byte stream <see cref="GPT1Model.Load"/> already reads
    /// (the order is identical to <c>convert_gpt2.py</c> / <see cref="GPT1Model.Save"/>),
    /// then loads it — so ordering and shapes are guaranteed by the validated load path.
    /// Tensor names are resolved with or without the <c>transformer.</c> prefix.
    /// </summary>
    public static class SafetensorsGpt2Loader
    {
        /// <summary>
        /// Loads <paramref name="safetensorsPath"/> into a new <see cref="GPT1Model"/>
        /// using the supplied GPT-2 <paramref name="config"/> (e.g. <c>Gpt2Config.Small</c>).
        /// The safetensors tensor element counts are validated against the config.
        /// </summary>
        public static GPT1Model Load(string safetensorsPath, GPT1Config config)
        {
            using var reader = new SafetensorsReader(safetensorsPath);
            return Load(reader, config);
        }

        public static GPT1Model Load(SafetensorsReader reader, GPT1Config config)
        {
            if (reader is null) { throw new ArgumentNullException(nameof(reader)); }
            if (config is null) { throw new ArgumentNullException(nameof(config)); }

            using var ms = new MemoryStream();
            using (var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                WriteBinStream(reader, config, writer);
            }

            ms.Position = 0;
            var model = new GPT1Model(config);
            using (var binReader = new BinaryReader(ms, Encoding.UTF8, leaveOpen: true))
            {
                model.Load(binReader);
            }
            return model;
        }

        // Serialises the mapped GPT-2 weights in GPT1Model.Load order.
        internal static void WriteBinStream(SafetensorsReader reader, GPT1Config cfg, BinaryWriter w)
        {
            int d = cfg.DModel, heads = cfg.NHeads, layers = cfg.NLayers;
            int vocab = cfg.VocabSize, ctx = cfg.ContextLength, dff = cfg.DFF;
            int dHead = d / heads;
            if (dHead * heads != d)
            {
                throw new InvalidDataException($"DModel ({d}) is not divisible by NHeads ({heads}).");
            }

            // 1–2. token + position embeddings.
            var wte = LoadTensor(reader, cfg, "wte.weight", (long)vocab * d);
            WriteParam(w, wte);
            WriteParam(w, LoadTensor(reader, cfg, "wpe.weight", (long)ctx * d));

            // 3. transformer blocks.
            for (var layer = 0; layer < layers; layer++)
            {
                var p = $"h.{layer}";

                WriteParam(w, LoadTensor(reader, cfg, $"{p}.ln_1.weight", d));
                WriteParam(w, LoadTensor(reader, cfg, $"{p}.ln_1.bias", d));

                var cAttn = LoadTensor(reader, cfg, $"{p}.attn.c_attn.weight", (long)d * 3 * d); // [d, 3d]
                var cAttnB = LoadTensor(reader, cfg, $"{p}.attn.c_attn.bias", 3L * d);           // [3d]
                var cProj = LoadTensor(reader, cfg, $"{p}.attn.c_proj.weight", (long)d * d);     // [d, d]

                // Per head: Wq, Bq, Wk, Bk, Wv, Bv, Wo  (Q/K/V cols at offset {0,1,2}*d).
                for (var h = 0; h < heads; h++)
                {
                    WriteColBlock(w, cAttn, d, 3 * d, 0 * d + h * dHead, dHead); // Wq [d, dHead]
                    WriteRange(w, cAttnB, 0 * d + h * dHead, dHead);            // Bq [dHead]
                    WriteColBlock(w, cAttn, d, 3 * d, 1 * d + h * dHead, dHead); // Wk
                    WriteRange(w, cAttnB, 1 * d + h * dHead, dHead);            // Bk
                    WriteColBlock(w, cAttn, d, 3 * d, 2 * d + h * dHead, dHead); // Wv
                    WriteRange(w, cAttnB, 2 * d + h * dHead, dHead);            // Bv
                    WriteRowBlock(w, cProj, d, h * dHead, dHead);              // Wo [dHead, d]
                }
                WriteParam(w, LoadTensor(reader, cfg, $"{p}.attn.c_proj.bias", d));

                WriteParam(w, LoadTensor(reader, cfg, $"{p}.ln_2.weight", d));
                WriteParam(w, LoadTensor(reader, cfg, $"{p}.ln_2.bias", d));

                WriteParam(w, LoadTensor(reader, cfg, $"{p}.mlp.c_fc.weight", (long)d * dff));   // [d, dff]
                WriteParam(w, LoadTensor(reader, cfg, $"{p}.mlp.c_fc.bias", dff));
                WriteParam(w, LoadTensor(reader, cfg, $"{p}.mlp.c_proj.weight", (long)dff * d)); // [dff, d]
                WriteParam(w, LoadTensor(reader, cfg, $"{p}.mlp.c_proj.bias", d));
            }

            // 4. final norm.
            WriteParam(w, LoadTensor(reader, cfg, "ln_f.weight", d));
            WriteParam(w, LoadTensor(reader, cfg, "ln_f.bias", d));

            // 5. LM head = wte transposed to [d, vocab].
            var lmHead = new float[(long)d * vocab];
            for (var i = 0; i < d; i++)
            {
                var row = i * vocab;
                for (var v = 0; v < vocab; v++)
                {
                    lmHead[row + v] = wte[v * d + i];
                }
            }
            WriteParam(w, lmHead);
        }

        // ── tensor IO + slicing helpers ─────────────────────────────────────
        private static float[] LoadTensor(SafetensorsReader reader, GPT1Config cfg, string suffix, long expected)
        {
            var name = Resolve(reader, suffix);
            var count = reader.ElementCount(name);
            if (count != expected)
            {
                throw new InvalidDataException(
                    $"Tensor '{name}' has {count} elements, expected {expected} for the given config.");
            }
            var data = new float[expected];
            reader.LoadF32(name, data);
            return data;
        }

        // HF GPT-2 ships these with or without the "transformer." prefix.
        private static string Resolve(SafetensorsReader reader, string suffix)
        {
            if (reader.Tensors.ContainsKey(suffix)) { return suffix; }
            var prefixed = "transformer." + suffix;
            if (reader.Tensors.ContainsKey(prefixed)) { return prefixed; }
            throw new KeyNotFoundException(
                $"GPT-2 tensor '{suffix}' (or 'transformer.{suffix}') not found in safetensors header.");
        }

        private static void WriteParam(BinaryWriter w, float[] data)
        {
            w.Write(data.Length);
            w.Write(MemoryMarshal.AsBytes<float>(data));
        }

        private static void WriteRange(BinaryWriter w, float[] src, int offset, int len)
        {
            w.Write(len);
            w.Write(MemoryMarshal.AsBytes(src.AsSpan(offset, len)));
        }

        // Extracts a [rows, colLen] column block from a row-major [rows, srcCols] matrix.
        private static void WriteColBlock(BinaryWriter w, float[] src, int rows, int srcCols, int colOffset, int colLen)
        {
            var outBuf = new float[rows * colLen];
            for (var i = 0; i < rows; i++)
            {
                var srcRow = i * srcCols + colOffset;
                var dstRow = i * colLen;
                for (var c = 0; c < colLen; c++)
                {
                    outBuf[dstRow + c] = src[srcRow + c];
                }
            }
            WriteParam(w, outBuf);
        }

        // Extracts a [rowLen, cols] row block from a row-major [rows, cols] matrix.
        private static void WriteRowBlock(BinaryWriter w, float[] src, int cols, int rowOffset, int rowLen)
        {
            var outBuf = new float[rowLen * cols];
            src.AsSpan(rowOffset * cols, rowLen * cols).CopyTo(outBuf);
            WriteParam(w, outBuf);
        }
    }
}
