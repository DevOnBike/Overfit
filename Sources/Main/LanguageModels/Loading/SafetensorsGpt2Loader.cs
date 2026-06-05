// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
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
    /// Rather than re-implement the parameter layout, the loader produces the exact
    /// byte sequence <see cref="GPT1Model.Load"/> already reads (identical order to
    /// <c>convert_gpt2.py</c> / <see cref="GPT1Model.Save"/>) and feeds it through a
    /// <see cref="SequentialChunkReadStream"/> — one mapped parameter block at a time.
    /// Because the whole <c>.bin</c> is never buffered, peak load RAM is
    /// <c>model + one parameter block</c> rather than <c>model + full serialized copy</c>
    /// (the earlier <see cref="MemoryStream"/> round-trip's ~2× peak). Tensor names are
    /// resolved with or without the <c>transformer.</c> prefix.
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
            using var reader = SafetensorsSource.Open(safetensorsPath);
            return Load(reader, config);
        }

        public static GPT1Model Load(ISafetensorsSource reader, GPT1Config config)
        {
            if (reader is null) { throw new ArgumentNullException(nameof(reader)); }
            if (config is null) { throw new ArgumentNullException(nameof(config)); }

            var model = new GPT1Model(config);
            // Stream the mapped weights through GPT1Model.Load one block at a time —
            // no full-model MemoryStream, so only the current block's scratch lives
            // alongside the (already-allocated) model parameters.
            using var stream = new SequentialChunkReadStream(EnumerateBlocks(reader, config));
            using var binReader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            model.Load(binReader);
            return model;
        }

        // Yields the mapped GPT-2 weights as length-prefixed param blocks, in
        // GPT1Model.Load order. Each yielded byte[] is one Parameter.Load record
        // (int32 element count + little-endian float payload).
        internal static IEnumerable<byte[]> EnumerateBlocks(ISafetensorsSource reader, GPT1Config cfg)
        {
            int d = cfg.DModel, heads = cfg.NHeads, layers = cfg.NLayers;
            int vocab = cfg.VocabSize, ctx = cfg.ContextLength, dff = cfg.DFF;
            int dHead = d / heads;
            if (dHead * heads != d)
            {
                throw new OverfitFormatException($"DModel ({d}) is not divisible by NHeads ({heads}).");
            }

            // 1–2. token + position embeddings. (wte is re-read for the LM head at the
            // end rather than retained, so it does not pin ~vocab*d floats across the load.)
            yield return BuildParam(LoadTensor(reader, cfg, "wte.weight", (long)vocab * d));
            yield return BuildParam(LoadTensor(reader, cfg, "wpe.weight", (long)ctx * d));

            // 3. transformer blocks.
            for (var layer = 0; layer < layers; layer++)
            {
                var p = $"h.{layer}";

                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.ln_1.weight", d));
                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.ln_1.bias", d));

                var cAttn = LoadTensor(reader, cfg, $"{p}.attn.c_attn.weight", (long)d * 3 * d); // [d, 3d]
                var cAttnB = LoadTensor(reader, cfg, $"{p}.attn.c_attn.bias", 3L * d);           // [3d]
                var cProj = LoadTensor(reader, cfg, $"{p}.attn.c_proj.weight", (long)d * d);     // [d, d]

                // Per head: Wq, Bq, Wk, Bk, Wv, Bv, Wo  (Q/K/V cols at offset {0,1,2}*d).
                for (var h = 0; h < heads; h++)
                {
                    yield return BuildParam(ColBlock(cAttn, d, 3 * d, 0 * d + h * dHead, dHead)); // Wq [d, dHead]
                    yield return BuildParam(Range(cAttnB, 0 * d + h * dHead, dHead));             // Bq [dHead]
                    yield return BuildParam(ColBlock(cAttn, d, 3 * d, 1 * d + h * dHead, dHead)); // Wk
                    yield return BuildParam(Range(cAttnB, 1 * d + h * dHead, dHead));             // Bk
                    yield return BuildParam(ColBlock(cAttn, d, 3 * d, 2 * d + h * dHead, dHead)); // Wv
                    yield return BuildParam(Range(cAttnB, 2 * d + h * dHead, dHead));             // Bv
                    yield return BuildParam(RowBlock(cProj, d, h * dHead, dHead));                // Wo [dHead, d]
                }
                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.attn.c_proj.bias", d));

                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.ln_2.weight", d));
                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.ln_2.bias", d));

                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.mlp.c_fc.weight", (long)d * dff));   // [d, dff]
                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.mlp.c_fc.bias", dff));
                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.mlp.c_proj.weight", (long)dff * d)); // [dff, d]
                yield return BuildParam(LoadTensor(reader, cfg, $"{p}.mlp.c_proj.bias", d));
            }

            // 4. final norm.
            yield return BuildParam(LoadTensor(reader, cfg, "ln_f.weight", d));
            yield return BuildParam(LoadTensor(reader, cfg, "ln_f.bias", d));

            // 5. LM head = wte transposed to [d, vocab]. Only untied models read it;
            // GPT-2 configs are untied. Building the block in place (transpose written
            // straight into the output bytes) avoids a second ~vocab*d float scratch.
            if (!cfg.TieWeights)
            {
                var wte = LoadTensor(reader, cfg, "wte.weight", (long)vocab * d);
                yield return BuildTransposedLmHead(wte, vocab, d);
            }
        }

        // ── tensor IO + slicing helpers ─────────────────────────────────────
        private static float[] LoadTensor(ISafetensorsSource reader, GPT1Config cfg, string suffix, long expected)
        {
            var name = Resolve(reader, suffix);
            var count = reader.ElementCount(name);
            if (count != expected)
            {
                throw new OverfitFormatException(
                    $"Tensor '{name}' has {count} elements, expected {expected} for the given config.");
            }
            var data = new float[expected];
            reader.LoadF32(name, data);
            return data;
        }

        // HF GPT-2 ships these with or without the "transformer." prefix.
        private static string Resolve(ISafetensorsSource reader, string suffix)
        {
            if (reader.Tensors.ContainsKey(suffix)) { return suffix; }
            var prefixed = "transformer." + suffix;
            if (reader.Tensors.ContainsKey(prefixed)) { return prefixed; }
            throw new KeyNotFoundException(
                $"GPT-2 tensor '{suffix}' (or 'transformer.{suffix}') not found in safetensors header.");
        }

        // A length-prefixed Parameter.Load record: int32 element count + LE float payload.
        private static byte[] BuildParam(float[] data)
        {
            var bytes = new byte[sizeof(int) + (long)data.Length * sizeof(float)];
            BinaryPrimitives.WriteInt32LittleEndian(bytes, data.Length);
            MemoryMarshal.AsBytes<float>(data).CopyTo(bytes.AsSpan(sizeof(int)));
            return bytes;
        }

        // wte [vocab, d] → LM head [d, vocab], transposed straight into the record bytes.
        private static byte[] BuildTransposedLmHead(float[] wte, int vocab, int d)
        {
            var count = (long)d * vocab;
            var bytes = new byte[sizeof(int) + count * sizeof(float)];
            BinaryPrimitives.WriteInt32LittleEndian(bytes, checked((int)count));
            var payload = bytes.AsSpan(sizeof(int));
            // dst[i*vocab + v] = wte[v*d + i]
            for (var i = 0; i < d; i++)
            {
                var dstRow = i * vocab;
                for (var v = 0; v < vocab; v++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(payload.Slice((dstRow + v) * sizeof(float), sizeof(float)), wte[v * d + i]);
                }
            }
            return bytes;
        }

        private static float[] Range(float[] src, int offset, int len)
        {
            var outBuf = new float[len];
            src.AsSpan(offset, len).CopyTo(outBuf);
            return outBuf;
        }

        // Extracts a [rows, colLen] column block from a row-major [rows, srcCols] matrix.
        private static float[] ColBlock(float[] src, int rows, int srcCols, int colOffset, int colLen)
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
            return outBuf;
        }

        // Extracts a [rowLen, cols] row block from a row-major [rows, cols] matrix.
        private static float[] RowBlock(float[] src, int cols, int rowOffset, int rowLen)
        {
            var outBuf = new float[rowLen * cols];
            src.AsSpan(rowOffset * cols, rowLen * cols).CopyTo(outBuf);
            return outBuf;
        }
    }
}
