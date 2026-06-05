// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Parameters;

namespace DevOnBike.Overfit.LanguageModels.Embeddings
{
    /// <summary>
    /// Loads a HuggingFace BERT-family encoder (e.g. sentence-transformers/all-MiniLM-L6-v2) from a raw
    /// <c>model.safetensors</c> directly into a <see cref="BertEncoder"/> — no Python, no conversion step.
    ///
    /// The two non-trivial mappings:
    /// <list type="bullet">
    /// <item>PyTorch <c>Linear.weight</c> is <c>[out, in]</c>; Overfit's <see cref="LinearLayer"/> /
    /// <see cref="FeedForwardLayer"/> store <c>[in, out]</c> — so FFN weights are transposed on load.</item>
    /// <item>HF packs Q/K/V/O as combined <c>[d, d]</c> matrices; Overfit's
    /// <see cref="MultiHeadAttentionLayer"/> is factored per head. After transposing to <c>[in, out]</c>,
    /// each head's Q/K/V is a <c>dHead</c>-wide column block and each head's output projection is a
    /// contiguous <c>dHead</c>-tall row block.</item>
    /// </list>
    /// Embedding tables and LayerNorm γ/β copy straight through. The HF key prefix
    /// (none / <c>bert.</c> / sentence-transformers wrappers) is auto-detected.
    /// </summary>
    public static class BertSafetensorsLoader
    {
        private static readonly string[] _candidatePrefixes =
        {
            "", "bert.", "model.", "0.auto_model.", "auto_model.",
        };

        /// <summary>Loads weights from <paramref name="safetensorsPath"/> into a freshly-built encoder.</summary>
        public static BertEncoder Load(string safetensorsPath, BertConfig config)
        {
            ArgumentException.ThrowIfNullOrEmpty(safetensorsPath);
            ArgumentNullException.ThrowIfNull(config);

            var encoder = new BertEncoder(config);
            try
            {
                using var reader = new SafetensorsReader(safetensorsPath);
                LoadInto(reader, encoder, config);
                encoder.Eval();
                return encoder;
            }
            catch
            {
                encoder.Dispose();
                throw;
            }
        }

        private static void LoadInto(SafetensorsReader reader, BertEncoder enc, BertConfig config)
        {
            var prefix = DetectPrefix(reader);
            var d = config.HiddenSize;
            var dFF = config.IntermediateSize;
            var nHeads = config.NumHeads;
            var dHead = d / nHeads;

            // ── embeddings (direct copy) ──
            LoadDirect(reader, prefix + "embeddings.word_embeddings.weight", enc.WordEmbeddings.Weight.DataSpan);
            LoadDirect(reader, prefix + "embeddings.position_embeddings.weight", enc.PositionEmbeddings.Weight.DataSpan);
            LoadDirect(reader, prefix + "embeddings.token_type_embeddings.weight", enc.TokenTypeEmbeddings.Weight.DataSpan);
            LoadDirect(reader, prefix + "embeddings.LayerNorm.weight", enc.EmbeddingLayerNorm.Gamma.DataSpan);
            LoadDirect(reader, prefix + "embeddings.LayerNorm.bias", enc.EmbeddingLayerNorm.Beta.DataSpan);

            // Scratch reused across layers: raw HF tensor + its transpose. Largest is dFF×d.
            var maxLen = Math.Max(d * d, dFF * d);
            var raw = new float[maxLen];
            var transposed = new float[maxLen];

            for (var i = 0; i < config.NumLayers; i++)
            {
                var lp = prefix + "encoder.layer." + i + ".";
                var block = enc.Layers[i];
                var attn = block.Attention;

                // ── self-attention Q/K/V: HF [d,d] (out,in) → transpose → per-head [d,dHead] column blocks ──
                LoadHeadProjection(reader, lp + "attention.self.query.weight", d, dHead, nHeads, attn.WqHeads, raw, transposed);
                LoadHeadProjection(reader, lp + "attention.self.key.weight", d, dHead, nHeads, attn.WkHeads, raw, transposed);
                LoadHeadProjection(reader, lp + "attention.self.value.weight", d, dHead, nHeads, attn.WvHeads, raw, transposed);
                LoadHeadBias(reader, lp + "attention.self.query.bias", dHead, nHeads, attn.BqHeads, raw);
                LoadHeadBias(reader, lp + "attention.self.key.bias", dHead, nHeads, attn.BkHeads, raw);
                LoadHeadBias(reader, lp + "attention.self.value.bias", dHead, nHeads, attn.BvHeads, raw);

                // ── attention output projection: HF [d,d] (out,in) → transpose → per-head [dHead,d] row blocks ──
                LoadOutputProjection(reader, lp + "attention.output.dense.weight", d, dHead, nHeads, attn.WoHeads, raw, transposed);
                LoadDirect(reader, lp + "attention.output.dense.bias", attn.Bo.DataSpan);

                // ── post-attention LayerNorm (Norm1) ──
                LoadDirect(reader, lp + "attention.output.LayerNorm.weight", block.Norm1.Gamma.DataSpan);
                LoadDirect(reader, lp + "attention.output.LayerNorm.bias", block.Norm1.Beta.DataSpan);

                // ── FFN: intermediate [dFF,d] → W1 [d,dFF]; output [d,dFF] → W2 [dFF,d] ──
                LoadTransposed(reader, lp + "intermediate.dense.weight", dFF, d, block.FFN.W1.DataSpan, raw);
                LoadDirect(reader, lp + "intermediate.dense.bias", block.FFN.B1.DataSpan);
                LoadTransposed(reader, lp + "output.dense.weight", d, dFF, block.FFN.W2.DataSpan, raw);
                LoadDirect(reader, lp + "output.dense.bias", block.FFN.B2.DataSpan);

                // ── post-FFN LayerNorm (Norm2) ──
                LoadDirect(reader, lp + "output.LayerNorm.weight", block.Norm2.Gamma.DataSpan);
                LoadDirect(reader, lp + "output.LayerNorm.bias", block.Norm2.Beta.DataSpan);
            }
        }

        private static string DetectPrefix(SafetensorsReader reader)
        {
            foreach (var p in _candidatePrefixes)
            {
                if (reader.Tensors.ContainsKey(p + "embeddings.word_embeddings.weight"))
                {
                    return p;
                }
            }

            throw new OverfitFormatException(
                "Could not locate 'embeddings.word_embeddings.weight' under any known prefix " +
                "(\"\", \"bert.\", \"model.\", \"0.auto_model.\", \"auto_model.\"). " +
                "Is this a BERT-family encoder safetensors file?");
        }

        private static void LoadDirect(SafetensorsReader reader, string name, Span<float> dst)
        {
            var count = reader.ElementCount(name);
            if (count != dst.Length)
            {
                throw new OverfitFormatException(
                    $"Tensor '{name}' has {count} elements but the target layer expects {dst.Length}. " +
                    "Check the BertConfig matches the model.");
            }

            reader.LoadF32(name, dst);
        }

        /// <summary>
        /// Per-head Q/K/V: HF weight is <c>[d, d]</c> in <c>[out, in]</c> order. Transpose to
        /// <c>[in, out]</c>, then head h is the <c>dHead</c>-wide column block <c>[h·dHead, (h+1)·dHead)</c>.
        /// </summary>
        private static void LoadHeadProjection(
            SafetensorsReader reader, string name, int d, int dHead, int nHeads,
            Parameter[] heads, float[] raw, float[] transposed)
        {
            var count = reader.ElementCount(name);
            if (count != (long)d * d)
            {
                throw new OverfitFormatException($"Tensor '{name}' element count {count} != d·d ({(long)d * d}).");
            }

            reader.LoadF32(name, raw.AsSpan(0, d * d));
            Transpose(raw, d, d, transposed); // [out,in] → [in,out]

            for (var h = 0; h < nHeads; h++)
            {
                var dst = heads[h].DataSpan; // [d, dHead]
                for (var i = 0; i < d; i++)
                {
                    var srcRow = i * d + h * dHead;
                    for (var j = 0; j < dHead; j++)
                    {
                        dst[i * dHead + j] = transposed[srcRow + j];
                    }
                }
            }
        }

        private static void LoadHeadBias(
            SafetensorsReader reader, string name, int dHead, int nHeads,
            Parameter[] heads, float[] raw)
        {
            var count = reader.ElementCount(name);
            if (count != (long)dHead * nHeads)
            {
                throw new OverfitFormatException($"Tensor '{name}' element count {count} != d ({(long)dHead * nHeads}).");
            }

            reader.LoadF32(name, raw.AsSpan(0, dHead * nHeads));
            for (var h = 0; h < nHeads; h++)
            {
                raw.AsSpan(h * dHead, dHead).CopyTo(heads[h].DataSpan);
            }
        }

        /// <summary>
        /// Attention output projection: HF weight is <c>[d, d]</c> in <c>[out, in]</c> order. Transpose to
        /// <c>[in, out]</c>; head h is then the contiguous <c>dHead</c>-tall row block — i.e. the slice
        /// <c>[h·dHead·d, (h+1)·dHead·d)</c> of the transposed buffer maps onto <c>WoHeads[h]</c> as-is.
        /// </summary>
        private static void LoadOutputProjection(
            SafetensorsReader reader, string name, int d, int dHead, int nHeads,
            Parameter[] heads, float[] raw, float[] transposed)
        {
            var count = reader.ElementCount(name);
            if (count != (long)d * d)
            {
                throw new OverfitFormatException($"Tensor '{name}' element count {count} != d·d ({(long)d * d}).");
            }

            reader.LoadF32(name, raw.AsSpan(0, d * d));
            Transpose(raw, d, d, transposed); // [out,in] → [in,out]=[d,d]

            for (var h = 0; h < nHeads; h++)
            {
                // WoHeads[h] is [dHead, d]; rows h·dHead..(h+1)·dHead of the [in=d, out=d] transpose.
                transposed.AsSpan(h * dHead * d, dHead * d).CopyTo(heads[h].DataSpan);
            }
        }

        /// <summary>Loads <paramref name="name"/> ([rows, cols], row-major) transposed into [cols, rows].</summary>
        private static void LoadTransposed(SafetensorsReader reader, string name, int rows, int cols, Span<float> dst, float[] raw)
        {
            var count = reader.ElementCount(name);
            if (count != (long)rows * cols)
            {
                throw new OverfitFormatException(
                    $"Tensor '{name}' element count {count} != {rows}·{cols} ({(long)rows * cols}). Check the BertConfig.");
            }

            if (dst.Length != rows * cols)
            {
                throw new OverfitFormatException($"Target for '{name}' has {dst.Length} elements, expected {rows * cols}.");
            }

            reader.LoadF32(name, raw.AsSpan(0, rows * cols));
            Transpose(raw, rows, cols, dst);
        }

        /// <summary>Transposes <paramref name="src"/> ([rows, cols] row-major) into <paramref name="dst"/> ([cols, rows]).</summary>
        private static void Transpose(ReadOnlySpan<float> src, int rows, int cols, Span<float> dst)
        {
            for (var r = 0; r < rows; r++)
            {
                var srcRow = r * cols;
                for (var c = 0; c < cols; c++)
                {
                    dst[c * rows + r] = src[srcRow + c];
                }
            }
        }
    }
}
