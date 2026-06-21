// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using LayerWeightBuffers = DevOnBike.Overfit.LanguageModels.Runtime.CachedLlamaInferenceEngine.LayerWeightBuffers;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Loads a Llama / Mistral / Qwen-family model straight from a HuggingFace
    /// <c>safetensors</c> repo (single file or sharded) into a
    /// <see cref="CachedLlamaInferenceEngine"/> — no Python, no
    /// <c>convert_llama.py</c>, no intermediate <c>.bin</c>. The C# port of that
    /// script's weight mapping, reading hyper-parameters from <c>config.json</c>
    /// (<see cref="LlamaConfigReader"/>).
    ///
    /// HF stores the projection weights in the same row-major <c>[out, in]</c> layout
    /// that the GGUF loader handles, so this is exactly <see cref="GgufLlamaLoader"/>'s
    /// F32 path with the data source swapped (and HF tensor names):
    /// <list type="bullet">
    ///   <item><c>q/k/v_proj.weight [heads*headDim, dModel]</c> → per-head <c>[dModel, headDim]</c>.</item>
    ///   <item><c>o_proj.weight [dModel, heads*headDim]</c> → per-head <c>[headDim, dModel]</c>.</item>
    ///   <item><c>mlp.gate/up_proj [dFF, dModel]</c> → transposed <c>[dModel, dFF]</c>; <c>down_proj [dModel, dFF]</c> → <c>[dFF, dModel]</c>.</item>
    ///   <item>LM head <c>[vocab, dModel]</c> (or tied <c>embed_tokens</c>) → transposed <c>[dModel, vocab]</c>.</item>
    /// </list>
    ///
    /// HF tensors are F32 / F16 / BF16, so every weight is read to F32 first; when
    /// <c>quantize</c> is set (default) and the dimensions are
    /// block-aligned, the F32 scratch is re-quantized to Q8_0-resident
    /// (<see cref="Q8Weight"/>) — the same RAM-friendly decode path the GGUF loader
    /// uses — otherwise it stays F32. Quantized HF dtypes (rare, e.g. GPTQ/AWQ) are
    /// not supported and will surface from the reader as a clear exception.
    /// </summary>
    public static class SafetensorsLlamaLoader
    {
        /// <summary>
        /// Loads from a model directory holding <c>config.json</c> and either
        /// <c>model.safetensors</c> or a <c>model.safetensors.index.json</c> shard set.
        /// </summary>
        public static CachedLlamaInferenceEngine Load(string modelDir, bool quantize = true)
        {
            var config = LlamaConfigReader.ReadFromDirectory(modelDir);
            using var source = SafetensorsSource.Open(modelDir);
            return Load(source, config, quantize);
        }

        /// <summary>
        /// Loads from an already-opened safetensors source with an explicit config
        /// (e.g. when <c>config.json</c> lives elsewhere, or for tests).
        /// </summary>
        public static CachedLlamaInferenceEngine Load(ISafetensorsSource source, GPT1Config config, bool quantize = true)
        {
            if (source is null)
            {
                throw new ArgumentNullException(nameof(source));
            }
            if (config is null)
            {
                throw new ArgumentNullException(nameof(config));
            }

            int d = config.DModel, nHeads = config.NHeads, nKv = config.KvHeads;
            int layers = config.NLayers, vocab = config.VocabSize, dff = config.DFF;
            var headDim = d / nHeads;
            if (headDim * nHeads != d)
            {
                throw new OverfitFormatException($"DModel ({d}) is not divisible by NHeads ({nHeads}).");
            }

            var block = Q8DotKernel.BlockSize;
            var attnQuant = quantize && d % block == 0 && headDim % block == 0;
            var ffnQuant = quantize && d % block == 0 && dff % block == 0;
            var lmQuant = quantize && d % block == 0;

            // ─── Embeddings (always F32-resident; the kernel reads it directly) ───
            var embedWeights = AllocAndLoad(source, "model.embed_tokens.weight", (long)vocab * d);

            // ─── Layers ───────────────────────────────────────────────────────
            var buffers = new LayerWeightBuffers[layers];
            for (var l = 0; l < layers; l++)
            {
                var p = $"model.layers.{l}";

                var attnNormGamma = AllocAndLoad(source, $"{p}.input_layernorm.weight", d);

                // Q and K are RoPE-rotated, so their rows are permuted from HF's
                // rotate-half layout into the adjacent-pair layout RopeKernel expects
                // (the same HF→GGUF permute llama.cpp applies). V and O are not rotated.
                var wq = LoadHeads(source, $"{p}.self_attn.q_proj.weight", nHeads, d, headDim, attnQuant, ropePermute: true);
                var wk = LoadHeads(source, $"{p}.self_attn.k_proj.weight", nKv, d, headDim, attnQuant, ropePermute: true);
                var wv = LoadHeads(source, $"{p}.self_attn.v_proj.weight", nKv, d, headDim, attnQuant, ropePermute: false);
                var wo = LoadOutputHeads(source, $"{p}.self_attn.o_proj.weight", nHeads, d, headDim, attnQuant);

                var bq = LoadBias(source, $"{p}.self_attn.q_proj.bias", nHeads, headDim, ropePermute: true);
                var bk = LoadBias(source, $"{p}.self_attn.k_proj.bias", nKv, headDim, ropePermute: true);
                var bv = LoadBias(source, $"{p}.self_attn.v_proj.bias", nKv, headDim, ropePermute: false);
                // o_proj has no per-head bias in the Llama/Qwen family — zeros per head.
                var bo = new TensorStorage<float>[nHeads];
                for (var h = 0; h < nHeads; h++)
                {
                    bo[h] = TensorStorage<float>.Unpooled(d);
                }

                var ffnNormGamma = AllocAndLoad(source, $"{p}.post_attention_layernorm.weight", d);

                var ffnGate = LoadProjection(source, $"{p}.mlp.gate_proj.weight", inDim: d, outDim: dff, ffnQuant);
                var ffnUp = LoadProjection(source, $"{p}.mlp.up_proj.weight", inDim: d, outDim: dff, ffnQuant);
                var ffnDown = LoadProjection(source, $"{p}.mlp.down_proj.weight", inDim: dff, outDim: d, ffnQuant);

                buffers[l] = new LayerWeightBuffers
                {
                    AttnNormGamma = attnNormGamma,
                    AttnNormBeta = TensorStorage<float>.Unpooled(0),
                    Wq = wq,
                    Bq = bq,
                    Wk = wk,
                    Bk = bk,
                    Wv = wv,
                    Bv = bv,
                    Wo = wo,
                    Bo = bo,
                    FfnNormGamma = ffnNormGamma,
                    FfnNormBeta = TensorStorage<float>.Unpooled(0),
                    FfnGate = ffnGate,
                    FfnUp = ffnUp,
                    FfnDown = ffnDown,
                };
            }

            // ─── Final norm + LM head ────────────────────────────────────────
            var finalNormGamma = AllocAndLoad(source, "model.norm.weight", d);
            var lmHead = LoadLmHead(source, config, embedWeights, lmQuant);

            return CachedLlamaInferenceEngine.CreateFromBuffers(
                config, embedWeights, finalNormGamma, TensorStorage<float>.Unpooled(0), lmHead, buffers);
        }

        // ─── Helpers ──────────────────────────────────────────────────────────

        private static TensorStorage<float> AllocAndLoad(ISafetensorsSource source, string name, long count)
        {
            var actual = source.ElementCount(name);
            if (actual != count)
            {
                throw new OverfitFormatException($"Tensor '{name}' has {actual} elements, expected {count}.");
            }
            var storage = TensorStorage<float>.Unpooled(checked((int)count));
            source.LoadF32(name, storage.AsSpan());
            return storage;
        }

        /// <summary>
        /// Q/K/V projection: file <c>[headCount*headDim, dModel]</c> → per-head
        /// <c>[dModel, headDim]</c>. Head h owns file rows <c>[h*headDim, (h+1)*headDim)</c>,
        /// each row being that output's contiguous dModel contraction vector.
        /// Q8 keeps that output-major layout verbatim; F32 transposes to input-major.
        /// </summary>
        private static DecodeWeight[] LoadHeads(
            ISafetensorsSource source, string name, int headCount, int dModel, int headDim, bool quantize, bool ropePermute)
        {
            var elems = checked((int)((long)headCount * headDim * dModel));
            ExpectCount(source, name, elems);
            using var scratch = new PooledBuffer<float>(elems, clearMemory: false);
            // RoPE-rotated weights (Q/K) need their headDim rows reordered from HF's
            // rotate-half layout to the adjacent-pair layout RopeKernel uses; the
            // permuted rows are staged here, one head at a time.
            using var permuted = ropePermute ? new PooledBuffer<float>(headDim * dModel, clearMemory: false) : default;
            source.LoadF32(name, scratch.Span);
            var heads = new DecodeWeight[headCount];
            for (var h = 0; h < headCount; h++)
            {
                var rowMajor = scratch.Span.Slice(h * headDim * dModel, headDim * dModel);
                if (ropePermute)
                {
                    PermuteRopeRows(rowMajor, permuted.Span.Slice(0, headDim * dModel), headDim, dModel);
                    rowMajor = permuted.Span.Slice(0, headDim * dModel);
                }

                if (quantize)
                {
                    heads[h] = Q8Weight.QuantizeRows(rowMajor, headDim, dModel);
                }
                else
                {
                    var storage = TensorStorage<float>.Unpooled(dModel * headDim);
                    var dst = storage.AsSpan();
                    for (var i = 0; i < dModel; i++)
                    {
                        for (var j = 0; j < headDim; j++)
                        {
                            dst[i * headDim + j] = rowMajor[j * dModel + i];
                        }
                    }
                    heads[h] = storage;
                }
            }
            return heads;
        }

        // HF→adjacent-pair row permute for a per-head [headDim, width] block:
        // out row 2i ← in row i, out row 2i+1 ← in row i+headDim/2. This is the
        // permute llama.cpp applies converting HF rotate-half weights to GGUF's
        // adjacent-pair (NEOX) layout, which is what RopeKernel expects.
        private static void PermuteRopeRows(ReadOnlySpan<float> src, Span<float> dst, int headDim, int width)
        {
            var half = headDim / 2;
            for (var i = 0; i < half; i++)
            {
                src.Slice(i * width, width).CopyTo(dst.Slice((2 * i) * width, width));
                src.Slice((i + half) * width, width).CopyTo(dst.Slice((2 * i + 1) * width, width));
            }
        }

        /// <summary>
        /// Output projection: file <c>[dModel, nHeads*headDim]</c> → per-head
        /// <c>[headDim, dModel]</c>. Head h owns a headDim-wide column band; gather it
        /// (strided across the dModel rows) into output-major order for Q8, or directly
        /// transpose for F32.
        /// </summary>
        private static DecodeWeight[] LoadOutputHeads(
            ISafetensorsSource source, string name, int nHeads, int dModel, int headDim, bool quantize)
        {
            var nHeadsHeadDim = nHeads * headDim;
            var elems = checked((int)((long)dModel * nHeadsHeadDim));
            ExpectCount(source, name, elems);
            using var scratch = new PooledBuffer<float>(elems, clearMemory: false);
            source.LoadF32(name, scratch.Span);
            var heads = new DecodeWeight[nHeads];
            for (var h = 0; h < nHeads; h++)
            {
                if (quantize)
                {
                    var gatherElems = dModel * headDim;
                    using var gather = new PooledBuffer<float>(gatherElems, clearMemory: false);
                    for (var o = 0; o < dModel; o++)
                    {
                        for (var i = 0; i < headDim; i++)
                        {
                            gather.Span[o * headDim + i] = scratch.Span[o * nHeadsHeadDim + h * headDim + i];
                        }
                    }
                    heads[h] = Q8Weight.QuantizeRows(gather.Span, dModel, headDim);
                }
                else
                {
                    var storage = TensorStorage<float>.Unpooled(headDim * dModel);
                    var dst = storage.AsSpan();
                    for (var i = 0; i < headDim; i++)
                    {
                        for (var j = 0; j < dModel; j++)
                        {
                            dst[i * dModel + j] = scratch.Span[j * nHeadsHeadDim + h * headDim + i];
                        }
                    }
                    heads[h] = storage;
                }
            }
            return heads;
        }

        /// <summary>
        /// FFN projection: file <c>[outDim, inDim]</c>. Q8 keeps the output-major
        /// layout (rows = outputs); F32 transposes to the kernel's input-major
        /// <c>[inDim, outDim]</c>.
        /// </summary>
        private static DecodeWeight LoadProjection(
            ISafetensorsSource source, string name, int inDim, int outDim, bool quantize)
        {
            var elems = checked((int)((long)inDim * outDim));
            ExpectCount(source, name, elems);
            using var scratch = new PooledBuffer<float>(elems, clearMemory: false);
            source.LoadF32(name, scratch.Span);
            if (quantize)
            {
                return Q8Weight.QuantizeRows(scratch.Span, outDim, inDim);
            }
            var storage = TensorStorage<float>.Unpooled(elems);
            var dst = storage.AsSpan();
            for (var i = 0; i < inDim; i++)
            {
                for (var o = 0; o < outDim; o++)
                {
                    dst[i * outDim + o] = scratch.Span[o * inDim + i];
                }
            }
            return storage;
        }

        /// <summary>
        /// LM head: separate <c>lm_head.weight [vocab, dModel]</c>, or the tied
        /// <c>embed_tokens</c> when <c>tie_word_embeddings</c>. Q8 keeps the file's
        /// output-major rows (row = token); F32 transposes to <c>[dModel, vocab]</c>.
        /// </summary>
        private static DecodeWeight LoadLmHead(
            ISafetensorsSource source, GPT1Config cfg, TensorStorage<float> embed, bool quantize)
        {
            int vocab = cfg.VocabSize, d = cfg.DModel;
            var tied = !source.Tensors.ContainsKey("lm_head.weight");

            if (tied)
            {
                var emb = embed.AsReadOnlySpan();
                return quantize
                    ? Q8Weight.QuantizeRows(emb, vocab, d)
                    : TransposeToInputMajor(emb, vocab, d);
            }

            var elems = checked((int)((long)vocab * d));
            ExpectCount(source, "lm_head.weight", elems);
            using var scratch = new PooledBuffer<float>(elems, clearMemory: false);
            source.LoadF32("lm_head.weight", scratch.Span);
            return quantize
                ? Q8Weight.QuantizeRows(scratch.Span, vocab, d)
                : TransposeToInputMajor(scratch.Span, vocab, d);
        }

        // [vocab, dModel] row-major → [dModel, vocab] (the kernel's input-major LM head).
        private static TensorStorage<float> TransposeToInputMajor(ReadOnlySpan<float> rowMajor, int vocab, int d)
        {
            var storage = TensorStorage<float>.Unpooled(checked((int)((long)vocab * d)));
            var dst = storage.AsSpan();
            for (var t = 0; t < vocab; t++)
            {
                var row = t * d;
                for (var k = 0; k < d; k++)
                {
                    dst[k * vocab + t] = rowMajor[row + k];
                }
            }
            return storage;
        }

        // Per-head bias [headCount*headDim] → headCount × [headDim]; zeros when absent.
        // RoPE-rotated biases (Q/K) get the same adjacent-pair permute as the weights.
        private static TensorStorage<float>[] LoadBias(
            ISafetensorsSource source, string name, int headCount, int headDim, bool ropePermute)
        {
            var bias = new TensorStorage<float>[headCount];
            if (!source.Tensors.ContainsKey(name))
            {
                for (var h = 0; h < headCount; h++)
                {
                    bias[h] = TensorStorage<float>.Unpooled(headDim);
                }
                return bias;
            }

            var elems = headCount * headDim;
            ExpectCount(source, name, elems);
            using var scratch = new PooledBuffer<float>(elems, clearMemory: false);
            source.LoadF32(name, scratch.Span);
            for (var h = 0; h < headCount; h++)
            {
                bias[h] = TensorStorage<float>.Unpooled(headDim);
                var src = scratch.Span.Slice(h * headDim, headDim);
                var dst = bias[h].AsSpan();
                if (ropePermute)
                {
                    PermuteRopeRows(src, dst, headDim, width: 1);
                }
                else
                {
                    src.CopyTo(dst);
                }
            }
            return bias;
        }

        private static void ExpectCount(ISafetensorsSource source, string name, long expected)
        {
            var actual = source.ElementCount(name);
            if (actual != expected)
            {
                throw new OverfitFormatException($"Tensor '{name}' has {actual} elements, expected {expected}.");
            }
        }
    }
}
