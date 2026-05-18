// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics.Tensors;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors.Core;
using LayerWeightBuffers = DevOnBike.Overfit.LanguageModels.Runtime.CachedLlamaInferenceEngine.LayerWeightBuffers;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Loads a Llama/Qwen-family GGUF file directly into a CachedLlamaInferenceEngine
    /// without intermediate binary conversion or Python tooling.
    ///
    /// Supports F32 / F16 / BF16 (Etap A). Quantized formats (Q4_K_M etc.) throw.
    ///
    /// Architecture support:
    ///   - qwen2 (Qwen2.5-0.5B, 3B, 7B, 14B, 32B)
    ///   - llama (Llama-2, Llama-3.x)
    ///   - mistral (Mistral 7B)
    /// </summary>
    public static class GgufLlamaLoader
    {
        public static CachedLlamaInferenceEngine Load(string path, bool? fp16Resident = null)
        {
            using var reader = new GgufReader(path);
            return LoadFromReader(reader, fp16Resident);
        }

        internal static CachedLlamaInferenceEngine LoadFromReader(GgufReader reader, bool? fp16ResidentOverride = null)
        {
            var arch = reader.GetMeta("general.architecture", "qwen2");

            // Slot 2c: FP16-resident keeps F16 weights as Half. Measured slower
            // than F32 decode (decode is compute-bound, not bandwidth-bound — the
            // F16->F32 widen is pure added cost), so it is an explicit opt-in for
            // RAM-constrained hosts only. Default (null) keeps the faster F32 path.
            // Valid only when the source tensors are F16.
            var sourceIsF16 = reader.Tensors.TryGetValue("blk.0.attn_q.weight", out var probeTensor)
                              && probeTensor.Type == GgmlType.F16;
            var fp16Resident = (fp16ResidentOverride ?? false) && sourceIsF16;

            // ─── Config from metadata ─────────────────────────────────────
            var nLayers = reader.GetMeta($"{arch}.block_count", 24);
            var dModel = reader.GetMeta($"{arch}.embedding_length", 896);
            var nHeads = reader.GetMeta($"{arch}.attention.head_count", 14);
            var nKvHeads = reader.GetMeta($"{arch}.attention.head_count_kv", nHeads);
            var dFF = reader.GetMeta($"{arch}.feed_forward_length", 4864);
            var ctxLen = reader.GetMeta($"{arch}.context_length", 8192);
            var ropeTheta = reader.GetMeta($"{arch}.rope.freq_base", 10000.0f);

            // Cap context length for memory sanity (32k+ models work but consume RAM).
            if (ctxLen > 8192) { ctxLen = 8192; }

            // Vocab from tokenizer metadata if present, else from token_embd shape
            var vocab = reader.GetMeta($"{arch}.vocab_size", 0);
            if (vocab == 0 && reader.Tensors.TryGetValue("token_embd.weight", out var embInfo))
            {
                // GGUF dim order: [dModel, vocab]. Last dim is vocab.
                vocab = (int)embInfo.Dims[^1];
            }
            if (vocab == 0) { throw new InvalidDataException("Cannot determine vocab size."); }

            var headDim = dModel / nHeads;
            if (headDim * nHeads != dModel)
            {
                throw new InvalidDataException(
                    $"dModel ({dModel}) is not divisible by nHeads ({nHeads}).");
            }

            // Tied embeddings: if no separate output.weight, lm_head reuses token_embd.
            var tieWeights = !reader.Tensors.ContainsKey("output.weight");

            var config = new GPT1Config
            {
                NLayers = nLayers,
                DModel = dModel,
                NHeads = nHeads,
                NKvHeads = nKvHeads,
                VocabSize = vocab,
                ContextLength = ctxLen,
                DFF = dFF,
                UseRoPE = true,
                RoPETheta = ropeTheta,
                FfnActivation = FeedForwardActivation.SwiGLU,
                TieWeights = tieWeights,
            };

            // ─── Embeddings ───────────────────────────────────────────────
            // GGUF: token_embd.weight dims [dModel, vocab] → file emb[token,dim] at token*dModel+dim
            // C# kernel: same layout (embed.Slice(token*dModel, dModel)) — no transpose.
            var embedWeights = AllocAndLoad(reader, "token_embd.weight", (long)vocab * dModel);

            // ─── Layer weights ────────────────────────────────────────────
            var layers = new LayerWeightBuffers[nLayers];
            // Scratch for full attn_q/k/v/o weight matrices.
            // Pool them so they're returned for reuse rather than waiting for Gen2 GC.
            var qFullElems = checked((int)((long)dModel * nHeads * headDim));
            var kFullElems = checked((int)((long)dModel * nKvHeads * headDim));
            var oFullElems = checked((int)((long)nHeads * headDim * dModel));
            var qFull = ArrayPool<float>.Shared.Rent(qFullElems);
            var kFull = ArrayPool<float>.Shared.Rent(kFullElems);
            var vFull = ArrayPool<float>.Shared.Rent(kFullElems);
            var oFull = ArrayPool<float>.Shared.Rent(oFullElems);
            var qBiasFull = ArrayPool<float>.Shared.Rent(nHeads * headDim);
            var kBiasFull = ArrayPool<float>.Shared.Rent(nKvHeads * headDim);
            var vBiasFull = ArrayPool<float>.Shared.Rent(nKvHeads * headDim);
            try
            {
                for (var l = 0; l < nLayers; l++)
                {
                    // Attention LayerNorm gamma (RMSNorm, no beta)
                    var attnNormGamma = AllocAndLoad(reader, $"blk.{l}.attn_norm.weight", dModel);
                    var attnNormBeta = TensorStorage<float>.Unpooled(0);

                    // Q, K, V, O full matrices
                    LoadTensor(reader, $"blk.{l}.attn_q.weight", qFull.AsSpan(0, qFullElems));
                    LoadTensor(reader, $"blk.{l}.attn_k.weight", kFull.AsSpan(0, kFullElems));
                    LoadTensor(reader, $"blk.{l}.attn_v.weight", vFull.AsSpan(0, kFullElems));
                    LoadTensor(reader, $"blk.{l}.attn_output.weight", oFull.AsSpan(0, oFullElems));

                    // Biases (optional — Qwen has them, Llama doesn't)
                    LoadTensorOrZeros(reader, $"blk.{l}.attn_q.bias", qBiasFull.AsSpan(0, nHeads * headDim));
                    LoadTensorOrZeros(reader, $"blk.{l}.attn_k.bias", kBiasFull.AsSpan(0, nKvHeads * headDim));
                    LoadTensorOrZeros(reader, $"blk.{l}.attn_v.bias", vBiasFull.AsSpan(0, nKvHeads * headDim));

                    // Split into per-head storages (with transposition)
                    var wq = SplitQuery(qFull, nHeads, dModel, headDim);
                    var bq = SplitBias(qBiasFull, nHeads, headDim);
                    var wk = SplitKeyValue(kFull, nKvHeads, dModel, headDim);
                    var bk = SplitBias(kBiasFull, nKvHeads, headDim);
                    var wv = SplitKeyValue(vFull, nKvHeads, dModel, headDim);
                    var bv = SplitBias(vBiasFull, nKvHeads, headDim);
                    var wo = SplitOutput(oFull, nHeads, dModel, headDim);
                    var bo = new TensorStorage<float>[nHeads];
                    for (var h = 0; h < nHeads; h++) { bo[h] = TensorStorage<float>.Unpooled(dModel); }

                    // FFN
                    var ffnNormGamma = AllocAndLoad(reader, $"blk.{l}.ffn_norm.weight", dModel);
                    var ffnNormBeta = TensorStorage<float>.Unpooled(0);
                    var ffnGate = AllocAndLoadTransposed(reader, $"blk.{l}.ffn_gate.weight", dModel, dFF);
                    var ffnUp = AllocAndLoadTransposed(reader, $"blk.{l}.ffn_up.weight", dModel, dFF);
                    var ffnDown = AllocAndLoadTransposed(reader, $"blk.{l}.ffn_down.weight", dFF, dModel);

                    layers[l] = new LayerWeightBuffers
                    {
                        AttnNormGamma = attnNormGamma,
                        AttnNormBeta = attnNormBeta,
                        Wq = ToResident(wq, fp16Resident),
                        Bq = bq,
                        Wk = ToResident(wk, fp16Resident),
                        Bk = bk,
                        Wv = ToResident(wv, fp16Resident),
                        Bv = bv,
                        Wo = ToResident(wo, fp16Resident),
                        Bo = bo,
                        FfnNormGamma = ffnNormGamma,
                        FfnNormBeta = ffnNormBeta,
                        FfnGate = ToResident(ffnGate, fp16Resident),
                        FfnUp = ToResident(ffnUp, fp16Resident),
                        FfnDown = ToResident(ffnDown, fp16Resident),
                    };
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(qFull);
                ArrayPool<float>.Shared.Return(kFull);
                ArrayPool<float>.Shared.Return(vFull);
                ArrayPool<float>.Shared.Return(oFull);
                ArrayPool<float>.Shared.Return(qBiasFull);
                ArrayPool<float>.Shared.Return(kBiasFull);
                ArrayPool<float>.Shared.Return(vBiasFull);
            }

            // ─── Final norm + LM head ─────────────────────────────────────
            var finalNormGamma = AllocAndLoad(reader, "output_norm.weight", dModel);
            var finalNormBeta = TensorStorage<float>.Unpooled(0);

            // LM head: tied embeddings use token_embd transposed; untied uses output.weight transposed
            // Kernel expects [dModel, vocab] (input-major), but file stores [vocab, dModel].
            var lmHead = TensorStorage<float>.Unpooled(checked((int)((long)vocab * dModel)));
            var lmHeadSpan = lmHead.AsSpan();
            var embSpan = embedWeights.AsSpan();

            if (tieWeights)
            {
                // Use token_embd, transpose to [dModel, vocab]
                for (var d = 0; d < dModel; d++)
                {
                    for (var t = 0; t < vocab; t++)
                    {
                        lmHeadSpan[d * vocab + t] = embSpan[t * dModel + d];
                    }
                }
            }
            else
            {
                // Read output.weight and transpose — use pool for the ~1.2 GB transient at 3B
                var outElems = checked((int)((long)vocab * dModel));
                var outputRaw = ArrayPool<float>.Shared.Rent(outElems);
                try
                {
                    LoadTensor(reader, "output.weight", outputRaw.AsSpan(0, outElems));
                    for (var d = 0; d < dModel; d++)
                    {
                        for (var t = 0; t < vocab; t++)
                        {
                            lmHeadSpan[d * vocab + t] = outputRaw[t * dModel + d];
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(outputRaw);
                }
            }

            return CachedLlamaInferenceEngine.CreateFromBuffers(
                config, embedWeights, finalNormGamma, finalNormBeta,
                ToResident(lmHead, fp16Resident), layers);
        }

        // ─── Helpers ────────────────────────────────────────────────────────

        private static TensorStorage<float> AllocAndLoad(GgufReader reader, string name, long elementCount)
        {
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new InvalidDataException($"Required tensor '{name}' missing from GGUF.");
            }
            if (info.ElementCount != elementCount)
            {
                throw new InvalidDataException(
                    $"Tensor '{name}' has {info.ElementCount} elements, expected {elementCount}.");
            }
            var storage = TensorStorage<float>.Unpooled(checked((int)elementCount));
            reader.LoadTensorAsF32(info, storage.AsSpan());
            return storage;
        }

        /// <summary>
        /// Loads an FFN weight transposed: GGUF stores [in, out], kernel wants [in, out]
        /// in input-major order. Same memory layout actually — direct copy works.
        /// Naming kept for symmetry/clarity with attention weights which DO need per-head transpose.
        /// </summary>
        private static TensorStorage<float> AllocAndLoadTransposed(GgufReader reader, string name, int inDim, int outDim)
        {
            // For FFN: GGUF file stores W[out, in] at flat (out*inDim + in).
            // Kernel needs W[in, out] at flat (in*outDim + out).
            // Transposition required.
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new InvalidDataException($"Required tensor '{name}' missing from GGUF.");
            }

            var elementCount = checked((int)((long)inDim * outDim));
            var raw = ArrayPool<float>.Shared.Rent(elementCount);
            try
            {
                reader.LoadTensorAsF32(info, raw.AsSpan(0, elementCount));

                var storage = TensorStorage<float>.Unpooled(elementCount);
                var dst = storage.AsSpan();
                for (var i = 0; i < inDim; i++)
                {
                    for (var o = 0; o < outDim; o++)
                    {
                        dst[i * outDim + o] = raw[o * inDim + i];
                    }
                }
                return storage;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(raw);
            }
        }

        private static void LoadTensor(GgufReader reader, string name, Span<float> dst)
        {
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new InvalidDataException($"Required tensor '{name}' missing from GGUF.");
            }
            reader.LoadTensorAsF32(info, dst);
        }

        private static void LoadTensorOrZeros(GgufReader reader, string name, Span<float> dst)
        {
            if (reader.Tensors.TryGetValue(name, out var info))
            {
                reader.LoadTensorAsF32(info, dst);
            }
            else
            {
                dst.Clear();
            }
        }

        /// <summary>
        /// Split full Q weight [nHeads*headDim, dModel] into per-head [dModel, headDim].
        /// GGUF file: Q[out, in] at flat (out*dModel + in) where out in [0, nHeads*headDim), in in [0, dModel).
        /// Per-head kernel layout: Wq_h[i, j] at flat (i*headDim + j) where i in [0, dModel), j in [0, headDim).
        /// Mapping: Wq_h[i, j] = Q[h*headDim + j, i] = file[(h*headDim + j)*dModel + i].
        /// </summary>
        private static TensorStorage<float>[] SplitQuery(float[] qFull, int nHeads, int dModel, int headDim)
        {
            var wq = new TensorStorage<float>[nHeads];
            for (var h = 0; h < nHeads; h++)
            {
                wq[h] = TensorStorage<float>.Unpooled(checked((int)((long)dModel * headDim)));
                var dst = wq[h].AsSpan();
                for (var i = 0; i < dModel; i++)
                {
                    for (var j = 0; j < headDim; j++)
                    {
                        dst[i * headDim + j] = qFull[(h * headDim + j) * dModel + i];
                    }
                }
            }
            return wq;
        }

        /// <summary>Same transposition as SplitQuery but iterates over nKvHeads.</summary>
        private static TensorStorage<float>[] SplitKeyValue(float[] kvFull, int nKvHeads, int dModel, int headDim)
        {
            var wkv = new TensorStorage<float>[nKvHeads];
            for (var kv = 0; kv < nKvHeads; kv++)
            {
                wkv[kv] = TensorStorage<float>.Unpooled(checked((int)((long)dModel * headDim)));
                var dst = wkv[kv].AsSpan();
                for (var i = 0; i < dModel; i++)
                {
                    for (var j = 0; j < headDim; j++)
                    {
                        dst[i * headDim + j] = kvFull[(kv * headDim + j) * dModel + i];
                    }
                }
            }
            return wkv;
        }

        /// <summary>
        /// Split full O weight [dModel, nHeads*headDim] into per-head [headDim, dModel].
        /// GGUF: O[out, in] at flat (out*nHeadsHeadDim + in) where out in [0, dModel), in in [0, nHeads*headDim).
        /// Per-head kernel: Wo_h[i, j] at flat (i*dModel + j) where i in [0, headDim), j in [0, dModel).
        /// Mapping: Wo_h[i, j] = O[j, h*headDim + i] = file[j*nHeadsHeadDim + h*headDim + i].
        /// </summary>
        private static TensorStorage<float>[] SplitOutput(float[] oFull, int nHeads, int dModel, int headDim)
        {
            var wo = new TensorStorage<float>[nHeads];
            var nHeadsHeadDim = nHeads * headDim;
            for (var h = 0; h < nHeads; h++)
            {
                wo[h] = TensorStorage<float>.Unpooled(checked((int)((long)headDim * dModel)));
                var dst = wo[h].AsSpan();
                for (var i = 0; i < headDim; i++)
                {
                    for (var j = 0; j < dModel; j++)
                    {
                        dst[i * dModel + j] = oFull[j * nHeadsHeadDim + h * headDim + i];
                    }
                }
            }
            return wo;
        }

        private static TensorStorage<float>[] SplitBias(float[] biasFull, int nHeads, int headDim)
        {
            var bias = new TensorStorage<float>[nHeads];
            for (var h = 0; h < nHeads; h++)
            {
                bias[h] = TensorStorage<float>.Unpooled(headDim);
                var dst = bias[h].AsSpan();
                for (var j = 0; j < headDim; j++)
                {
                    dst[j] = biasFull[h * headDim + j];
                }
            }
            return bias;
        }

        /// <summary>
        /// Wraps a freshly-loaded F32 weight as a <see cref="MatrixWeight"/>. When
        /// <paramref name="fp16"/>, narrows it to FP16-resident storage (Slot 2c)
        /// and disposes the F32 buffer. F16 → F32 → F16 round-trips exactly for an
        /// F16-source GGUF, so this is lossless; the transient load peak is ~1.5×
        /// one tensor (the F32 buffer is freed immediately after each conversion).
        /// </summary>
        private static MatrixWeight ToResident(TensorStorage<float> f32, bool fp16)
        {
            if (!fp16)
            {
                return f32;
            }

            var half = TensorStorage<Half>.Unpooled(f32.Length);
            TensorPrimitives.ConvertToHalf(f32.AsReadOnlySpan(), half.AsSpan());
            f32.Dispose();
            return half;
        }

        private static MatrixWeight[] ToResident(TensorStorage<float>[] f32s, bool fp16)
        {
            var result = new MatrixWeight[f32s.Length];
            for (var i = 0; i < f32s.Length; i++)
            {
                result[i] = ToResident(f32s[i], fp16);
            }
            return result;
        }
    }
}
