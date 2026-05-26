// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Runtime;
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
    ///   - qwen2moe (Qwen1.5-MoE — routed experts + sigmoid-gated shared expert)
    ///   - Mixtral (llama-arch MoE — routed experts only, no shared expert)
    /// </summary>
    public static class GgufLlamaLoader
    {
        /// <param name="path">Path to the GGUF model file.</param>
        /// <param name="quantize">
        /// When true (default) FFN / LM-head / attention weights become Q8_0-resident
        /// where dimensions allow. When false every weight loads as F32 — the
        /// pre-quantization decode path, used as the parity reference (step 2.5).
        /// </param>
        /// <param name="mmap">
        /// When true (default) the verbatim-layout K-quant weights (Q4_K / Q6_K — FFN, LM head,
        /// and per-head attention Q/K/V) are backed by zero-copy slices of a read-only memory map
        /// of the file instead of managed <c>byte[]</c> copies. The map's lifetime is handed
        /// to the returned engine, which disposes it last. Cuts committed RAM for those weights
        /// to zero (the OS pages them in as shared/clean working set, reclaimable without swap).
        /// Has no effect on Q8_0 (de-interleaved) or F32-fallback weights, which still copy — so
        /// when the file contains no Q4_K/Q6_K tensors (pure-F32 / pure-Q8_0) the map is skipped
        /// entirely and load behaves exactly as the copy path (no file handle kept open).
        /// </param>
        public static CachedLlamaInferenceEngine Load(string path, bool quantize = true, bool mmap = true)
        {
            using var reader = new GgufReader(path);

            // Skip the map unless it would actually back a weight: it only helps verbatim
            // Q4_K/Q6_K weights, and only when quantization is on. Otherwise the copy path is
            // identical and avoids holding the file handle open for the engine's lifetime.
            if (!mmap || !quantize || !FileHasVerbatimKQuant(reader))
            {
                return LoadFromReader(reader, quantize);
            }

            // The map must outlive the reader (its FileStream closes when `using` exits) —
            // weights hold slices into the map, not the reader. On any failure during build
            // we own the map and must dispose it; on success ownership transfers to the engine.
            var blob = new MemoryMappedModelFile(path);

            try
            {
                return LoadFromReader(reader, quantize, blob);
            }
            catch
            {
                blob.Dispose();
                throw;
            }
        }

        /// <summary>
        /// True when the file holds at least one Q4_K or Q6_K tensor — the only formats the
        /// mmap path slices verbatim. Pure-F32 / pure-Q8_0 files gain nothing from the map.
        /// </summary>
        private static bool FileHasVerbatimKQuant(GgufReader reader)
        {
            foreach (var kv in reader.Tensors)
            {
                var t = kv.Value.Type;
                if (t == GgmlType.Q4_K || t == GgmlType.Q6_K)
                {
                    return true;
                }
            }
            return false;
        }

        internal static CachedLlamaInferenceEngine LoadFromReader(
            GgufReader reader, bool quantize = true, MemoryMappedModelFile? mmap = null)
        {
            var arch = reader.GetMeta("general.architecture", "qwen2");

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

            // Mixture of Experts: the FFN is replaced by a router + routed experts. Qwen-MoE adds a
            // sigmoid-gated shared expert (`ffn_*_shexp`) that runs every token; Mixtral has no shared
            // expert (routed-only) — detected by the presence of the shared tensors. The routed-expert
            // FFN length can differ from the dense `feed_forward_length`, so it's read from the 3-D
            // expert tensor shape.
            var expertCount = reader.GetMeta($"{arch}.expert_count", 0);
            var expertUsedCount = reader.GetMeta($"{arch}.expert_used_count", 0);
            var isMoe = expertCount > 0 && expertUsedCount > 0;
            // Two GGUF expert layouts: merged 3-D `ffn_gate_exps.weight` [in, out, n_expert] (Qwen-MoE
            // and newer Mixtral conversions) vs. one 2-D tensor per expert `ffn_gate.{e}.weight`
            // (older Mixtral). expertDff = the experts' FFN length (Dims[1] either way).
            var mergedExperts = isMoe && reader.Tensors.ContainsKey("blk.0.ffn_gate_exps.weight");
            var expertDff = !isMoe ? 0
                : mergedExperts ? (int)reader.Tensors["blk.0.ffn_gate_exps.weight"].Dims[1]
                : (int)reader.Tensors["blk.0.ffn_gate.0.weight"].Dims[1];
            var hasSharedExpert = isMoe && reader.Tensors.ContainsKey("blk.0.ffn_gate_shexp.weight");

            // Top-k weight renormalisation default is arch-specific when the GGUF omits the key:
            // Qwen1.5-MoE uses norm_topk_prob=false (raw full-softmax probs); Mixtral and other
            // routed-only MoE renormalise the top-k to sum 1.
            var normalizeDefault = !arch.Contains("qwen2moe", StringComparison.Ordinal);

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
                ExpertCount = expertCount,
                ExpertUsedCount = expertUsedCount,
                ExpertFeedForwardLength = expertDff,
                HasSharedExpert = hasSharedExpert,
                // llama.cpp exposes top-k renormalisation as {arch}.expert_weights_norm; when absent
                // the default is arch-specific (false for qwen2moe, true for Mixtral/routed-only).
                NormalizeExpertWeights = reader.GetMeta($"{arch}.expert_weights_norm", normalizeDefault),
            };

            // ─── Embeddings ───────────────────────────────────────────────
            // GGUF: token_embd.weight dims [dModel, vocab] → file emb[token,dim] at token*dModel+dim
            // C# kernel: same layout (row = token) — no transpose. Kept in native K-quant layout
            // (verbatim / mmap-able) when the file stores it as Q4_K/Q6_K and quantization is on;
            // only the looked-up row is dequantized per token. F32 fallback otherwise (the old path).
            var embedWeights = LoadEmbedding(reader, dModel, vocab, quantize, mmap);

            // ─── Layer weights ────────────────────────────────────────────
            var layers = new LayerWeightBuffers[nLayers];

            // Per-tensor dispatch (step 3.2b) — a Q4_K_M file is heterogeneous
            // (typically Q/K/O = Q4_K, V = Q6_K), so attn_q/k/v each pick
            // Q4_K-native / Q8_0-native / F32-fallback independently from the
            // file's tensor type. Wo can't be K-quant per-head (headDim < the
            // 256-element K-quant super-block) but CAN be Q8_0 per-head — its
            // dispatcher (`LoadOutputHeads`) handles that.
            var qFullElems = checked((int)((long)dModel * nHeads * headDim));
            var kFullElems = checked((int)((long)dModel * nKvHeads * headDim));
            var oFullElems = checked((int)((long)nHeads * headDim * dModel));

            var attnQuantizable = quantize
                && dModel % Q8DotKernel.BlockSize == 0 && headDim % Q8DotKernel.BlockSize == 0;

            // F32 scratch is needed for any tensor that isn't K-quant-native on
            // disk (the F32-fallback path). Peek layer 0's types — quant files
            // are uniform across layers — to decide per tensor.
            var qNeedsF32 = !attnQuantizable
                || !IsKQuantNative(reader.Tensors["blk.0.attn_q.weight"], dModel);
            var kNeedsF32 = !attnQuantizable
                || !IsKQuantNative(reader.Tensors["blk.0.attn_k.weight"], dModel);
            var vNeedsF32 = !attnQuantizable
                || !IsKQuantNative(reader.Tensors["blk.0.attn_v.weight"], dModel);
            // Wo: K-quant per-head is blocked by headDim < 256, but Q8_0 per-head works.
            var oNeedsF32 = !attnQuantizable
                || reader.Tensors["blk.0.attn_output.weight"].Type != GgmlType.Q8_0;

            float[] qFull = qNeedsF32 ? ArrayPool<float>.Shared.Rent(qFullElems) : [];
            float[] kFull = kNeedsF32 ? ArrayPool<float>.Shared.Rent(kFullElems) : [];
            float[] vFull = vNeedsF32 ? ArrayPool<float>.Shared.Rent(kFullElems) : [];
            float[] oFull = oNeedsF32 ? ArrayPool<float>.Shared.Rent(oFullElems) : [];
            // Attention biases are always F32 in GGUF — bias scratch always needed.
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

                    // Per-tensor dispatch (step 3.2b) — each of attn_q/k/v picks
                    // Q4_K-native / Q8_0-native / F32-fallback independently from
                    // its file format. Wo dispatches separately (Q8_0 OK per-head;
                    // K-quant not — headDim < the 256-element super-block).
                    var wq = LoadQkvHeads(reader, $"blk.{l}.attn_q.weight", nHeads, dModel, headDim, qFull, attnQuantizable, mmap);
                    var wk = LoadQkvHeads(reader, $"blk.{l}.attn_k.weight", nKvHeads, dModel, headDim, kFull, attnQuantizable, mmap);
                    var wv = LoadQkvHeads(reader, $"blk.{l}.attn_v.weight", nKvHeads, dModel, headDim, vFull, attnQuantizable, mmap);
                    var wo = LoadOutputHeads(reader, $"blk.{l}.attn_output.weight", nHeads, dModel, headDim, oFull, attnQuantizable);

                    // Attention biases (optional — Qwen has them, Llama doesn't);
                    // always F32 in GGUF, never quantized.
                    LoadTensorOrZeros(reader, $"blk.{l}.attn_q.bias", qBiasFull.AsSpan(0, nHeads * headDim));
                    LoadTensorOrZeros(reader, $"blk.{l}.attn_k.bias", kBiasFull.AsSpan(0, nKvHeads * headDim));
                    LoadTensorOrZeros(reader, $"blk.{l}.attn_v.bias", vBiasFull.AsSpan(0, nKvHeads * headDim));
                    var bq = SplitBias(qBiasFull, nHeads, headDim);
                    var bk = SplitBias(kBiasFull, nKvHeads, headDim);
                    var bv = SplitBias(vBiasFull, nKvHeads, headDim);
                    var bo = new TensorStorage<float>[nHeads];
                    for (var h = 0; h < nHeads; h++) { bo[h] = TensorStorage<float>.Unpooled(dModel); }

                    // FFN
                    var ffnNormGamma = AllocAndLoad(reader, $"blk.{l}.ffn_norm.weight", dModel);
                    var ffnNormBeta = TensorStorage<float>.Unpooled(0);

                    DecodeWeight ffnGate = default, ffnUp = default, ffnDown = default;
                    float[]? moeRouter = null, moeSharedGateInp = null;
                    DecodeWeight[]? moeGate = null, moeUp = null, moeDown = null;
                    DecodeWeight moeShGate = default, moeShUp = default, moeShDown = default;

                    if (isMoe)
                    {
                        // Router + routed experts (3-D tensors). Qwen-MoE additionally has a
                        // sigmoid-gated shared expert; Mixtral does not (hasSharedExpert == false).
                        moeRouter = LoadRouter(reader, reader.Tensors[$"blk.{l}.ffn_gate_inp.weight"], dModel, expertCount);
                        if (mergedExperts)
                        {
                            moeGate = LoadExperts(reader, reader.Tensors[$"blk.{l}.ffn_gate_exps.weight"]);
                            moeUp = LoadExperts(reader, reader.Tensors[$"blk.{l}.ffn_up_exps.weight"]);
                            moeDown = LoadExperts(reader, reader.Tensors[$"blk.{l}.ffn_down_exps.weight"]);
                        }
                        else
                        {
                            // Older Mixtral: one 2-D weight per expert, loaded with the same resident
                            // dispatch as a dense FFN (Q4_K verbatim/mmap, Q5/Q6/Q8/F32).
                            moeGate = LoadExpertsSplit(reader, l, "ffn_gate", dModel, expertDff, expertCount, mmap);
                            moeUp = LoadExpertsSplit(reader, l, "ffn_up", dModel, expertDff, expertCount, mmap);
                            moeDown = LoadExpertsSplit(reader, l, "ffn_down", expertDff, dModel, expertCount, mmap);
                        }
                        if (hasSharedExpert)
                        {
                            moeShGate = AllocAndLoadResident(reader, $"blk.{l}.ffn_gate_shexp.weight", dModel, dFF, mmap);
                            moeShUp = AllocAndLoadResident(reader, $"blk.{l}.ffn_up_shexp.weight", dModel, dFF, mmap);
                            moeShDown = AllocAndLoadResident(reader, $"blk.{l}.ffn_down_shexp.weight", dFF, dModel, mmap);
                            moeSharedGateInp = LoadF32Vector(reader, $"blk.{l}.ffn_gate_inp_shexp.weight", dModel);
                        }
                    }
                    else if (quantize && dModel % Q8DotKernel.BlockSize == 0 && dFF % Q8DotKernel.BlockSize == 0)
                    {
                        ffnGate = AllocAndLoadResident(reader, $"blk.{l}.ffn_gate.weight", dModel, dFF, mmap);
                        ffnUp = AllocAndLoadResident(reader, $"blk.{l}.ffn_up.weight", dModel, dFF, mmap);
                        ffnDown = AllocAndLoadResident(reader, $"blk.{l}.ffn_down.weight", dFF, dModel, mmap);
                    }
                    else
                    {
                        ffnGate = AllocAndLoadTransposed(reader, $"blk.{l}.ffn_gate.weight", dModel, dFF);
                        ffnUp = AllocAndLoadTransposed(reader, $"blk.{l}.ffn_up.weight", dModel, dFF);
                        ffnDown = AllocAndLoadTransposed(reader, $"blk.{l}.ffn_down.weight", dFF, dModel);
                    }

                    layers[l] = new LayerWeightBuffers
                    {
                        AttnNormGamma = attnNormGamma,
                        AttnNormBeta = attnNormBeta,
                        Wq = wq,
                        Bq = bq,
                        Wk = wk,
                        Bk = bk,
                        Wv = wv,
                        Bv = bv,
                        Wo = wo,
                        Bo = bo,
                        FfnNormGamma = ffnNormGamma,
                        FfnNormBeta = ffnNormBeta,
                        FfnGate = ffnGate,
                        FfnUp = ffnUp,
                        FfnDown = ffnDown,
                        MoeRouter = moeRouter,
                        MoeGate = moeGate,
                        MoeUp = moeUp,
                        MoeDown = moeDown,
                        MoeSharedGate = moeShGate,
                        MoeSharedUp = moeShUp,
                        MoeSharedDown = moeShDown,
                        MoeSharedGateInp = moeSharedGateInp,
                    };
                }
            }
            finally
            {
                // qFull..oFull are empty sentinels on the native Q8_0 path.
                if (qFull.Length > 0) { ArrayPool<float>.Shared.Return(qFull); }
                if (kFull.Length > 0) { ArrayPool<float>.Shared.Return(kFull); }
                if (vFull.Length > 0) { ArrayPool<float>.Shared.Return(vFull); }
                if (oFull.Length > 0) { ArrayPool<float>.Shared.Return(oFull); }
                ArrayPool<float>.Shared.Return(qBiasFull);
                ArrayPool<float>.Shared.Return(kBiasFull);
                ArrayPool<float>.Shared.Return(vBiasFull);
            }

            // ─── Final norm + LM head ─────────────────────────────────────
            var finalNormGamma = AllocAndLoad(reader, "output_norm.weight", dModel);
            var finalNormBeta = TensorStorage<float>.Unpooled(0);

            // LM head — step 2.3a: resident as Q8_0 (output-major). The file
            // stores it [vocab, dModel] — row = token = one output's contraction
            // vector — which is exactly Q8Weight's layout, so no transpose.
            // tied → token_embd; untied → output.weight. When quantization is
            // disabled, or dModel is not a multiple of the Q8 block size, fall
            // back to an F32 transposed LM head (the kernel's input-major layout).
            DecodeWeight lmHead;
            if (quantize && dModel % Q8DotKernel.BlockSize == 0)
            {
                var lmHeadInfo = reader.Tensors[tieWeights ? "token_embd.weight" : "output.weight"];
                if (lmHeadInfo.Type == GgmlType.Q4_K && dModel % Q4KWeight.SuperBlockElements == 0)
                {
                    // Native Q4_K — read the file's blocks straight in (step 3.2b).
                    lmHead = LoadQ4KNative(reader, lmHeadInfo, dModel, vocab, mmap);
                }
                else if (lmHeadInfo.Type == GgmlType.Q6_K && dModel % Q6KWeight.SuperBlockElements == 0)
                {
                    // Native Q6_K — read the file's blocks straight in (step 3.3c).
                    lmHead = LoadQ6KNative(reader, lmHeadInfo, dModel, vocab, mmap);
                }
                else if (lmHeadInfo.Type == GgmlType.Q8_0)
                {
                    // Native Q8_0 — read the file's blocks straight in (step 2.4).
                    lmHead = LoadQ8Native(reader, lmHeadInfo, dModel, vocab);
                }
                else if (tieWeights)
                {
                    // Reached only when token_embd is F16/F32/BF16 (the K-quant/Q8 cases are
                    // caught above), so the embedding is F32-backed here — .F32 is valid.
                    lmHead = Q8Weight.QuantizeRows(embedWeights.F32, vocab, dModel);
                }
                else
                {
                    var outElems = checked((int)((long)vocab * dModel));
                    using var outputRaw = new PooledArray<float>(outElems);
                    LoadTensor(reader, "output.weight", outputRaw.Span);
                    lmHead = Q8Weight.QuantizeRows(outputRaw.Span, vocab, dModel);
                }
            }
            else
            {
                // F32 fallback — transpose [vocab, dModel] → [dModel, vocab].
                var f32LmHead = TensorStorage<float>.Unpooled(checked((int)((long)vocab * dModel)));
                var lmHeadSpan = f32LmHead.AsSpan();
                if (tieWeights)
                {
                    // F32 fallback (quantize off / dModel not Q8-aligned) ⇒ embedding is F32-backed.
                    var embSpan = embedWeights.F32;
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
                    var outElems = checked((int)((long)vocab * dModel));
                    using var outputRaw = new PooledArray<float>(outElems);
                    LoadTensor(reader, "output.weight", outputRaw.Span);
                    var src = outputRaw.Span;
                    for (var d = 0; d < dModel; d++)
                    {
                        for (var t = 0; t < vocab; t++)
                        {
                            lmHeadSpan[d * vocab + t] = src[t * dModel + d];
                        }
                    }
                }

                lmHead = f32LmHead;
            }

            return CachedLlamaInferenceEngine.CreateFromBuffers(
                config, embedWeights, finalNormGamma, finalNormBeta, lmHead, layers, mmap);
        }

        // ─── Helpers ────────────────────────────────────────────────────────

        /// <summary>
        /// Loads the token-embedding table. The file stores <c>token_embd.weight</c> row-major
        /// [vocab, dModel] (row = token) — exactly <see cref="Q4KWeight"/>/<see cref="Q6KWeight"/>'s
        /// output-major layout — so when it is K-quant on disk and quantization is on it is kept
        /// verbatim (mmap-able; zero managed bytes) and the lookup dequantizes only the row it needs.
        /// Otherwise (F16/F32/BF16 source, or quantization off) it loads dequantized to F32, as before.
        ///
        /// Cuts the embedding-table RAM on a Q4_K_M model from a full F32 copy (~1.2 GB for Qwen-3B,
        /// vocab 151936 × dModel 2048) to the verbatim Q6_K bytes (~255 MB, off the managed heap when
        /// mmap-backed) — the largest remaining post-mmap RAM lever.
        /// </summary>
        private static DecodeWeight LoadEmbedding(
            GgufReader reader, int dModel, int vocab, bool quantize, MemoryMappedModelFile? mmap)
        {
            if (!reader.Tensors.TryGetValue("token_embd.weight", out var info))
            {
                throw new InvalidDataException("Required tensor 'token_embd.weight' missing from GGUF.");
            }
            if (info.ElementCount != (long)vocab * dModel)
            {
                throw new InvalidDataException(
                    $"Tensor 'token_embd.weight' has {info.ElementCount} elements, expected {(long)vocab * dModel}.");
            }

            if (quantize && info.Type == GgmlType.Q4_K && dModel % Q4KWeight.SuperBlockElements == 0)
            {
                return LoadQ4KNative(reader, info, dModel, vocab, mmap);
            }
            if (quantize && info.Type == GgmlType.Q6_K && dModel % Q6KWeight.SuperBlockElements == 0)
            {
                return LoadQ6KNative(reader, info, dModel, vocab, mmap);
            }

            // F32 fallback — full dequant into a flat [vocab × dModel] row-major buffer.
            var storage = TensorStorage<float>.Unpooled(checked((int)((long)vocab * dModel)));
            reader.LoadTensorAsF32(info, storage.AsSpan());
            return storage;
        }

        internal static TensorStorage<float> AllocAndLoad(GgufReader reader, string name, long elementCount)
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

        // ── Mixture of Experts (qwen2moe) ─────────────────────────────────────

        /// <summary>
        /// Splits a 3-D GGUF expert tensor <c>[ne0, ne1, n_expert]</c> into one
        /// <see cref="DecodeWeight"/> per expert (input dim = <c>ne0</c>, output dim = <c>ne1</c>) —
        /// e.g. <c>ffn_gate_exps</c> / <c>ffn_up_exps</c> / <c>ffn_down_exps</c>. Each expert is a
        /// contiguous slice (the expert axis <c>ne2</c> is slowest), so Q4_K/Q6_K experts keep their
        /// bytes verbatim and Q8_0 experts de-interleave per expert. Managed copies for now (mmap
        /// per-expert slicing is a follow-on).
        /// </summary>
        internal static DecodeWeight[] LoadExperts(GgufReader reader, GgufTensorInfo info)
        {
            if (info.Dims.Length != 3)
            {
                throw new InvalidDataException(
                    $"Expert tensor '{info.Name}' must be 3-D [ne0, ne1, n_expert], got {info.Dims.Length} dims.");
            }

            var inDim = checked((int)info.Dims[0]);
            var outDim = checked((int)info.Dims[1]);
            var expertCount = checked((int)info.Dims[2]);
            var perExpert = (long)inDim * outDim;
            var experts = new DecodeWeight[expertCount];

            switch (info.Type)
            {
                case GgmlType.Q4_K:
                case GgmlType.Q6_K:
                    {
                        var superBytes = info.Type == GgmlType.Q4_K
                            ? Q4KWeight.SuperBlockBytes : Q6KWeight.SuperBlockBytes;
                        var expertBytes = checked((int)(perExpert / GgmlDequant.SuperBlockElements * superBytes));
                        var total = checked((int)((long)expertBytes * expertCount));
                        using var whole = new PooledArray<byte>(total);

                        if (info.Type == GgmlType.Q4_K) { reader.LoadTensorQ4_KRaw(info, whole.Span); }
                        else { reader.LoadTensorQ6_KRaw(info, whole.Span); }

                        for (var e = 0; e < expertCount; e++)
                        {
                            var bytes = new byte[expertBytes];
                            whole.Span.Slice(e * expertBytes, expertBytes).CopyTo(bytes);
                            experts[e] = info.Type == GgmlType.Q4_K
                                ? new Q4KWeight(bytes, inDim, outDim)
                                : new Q6KWeight(bytes, inDim, outDim);
                        }
                        break;
                    }

                case GgmlType.Q8_0:
                    {
                        var elems = checked((int)(perExpert * expertCount));
                        var blocksPerExpert = checked((int)(perExpert / Q8DotKernel.BlockSize));
                        using var quants = new PooledArray<sbyte>(elems);
                        using var scales = new PooledArray<float>(blocksPerExpert * expertCount);
                        reader.LoadTensorQ8_0Raw(info, quants.Span, scales.Span);

                        var perElems = checked((int)perExpert);
                        for (var e = 0; e < expertCount; e++)
                        {
                            var q = new sbyte[perElems];
                            var s = new float[blocksPerExpert];
                            quants.Span.Slice(e * perElems, perElems).CopyTo(q);
                            scales.Span.Slice(e * blocksPerExpert, blocksPerExpert).CopyTo(s);
                            experts[e] = new Q8Weight(q, s, inDim, outDim);
                        }
                        break;
                    }

                case GgmlType.Q5_0:
                case GgmlType.Q5_K:
                    {
                        // No native 5-bit dot kernel — and Q4_K_M "_M" mixes put Q5_0 on some
                        // ffn_down_exps. Dequant each expert to F32 then re-quantize to Q8 (near-lossless
                        // from a 5-bit source; reuses the Q8 dot kernel). Streamed one expert at a time so
                        // peak load RAM is a single expert's F32 (~perExpert floats), not the whole
                        // tensor's — steady-state stays Q8 (~1 B/elem), keeping the smaller file a RAM win.
                        var perElems = checked((int)perExpert);
                        using var f32 = new PooledArray<float>(perElems);
                        for (var e = 0; e < expertCount; e++)
                        {
                            reader.LoadQ5RegionAsF32(info, (long)e * perElems, f32.Span);
                            experts[e] = Q8Weight.QuantizeRows(f32.Span, outDim, inDim);
                        }
                        break;
                    }

                default:
                    throw new NotSupportedException(
                        $"Expert tensor '{info.Name}' type {info.Type} is not supported " +
                        $"(expected Q4_K / Q5_0 / Q5_K / Q6_K / Q8_0).");
            }

            return experts;
        }

        /// <summary>
        /// Loads the older Mixtral split-expert layout: one 2-D GGUF tensor per expert
        /// (<c>blk.{layer}.{namePart}.{e}.weight</c>, e.g. <c>ffn_gate.0.weight</c>) into one
        /// <see cref="DecodeWeight"/> per expert, via the same resident dispatch as a dense FFN weight
        /// (<see cref="AllocAndLoadResident"/>: Q4_K/Q6_K verbatim/mmap, Q5_0/Q5_K → Q8, Q8_0 native,
        /// F32 fallback). <paramref name="inDim"/>/<paramref name="outDim"/> are the per-expert
        /// projection dims (gate/up: dModel→expertDff; down: expertDff→dModel).
        /// </summary>
        internal static DecodeWeight[] LoadExpertsSplit(
            GgufReader reader, int layer, string namePart, int inDim, int outDim, int expertCount,
            MemoryMappedModelFile? mmap)
        {
            var experts = new DecodeWeight[expertCount];
            for (var e = 0; e < expertCount; e++)
            {
                experts[e] = AllocAndLoadResident(reader, $"blk.{layer}.{namePart}.{e}.weight", inDim, outDim, mmap);
            }
            return experts;
        }

        /// <summary>
        /// Loads the MoE router <c>ffn_gate_inp</c> (<c>[ne0=dModel, ne1=expertCount]</c>, F32) and
        /// transposes GGUF's output-major layout to the input-major <c>[dModel × expertCount]</c> the
        /// single-token projection kernel expects (mirrors the F32 FFN transpose).
        /// </summary>
        /// <summary>
        /// Dequantizes a 3-D expert tensor to F32, one <see cref="DecodeWeight"/> per expert, in the
        /// input-major <c>[ne0 × ne1]</c> layout the F32 projection kernel expects (the GGUF stores it
        /// output-major, so each expert is transposed). Used to build an F32 reference for decode
        /// parity — large (4× the quantized size), so it's a test/diagnostic helper, not a load path.
        /// </summary>
        internal static DecodeWeight[] LoadExpertsF32(GgufReader reader, GgufTensorInfo info)
        {
            if (info.Dims.Length != 3)
            {
                throw new InvalidDataException($"Expert tensor '{info.Name}' must be 3-D, got {info.Dims.Length} dims.");
            }

            var inDim = checked((int)info.Dims[0]);
            var outDim = checked((int)info.Dims[1]);
            var expertCount = checked((int)info.Dims[2]);
            var perExpert = checked(inDim * outDim);
            var total = checked(perExpert * expertCount);

            using var whole = new PooledArray<float>(total);
            reader.LoadTensorAsF32(info, whole.Span);   // expert-major, each [outDim, inDim]
            var src = whole.Span;

            var experts = new DecodeWeight[expertCount];
            for (var e = 0; e < expertCount; e++)
            {
                var storage = TensorStorage<float>.Unpooled(perExpert);
                var dst = storage.AsSpan();
                var srcBase = e * perExpert;
                for (var o = 0; o < outDim; o++)
                {
                    for (var i = 0; i < inDim; i++)
                    {
                        dst[i * outDim + o] = src[srcBase + o * inDim + i];   // [out,in] → [in,out]
                    }
                }
                experts[e] = storage;
            }
            return experts;
        }

        /// <summary>Loads a small F32 vector tensor (e.g. the shared-expert gate <c>ffn_gate_inp_shexp</c>).</summary>
        internal static float[] LoadF32Vector(GgufReader reader, string name, int count)
        {
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new InvalidDataException($"Required tensor '{name}' missing from GGUF.");
            }
            var arr = new float[count];
            reader.LoadTensorAsF32(info, arr);
            return arr;
        }

        internal static float[] LoadRouter(GgufReader reader, GgufTensorInfo info, int dModel, int expertCount)
        {
            var n = checked(dModel * expertCount);
            using var raw = new PooledArray<float>(n);
            reader.LoadTensorAsF32(info, raw.Span);
            var src = raw.Span;

            var w = new float[n];
            for (var e = 0; e < expertCount; e++)
            {
                for (var d = 0; d < dModel; d++)
                {
                    w[d * expertCount + e] = src[e * dModel + d];   // [expert,dModel] → [dModel,expert]
                }
            }
            return w;
        }

        /// <summary>
        /// Loads an FFN weight transposed: GGUF stores [in, out], kernel wants [in, out]
        /// in input-major order. Same memory layout actually — direct copy works.
        /// Naming kept for symmetry/clarity with attention weights which DO need per-head transpose.
        /// </summary>
        internal static TensorStorage<float> AllocAndLoadTransposed(GgufReader reader, string name, int inDim, int outDim)
        {
            // For FFN: GGUF file stores W[out, in] at flat (out*inDim + in).
            // Kernel needs W[in, out] at flat (in*outDim + out).
            // Transposition required.
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new InvalidDataException($"Required tensor '{name}' missing from GGUF.");
            }

            var elementCount = checked((int)((long)inDim * outDim));
            using var raw = new PooledArray<float>(elementCount);
            reader.LoadTensorAsF32(info, raw.Span);

            var storage = TensorStorage<float>.Unpooled(elementCount);
            var dst = storage.AsSpan();
            var src = raw.Span;
            for (var i = 0; i < inDim; i++)
            {
                for (var o = 0; o < outDim; o++)
                {
                    dst[i * outDim + o] = src[o * inDim + i];
                }
            }
            return storage;
        }

        /// <summary>
        /// Loads a weight in the best-fit resident format. If the GGUF tensor is
        /// already Q4_K on disk (and dims align), its bytes are kept verbatim
        /// (step 3.2b). Q8_0 → kept as <see cref="Q8Weight"/> (step 2.4). Any
        /// other source (F32 / F16 / BF16, also Q6_K until 3.3 lands its native
        /// kernel) is dequantized to F32 and re-quantized to Q8. The file stores
        /// it [outDim, inDim] — row = one output's contraction vector —
        /// matching both Q4KWeight's and Q8Weight's output-major layout, so no
        /// transpose any path.
        /// </summary>
        internal static DecodeWeight AllocAndLoadResident(
            GgufReader reader, string name, int inDim, int outDim, MemoryMappedModelFile? mmap)
        {
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new InvalidDataException($"Required tensor '{name}' missing from GGUF.");
            }

            if (info.Type == GgmlType.Q4_K && inDim % Q4KWeight.SuperBlockElements == 0)
            {
                return LoadQ4KNative(reader, info, inDim, outDim, mmap);
            }
            if (info.Type == GgmlType.Q6_K && inDim % Q6KWeight.SuperBlockElements == 0)
            {
                return LoadQ6KNative(reader, info, inDim, outDim, mmap);
            }
            if (info.Type == GgmlType.Q8_0)
            {
                return LoadQ8Native(reader, info, inDim, outDim);
            }

            var elementCount = checked((int)((long)inDim * outDim));
            using var raw = new PooledArray<float>(elementCount);
            reader.LoadTensorAsF32(info, raw.Span);
            return Q8Weight.QuantizeRows(raw.Span, outDim, inDim);
        }

        /// <summary>
        /// Reads an already-Q8_0 GGUF tensor straight into a <see cref="Q8Weight"/>
        /// — the native step-2.4 path. GGUF's <c>block_q8_0</c> layout is
        /// output-major with blocks along the contraction dim, exactly matching
        /// <see cref="Q8Weight"/>; the read just de-interleaves (no dequant).
        /// </summary>
        private static Q8Weight LoadQ8Native(GgufReader reader, GgufTensorInfo info, int inDim, int outDim)
        {
            var quants = new sbyte[checked((int)((long)inDim * outDim))];
            var scales = new float[checked((int)((long)outDim * (inDim / Q8DotKernel.BlockSize)))];
            reader.LoadTensorQ8_0Raw(info, quants, scales);
            return new Q8Weight(quants, scales, inDim, outDim);
        }

        /// <summary>
        /// True when the tensor is in a K-quant format we can load natively
        /// (Q4_K, Q6_K, Q8_0) with the given contraction dim. Used by the
        /// per-tensor dispatch (steps 3.2b / 3.3c) to decide whether to rent
        /// F32 scratch.
        /// </summary>
        private static bool IsKQuantNative(GgufTensorInfo info, int dModel)
            => (info.Type == GgmlType.Q4_K && dModel % Q4KWeight.SuperBlockElements == 0)
            || (info.Type == GgmlType.Q6_K && dModel % Q6KWeight.SuperBlockElements == 0)
            || (info.Type == GgmlType.Q8_0 && dModel % Q8DotKernel.BlockSize == 0);

        /// <summary>
        /// Loads a weight as Q4_K-resident — the file's <c>block_q4_K</c> bytes
        /// are kept verbatim (no de-interleave; Q4_K's layout matches
        /// <see cref="Q4KWeight"/>'s exactly). The file stores it [outDim, inDim]
        /// with Q4_K super-blocks along the contraction dim, exactly matching
        /// Q4KWeight's output-major layout (step 3.2b).
        /// </summary>
        private static Q4KWeight LoadQ4KNative(
            GgufReader reader, GgufTensorInfo info, int inDim, int outDim, MemoryMappedModelFile? mmap)
        {
            var blocksPerRow = inDim / Q4KWeight.SuperBlockElements;
            var totalBytes = checked((int)((long)outDim * blocksPerRow * Q4KWeight.SuperBlockBytes));

            if (mmap is not null)
            {
                // Zero-copy: the file's block bytes ARE Q4KWeight's layout, verbatim.
                var slice = mmap.Slice(reader.DataStart + (long)info.Offset, totalBytes);
                return new Q4KWeight(slice, inDim, outDim);
            }

            var bytes = new byte[totalBytes];
            reader.LoadTensorQ4_KRaw(info, bytes);
            return new Q4KWeight(bytes, inDim, outDim);
        }

        /// <summary>
        /// Per-tensor dispatch for an attention Q/K/V weight: native Q4_K /
        /// native Q8_0 / F32-fallback (dequantize-then-split-then-Q8). The F32
        /// fallback uses <paramref name="f32Scratch"/>, which must be rented
        /// when that path can fire. Wo uses <see cref="LoadOutputHeads"/>.
        /// </summary>
        private static DecodeWeight[] LoadQkvHeads(
            GgufReader reader, string name, int headCount, int dModel, int headDim,
            float[] f32Scratch, bool quantizable, MemoryMappedModelFile? mmap)
        {
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new InvalidDataException($"Required tensor '{name}' missing from GGUF.");
            }

            if (quantizable && info.Type == GgmlType.Q4_K && dModel % Q4KWeight.SuperBlockElements == 0)
            {
                return LoadQkvHeadsQ4K(reader, info, headCount, dModel, headDim, mmap);
            }
            if (quantizable && info.Type == GgmlType.Q6_K && dModel % Q6KWeight.SuperBlockElements == 0)
            {
                return LoadQkvHeadsQ6K(reader, info, headCount, dModel, headDim, mmap);
            }
            if (quantizable && info.Type == GgmlType.Q8_0)
            {
                return LoadQkvHeadsQ8(reader, info, headCount, dModel, headDim);
            }

            // F32 fallback — dequant whole tensor into scratch, per-head split.
            // SplitQuery and SplitKeyValue have identical bodies (differ only in
            // the head-count parameter name) — SplitQuery serves both here.
            var elems = checked((int)((long)headCount * dModel * headDim));
            LoadTensor(reader, name, f32Scratch.AsSpan(0, elems));
            return SplitQuery(f32Scratch, headCount, dModel, headDim, quantizable);
        }

        /// <summary>
        /// Per-tensor dispatch for the attention output projection. Wo can't be
        /// K-quant per-head (headDim &lt; the 256-element super-block) but it
        /// CAN be Q8_0 per-head — headDim has integer Q8 blocks of 32. Anything
        /// else falls back to F32 dequant + per-head split (+ Q8 re-quantize).
        /// </summary>
        private static DecodeWeight[] LoadOutputHeads(
            GgufReader reader, string name, int nHeads, int dModel, int headDim,
            float[] oFullScratch, bool quantizable)
        {
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new InvalidDataException($"Required tensor '{name}' missing from GGUF.");
            }

            if (quantizable && info.Type == GgmlType.Q8_0)
            {
                return LoadOutputHeadsQ8(reader, info, nHeads, dModel, headDim);
            }

            // F32 fallback (F16 / F32 / BF16 / Q4_K / Q6_K → LoadTensorAsF32 dequantizes).
            var elems = checked((int)((long)dModel * nHeads * headDim));
            LoadTensor(reader, name, oFullScratch.AsSpan(0, elems));
            return SplitOutput(oFullScratch, nHeads, dModel, headDim, quantizable);
        }

        /// <summary>
        /// Loads an attention Q/K/V weight from an already-Q4_K GGUF tensor and
        /// splits it per head — the native step-3.2b path. The file stores it
        /// [headCount*headDim, dModel] row-major with Q4_K super-blocks along
        /// dModel, so head h is a contiguous output-row range = a contiguous
        /// byte slice of Q4_K super-blocks. Copy that slice into a per-head
        /// <see cref="Q4KWeight"/> — no de-interleave (unlike Q8_0).
        /// </summary>
        private static DecodeWeight[] LoadQkvHeadsQ4K(
            GgufReader reader, GgufTensorInfo info, int headCount, int dModel, int headDim,
            MemoryMappedModelFile? mmap)
        {
            var blocksPerRow = dModel / Q4KWeight.SuperBlockElements;
            var bytesPerRow = blocksPerRow * Q4KWeight.SuperBlockBytes;
            var headBytes = headDim * bytesPerRow;

            if (mmap is not null)
            {
                // Zero-copy: head h's output rows are a contiguous byte run in the file,
                // and the run IS Q4KWeight's layout verbatim — slice it straight in.
                var baseOffset = reader.DataStart + (long)info.Offset;
                var heads = new DecodeWeight[headCount];
                for (var h = 0; h < headCount; h++)
                {
                    var slice = mmap.Slice(baseOffset + (long)h * headBytes, headBytes);
                    heads[h] = new Q4KWeight(slice, dModel, headDim);
                }
                return heads;
            }

            var totalBytes = checked((int)((long)headCount * headBytes));
            using var raw = new PooledArray<byte>(totalBytes);
            reader.LoadTensorQ4_KRaw(info, raw.Span);

            var result = new DecodeWeight[headCount];
            for (var h = 0; h < headCount; h++)
            {
                var bytes = new byte[headBytes];
                raw.Span.Slice(h * headBytes, headBytes).CopyTo(bytes);
                result[h] = new Q4KWeight(bytes, dModel, headDim);
            }
            return result;
        }

        /// <summary>
        /// Loads a weight as Q6_K-resident — the file's <c>block_q6_K</c> bytes
        /// are kept verbatim (no de-interleave; Q6_K's layout matches
        /// <see cref="Q6KWeight"/>'s exactly, like Q4_K). The file stores it
        /// [outDim, inDim] with Q6_K super-blocks along the contraction dim,
        /// exactly matching Q6KWeight's output-major layout (step 3.3c).
        /// </summary>
        private static Q6KWeight LoadQ6KNative(
            GgufReader reader, GgufTensorInfo info, int inDim, int outDim, MemoryMappedModelFile? mmap)
        {
            var blocksPerRow = inDim / Q6KWeight.SuperBlockElements;
            var totalBytes = checked((int)((long)outDim * blocksPerRow * Q6KWeight.SuperBlockBytes));

            if (mmap is not null)
            {
                // Zero-copy: the file's block bytes ARE Q6KWeight's layout, verbatim.
                var slice = mmap.Slice(reader.DataStart + (long)info.Offset, totalBytes);
                return new Q6KWeight(slice, inDim, outDim);
            }

            var bytes = new byte[totalBytes];
            reader.LoadTensorQ6_KRaw(info, bytes);
            return new Q6KWeight(bytes, inDim, outDim);
        }

        /// <summary>
        /// Loads an attention Q/K/V weight from an already-Q6_K GGUF tensor and
        /// splits it per head — the native step-3.3c path. Mirrors
        /// <see cref="LoadQkvHeadsQ4K"/> exactly (same layout shape, different
        /// super-block size of 210 B vs 144 B): head h's rows are a contiguous
        /// byte slice of Q6_K super-blocks, copied straight into a per-head
        /// <see cref="Q6KWeight"/>.
        /// </summary>
        private static DecodeWeight[] LoadQkvHeadsQ6K(
            GgufReader reader, GgufTensorInfo info, int headCount, int dModel, int headDim,
            MemoryMappedModelFile? mmap)
        {
            var blocksPerRow = dModel / Q6KWeight.SuperBlockElements;
            var bytesPerRow = blocksPerRow * Q6KWeight.SuperBlockBytes;
            var headBytes = headDim * bytesPerRow;

            if (mmap is not null)
            {
                // Zero-copy: head h's output rows are a contiguous byte run in the file,
                // and the run IS Q6KWeight's layout verbatim — slice it straight in.
                var baseOffset = reader.DataStart + (long)info.Offset;
                var heads = new DecodeWeight[headCount];
                for (var h = 0; h < headCount; h++)
                {
                    var slice = mmap.Slice(baseOffset + (long)h * headBytes, headBytes);
                    heads[h] = new Q6KWeight(slice, dModel, headDim);
                }
                return heads;
            }

            var totalBytes = checked((int)((long)headCount * headBytes));
            using var raw = new PooledArray<byte>(totalBytes);
            reader.LoadTensorQ6_KRaw(info, raw.Span);

            var result = new DecodeWeight[headCount];
            for (var h = 0; h < headCount; h++)
            {
                var bytes = new byte[headBytes];
                raw.Span.Slice(h * headBytes, headBytes).CopyTo(bytes);
                result[h] = new Q6KWeight(bytes, dModel, headDim);
            }
            return result;
        }

        /// <summary>
        /// Loads an attention Q/K/V weight from an already-Q8_0 GGUF tensor and
        /// splits it per head — the native step-2.4b path. The file stores it
        /// [nHeads*headDim, dModel]; head h is a contiguous output-row range,
        /// hence a contiguous slice of Q8_0 blocks — copied straight into a
        /// per-head <see cref="Q8Weight"/> with no F32 round-trip.
        /// </summary>
        private static DecodeWeight[] LoadQkvHeadsQ8(
            GgufReader reader, GgufTensorInfo info, int nHeads, int dModel, int headDim)
        {
            var blocksPerRow = dModel / Q8DotKernel.BlockSize;
            var elems = checked((int)((long)nHeads * headDim * dModel));
            var totalBlocks = checked((int)((long)nHeads * headDim * blocksPerRow));

            using var quants = new PooledArray<sbyte>(elems);
            using var scales = new PooledArray<float>(totalBlocks);
            reader.LoadTensorQ8_0Raw(info, quants.Span, scales.Span);

            var heads = new DecodeWeight[nHeads];
            for (var h = 0; h < nHeads; h++)
            {
                var headQuants = new sbyte[headDim * dModel];
                var headScales = new float[headDim * blocksPerRow];
                quants.Span.Slice(h * headDim * dModel, headDim * dModel).CopyTo(headQuants);
                scales.Span.Slice(h * headDim * blocksPerRow, headDim * blocksPerRow).CopyTo(headScales);
                heads[h] = new Q8Weight(headQuants, headScales, dModel, headDim);
            }
            return heads;
        }

        /// <summary>
        /// Loads the attention output projection from an already-Q8_0 GGUF
        /// tensor and splits it per head — the native step-2.4b path. The file
        /// stores it [dModel, nHeads*headDim]; head h owns a headDim-wide column
        /// band, so each of the dModel output rows contributes a contiguous
        /// block run, gathered (strided across rows) into a per-head Q8Weight.
        /// </summary>
        private static DecodeWeight[] LoadOutputHeadsQ8(
            GgufReader reader, GgufTensorInfo info, int nHeads, int dModel, int headDim)
        {
            var nHeadsHeadDim = nHeads * headDim;
            var rowBlocks = nHeadsHeadDim / Q8DotKernel.BlockSize;   // blocks per output row
            var headBlocks = headDim / Q8DotKernel.BlockSize;        // blocks per head, per row
            var elems = checked((int)((long)dModel * nHeadsHeadDim));
            var totalBlocks = checked((int)((long)dModel * rowBlocks));

            using var quants = new PooledArray<sbyte>(elems);
            using var scales = new PooledArray<float>(totalBlocks);
            reader.LoadTensorQ8_0Raw(info, quants.Span, scales.Span);

            var heads = new DecodeWeight[nHeads];
            for (var h = 0; h < nHeads; h++)
            {
                var headQuants = new sbyte[dModel * headDim];
                var headScales = new float[dModel * headBlocks];
                for (var o = 0; o < dModel; o++)
                {
                    quants.Span.Slice(o * nHeadsHeadDim + h * headDim, headDim)
                        .CopyTo(headQuants.AsSpan(o * headDim, headDim));
                    scales.Span.Slice(o * rowBlocks + h * headBlocks, headBlocks)
                        .CopyTo(headScales.AsSpan(o * headBlocks, headBlocks));
                }
                heads[h] = new Q8Weight(headQuants, headScales, headDim, dModel);
            }
            return heads;
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
        private static DecodeWeight[] SplitQuery(
            float[] qFull, int nHeads, int dModel, int headDim, bool quantize)
        {
            var wq = new DecodeWeight[nHeads];
            for (var h = 0; h < nHeads; h++)
            {
                if (quantize)
                {
                    // File rows [h*headDim, (h+1)*headDim) are head h's outputs;
                    // each row is that output's contiguous dModel contraction
                    // vector — exactly Q8Weight's output-major layout, no transpose.
                    wq[h] = Q8Weight.QuantizeRows(
                        qFull.AsSpan(h * headDim * dModel, headDim * dModel), headDim, dModel);
                }
                else
                {
                    var storage = TensorStorage<float>.Unpooled(checked((int)((long)dModel * headDim)));
                    var dst = storage.AsSpan();
                    for (var i = 0; i < dModel; i++)
                    {
                        for (var j = 0; j < headDim; j++)
                        {
                            dst[i * headDim + j] = qFull[(h * headDim + j) * dModel + i];
                        }
                    }
                    wq[h] = storage;
                }
            }
            return wq;
        }

        /// <summary>Same per-head split as SplitQuery but iterates over nKvHeads.</summary>
        private static DecodeWeight[] SplitKeyValue(
            float[] kvFull, int nKvHeads, int dModel, int headDim, bool quantize)
        {
            var wkv = new DecodeWeight[nKvHeads];
            for (var kv = 0; kv < nKvHeads; kv++)
            {
                if (quantize)
                {
                    wkv[kv] = Q8Weight.QuantizeRows(
                        kvFull.AsSpan(kv * headDim * dModel, headDim * dModel), headDim, dModel);
                }
                else
                {
                    var storage = TensorStorage<float>.Unpooled(checked((int)((long)dModel * headDim)));
                    var dst = storage.AsSpan();
                    for (var i = 0; i < dModel; i++)
                    {
                        for (var j = 0; j < headDim; j++)
                        {
                            dst[i * headDim + j] = kvFull[(kv * headDim + j) * dModel + i];
                        }
                    }
                    wkv[kv] = storage;
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
        private static DecodeWeight[] SplitOutput(
            float[] oFull, int nHeads, int dModel, int headDim, bool quantize)
        {
            var wo = new DecodeWeight[nHeads];
            var nHeadsHeadDim = nHeads * headDim;
            for (var h = 0; h < nHeads; h++)
            {
                if (quantize)
                {
                    // Output o's headDim contraction vector for head h is the run
                    // oFull[o*nHeadsHeadDim + h*headDim ..]; gather the dModel rows
                    // contiguously into output-major order, then quantize.
                    var gatherElems = checked((int)((long)dModel * headDim));
                    using var gather = new PooledArray<float>(gatherElems);
                    for (var o = 0; o < dModel; o++)
                    {
                        for (var i = 0; i < headDim; i++)
                        {
                            gather.Span[o * headDim + i] = oFull[o * nHeadsHeadDim + h * headDim + i];
                        }
                    }
                    wo[h] = Q8Weight.QuantizeRows(gather.Span, dModel, headDim);
                }
                else
                {
                    var storage = TensorStorage<float>.Unpooled(checked((int)((long)headDim * dModel)));
                    var dst = storage.AsSpan();
                    for (var i = 0; i < headDim; i++)
                    {
                        for (var j = 0; j < dModel; j++)
                        {
                            dst[i * dModel + j] = oFull[j * nHeadsHeadDim + h * headDim + i];
                        }
                    }
                    wo[h] = storage;
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
    }
}
