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

            // RoPE pairing depends on how llama.cpp stored Q/K for this arch. Qwen2/Qwen2-MoE use NEOX
            // rope and llama.cpp leaves Q/K in the original HF layout → split-half pairing (x[i],x[i+d/2]).
            // Llama/Mistral use NORM rope and llama.cpp PERMUTES Q/K at conversion into adjacent layout →
            // adjacent pairing (x[2i],x[2i+1]), which is the default. Getting this wrong leaves position 0
            // correct but corrupts every later position (attention collapses onto the current token).
            var ropeSplitHalf = arch is "qwen2" or "qwen2moe" or "qwen3" or "qwen3moe" or "phi3" or "gemma2";

            // Gemma-2: GeGLU FFN, (1+w) RMSNorm, embedding ×√d_model, sandwich norm (post_attention/post_ffw),
            // attn + final logit soft-capping, and alternating sliding-window attention (deferred — we cap the
            // context at the sliding window so it's a no-op).
            var isGemma = arch == "gemma2";
            // (1+w) RMSNorm: the llama.cpp gemma GGUF ALREADY bakes the +1 into the stored norm weights — adding it
            // again double-applies and produces garbage (A/B-verified: no-offset → "Paris", offset → repeated punct).
            // So DON'T add +1 here. Embedding ×√d_model IS needed (also A/B-verified).
            var gemmaEmbeddingScale = isGemma ? MathF.Sqrt(dModel) : 1f;
            var attnSoftcap = isGemma ? reader.GetMeta($"{arch}.attn_logit_softcapping", 0f) : 0f;
            var finalSoftcap = isGemma ? reader.GetMeta($"{arch}.final_logit_softcapping", 0f) : 0f;
            var gemmaSlidingWindow = isGemma ? reader.GetMeta($"{arch}.attention.sliding_window", 4096) : 0;

            // Cap context length for memory sanity (32k+ models work but consume RAM). Gemma-2: cap at the sliding
            // window so the (deferred) alternating local/global attention is a no-op.
            if (ctxLen > 8192)
            {
                ctxLen = 8192;
            }
            if (isGemma && gemmaSlidingWindow > 0 && ctxLen > gemmaSlidingWindow)
            {
                ctxLen = gemmaSlidingWindow;
            }

            // Vocab from tokenizer metadata if present, else from token_embd shape
            var vocab = reader.GetMeta($"{arch}.vocab_size", 0);
            if (vocab == 0 && reader.Tensors.TryGetValue("token_embd.weight", out var embInfo))
            {
                // GGUF dim order: [dModel, vocab]. Last dim is vocab.
                vocab = (int)embInfo.Dims[^1];
            }
            if (vocab == 0)
            {
                throw new OverfitFormatException("Cannot determine vocab size.");
            }

            // Qwen3 sets head_dim explicitly (head_dim ≠ dModel/nHeads, so the q/k/v projections are not square);
            // every other arch we load derives it as dModel/nHeads.
            var headDim = reader.GetMeta($"{arch}.attention.key_length", dModel / nHeads);
            if (headDim <= 0 || (headDim == dModel / nHeads && headDim * nHeads != dModel))
            {
                throw new OverfitFormatException(
                    $"Could not determine head_dim for arch '{arch}' (dModel {dModel}, nHeads {nHeads}).");
            }
            var usesQkNorm = arch is "qwen3" or "qwen3moe";

            // Phi-3 packs Q/K/V into one fused `attn_qkv.weight` and gate+up into one fused `ffn_up.weight`
            // (2×dFF wide), and uses NEOX (split-half) RoPE with a "longrope" per-dimension frequency rescale
            // (rope_factors_short/long [head_dim/2]). We dequant-split the fused tensors at load and read the
            // short-context factor array (exact for ≤ original_context_length; mscale = 1 in that regime).
            var fusedQkv = arch == "phi3";
            var fusedGateUp = arch == "phi3";
            var ropeOriginalCtx = reader.GetMeta($"{arch}.rope.scaling.original_context_length", 0);
            float[]? ropeFreqFactors = null;
            if (arch == "phi3" && reader.Tensors.ContainsKey("rope_factors_short.weight"))
            {
                ropeFreqFactors = LoadF32Vector(reader, "rope_factors_short.weight", headDim / 2);
                // Cap context to the short-factor regime (original training context) so the stored short
                // factors are exact and the long-rope attention mscale stays 1.0. Long-context (long_factor +
                // mscale) is a follow-on.
                if (ropeOriginalCtx > 0 && ctxLen > ropeOriginalCtx)
                {
                    ctxLen = ropeOriginalCtx;
                }
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
                HeadDim = headDim,
                UsesQkNorm = usesQkNorm,
                VocabSize = vocab,
                ContextLength = ctxLen,
                DFF = dFF,
                UseRoPE = true,
                RoPETheta = ropeTheta,
                RopeSplitHalf = ropeSplitHalf,
                RopeFreqFactors = ropeFreqFactors,
                RopeAttnFactor = 1f,
                FfnActivation = isGemma ? FeedForwardActivation.GeGLU : FeedForwardActivation.SwiGLU,
                EmbeddingScale = gemmaEmbeddingScale,
                AttnLogitSoftcap = attnSoftcap,
                FinalLogitSoftcap = finalSoftcap,
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

            // F32 scratch is needed for any tensor that isn't K-quant-native on disk (the F32-fallback path).
            // A Q4_K_M file is NOT type-uniform across layers — llama.cpp varies the quant per layer (e.g. some
            // attn_v are Q8_0, others Q5_0) — so we must scan EVERY layer, not just blk.0, or a later F32-fallback
            // layer would dereference a scratch buffer that was never rented.
            // Phi-3 stores Q/K/V fused (no attn_q/k/v tensors) — skip the per-tensor scratch + scan and
            // dequant the one fused tensor into a dedicated buffer instead.
            var qNeedsF32 = !fusedQkv && (!attnQuantizable || AnyLayerNeedsF32(reader, "attn_q", nLayers, dModel));
            var kNeedsF32 = !fusedQkv && (!attnQuantizable || AnyLayerNeedsF32(reader, "attn_k", nLayers, dModel));
            var vNeedsF32 = !fusedQkv && (!attnQuantizable || AnyLayerNeedsF32(reader, "attn_v", nLayers, dModel));
            // Wo: K-quant per-head is blocked by headDim < 256, but Q8_0 per-head works (else F32 fallback).
            var oNeedsF32 = !attnQuantizable || AnyLayerOutputNeedsF32(reader, nLayers);

            using var qFull = qNeedsF32 ? new PooledBuffer<float>(qFullElems, clearMemory: false) : default;
            using var kFull = kNeedsF32 ? new PooledBuffer<float>(kFullElems, clearMemory: false) : default;
            using var vFull = vNeedsF32 ? new PooledBuffer<float>(kFullElems, clearMemory: false) : default;
            using var oFull = oNeedsF32 ? new PooledBuffer<float>(oFullElems, clearMemory: false) : default;
            // Fused-tensor scratch (Phi-3): qkv = (nHeads+2·nKvHeads)·headDim × dModel; gate_up = 2·dFF × dModel.
            var fusedQkvElems = fusedQkv ? checked((int)((long)((nHeads + (2 * nKvHeads)) * headDim) * dModel)) : 0;
            var fusedGateUpElems = fusedGateUp ? checked((int)((long)(2 * dFF) * dModel)) : 0;
            using var qkvFused = fusedQkv ? new PooledBuffer<float>(fusedQkvElems, clearMemory: false) : default;
            using var gateUpFused = fusedGateUp ? new PooledBuffer<float>(fusedGateUpElems, clearMemory: false) : default;
            // Attention biases are always F32 in GGUF — bias scratch always needed.
            using var qBiasFull = new PooledBuffer<float>(nHeads * headDim, clearMemory: false);
            using var kBiasFull = new PooledBuffer<float>(nKvHeads * headDim, clearMemory: false);
            using var vBiasFull = new PooledBuffer<float>(nKvHeads * headDim, clearMemory: false);
            // Gemma's (1+w) RMSNorm offset is already baked into the GGUF weights (see note above), so this is a
            // plain load — the wrapper is kept so all gemma norm tensors (incl. the sandwich post-norms) go through
            // one place if the offset ever needs reinstating for a differently-converted GGUF.
            TensorStorage<float> LoadNormGamma(string name) => AllocAndLoad(reader, name, dModel);

            for (var l = 0; l < nLayers; l++)
            {
                // Attention LayerNorm gamma (RMSNorm, no beta)
                var attnNormGamma = LoadNormGamma($"blk.{l}.attn_norm.weight");
                var attnNormBeta = TensorStorage<float>.Unpooled(0);

                // Gemma-2 sandwich norm: extra RMSNorm after attention and after FFN (before each residual).
                TensorStorage<float>? postAttnNorm = null, postFfwNorm = null;
                if (isGemma)
                {
                    postAttnNorm = LoadNormGamma($"blk.{l}.post_attention_norm.weight");
                    postFfwNorm = LoadNormGamma($"blk.{l}.post_ffw_norm.weight");
                }

                // Qwen3 per-head RMSNorm weights on Q and K (over head_dim), applied before RoPE.
                TensorStorage<float>? qNorm = null, kNorm = null;
                if (usesQkNorm)
                {
                    qNorm = AllocAndLoad(reader, $"blk.{l}.attn_q_norm.weight", headDim);
                    kNorm = AllocAndLoad(reader, $"blk.{l}.attn_k_norm.weight", headDim);
                }

                // Per-tensor dispatch (step 3.2b) — each of attn_q/k/v picks
                // Q4_K-native / Q8_0-native / F32-fallback independently from
                // its file format. Wo dispatches separately (Q8_0 OK per-head;
                // K-quant not — headDim < the 256-element super-block).
                DecodeWeight[] wq, wk, wv;
                if (fusedQkv)
                {
                    // Phi-3: one fused attn_qkv [out=(nHeads+2·nKvHeads)·headDim, in=dModel], output-major
                    // = [Q rows | K rows | V rows]. Dequant once, slice the three contiguous row ranges, and
                    // run the same per-head split (→ per-head Q8 when attnQuantizable). Row boundaries land on
                    // whole output neurons, so the slices are exact regardless of the on-disk quant.
                    LoadTensor(reader, $"blk.{l}.attn_qkv.weight", qkvFused.Span.Slice(0, fusedQkvElems));
                    var qElems = nHeads * headDim * dModel;
                    var kvElems = nKvHeads * headDim * dModel;
                    wq = SplitQuery(qkvFused.Span.Slice(0, qElems), nHeads, dModel, headDim, attnQuantizable);
                    wk = SplitKeyValue(qkvFused.Span.Slice(qElems, kvElems), nKvHeads, dModel, headDim, attnQuantizable);
                    wv = SplitKeyValue(qkvFused.Span.Slice(qElems + kvElems, kvElems), nKvHeads, dModel, headDim, attnQuantizable);
                }
                else
                {
                    wq = LoadQkvHeads(reader, $"blk.{l}.attn_q.weight", nHeads, dModel, headDim, qFull.Span, attnQuantizable, mmap);
                    wk = LoadQkvHeads(reader, $"blk.{l}.attn_k.weight", nKvHeads, dModel, headDim, kFull.Span, attnQuantizable, mmap);
                    wv = LoadQkvHeads(reader, $"blk.{l}.attn_v.weight", nKvHeads, dModel, headDim, vFull.Span, attnQuantizable, mmap);
                }
                var wo = LoadOutputHeads(reader, $"blk.{l}.attn_output.weight", nHeads, dModel, headDim, oFull.Span, attnQuantizable);

                // Whole-matrix Q4_K attention handles (M2 plumbing; empty unless Q4_K + mmap + repackable).
                // Output-major dims: Q/K/V contract over dModel; O contracts over nHeads·headDim. Dormant
                // until the M3 OVERFIT_REPACK_ATTN decode path; the per-head wq/wk/wv/wo above stay active.
                // (Note: whole-O reads the on-disk Q4_K bytes directly even though per-head O is dequantized.)
                var wqWhole = TryLoadWholeAttnQ4K(reader, $"blk.{l}.attn_q.weight", nHeads * headDim, dModel, mmap);
                var wkWhole = TryLoadWholeAttnQ4K(reader, $"blk.{l}.attn_k.weight", nKvHeads * headDim, dModel, mmap);
                var wvWhole = TryLoadWholeAttnQ4K(reader, $"blk.{l}.attn_v.weight", nKvHeads * headDim, dModel, mmap);
                var woWhole = TryLoadWholeAttnQ4K(reader, $"blk.{l}.attn_output.weight", dModel, nHeads * headDim, mmap);

                // Attention biases (optional — Qwen has them, Llama doesn't);
                // always F32 in GGUF, never quantized.
                LoadTensorOrZeros(reader, $"blk.{l}.attn_q.bias", qBiasFull.Span.Slice(0, nHeads * headDim));
                LoadTensorOrZeros(reader, $"blk.{l}.attn_k.bias", kBiasFull.Span.Slice(0, nKvHeads * headDim));
                LoadTensorOrZeros(reader, $"blk.{l}.attn_v.bias", vBiasFull.Span.Slice(0, nKvHeads * headDim));
                var bq = SplitBias(qBiasFull.Span, nHeads, headDim);
                var bk = SplitBias(kBiasFull.Span, nKvHeads, headDim);
                var bv = SplitBias(vBiasFull.Span, nKvHeads, headDim);
                var bo = new TensorStorage<float>[nHeads];
                for (var h = 0; h < nHeads; h++)
                {
                    bo[h] = TensorStorage<float>.Unpooled(dModel);
                }

                // FFN
                var ffnNormGamma = LoadNormGamma($"blk.{l}.ffn_norm.weight");
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
                else if (fusedGateUp)
                {
                    // Phi-3: one fused ffn_up [out=2·dFF, in=dModel], output-major = [gate rows | up rows]
                    // (HF gate_up_proj → chunk(2): first half gate, second half up). Dequant, slice the two
                    // halves, requantize each to Q8 (compact, ≥ the on-disk precision). ffn_down is separate.
                    LoadTensor(reader, $"blk.{l}.ffn_up.weight", gateUpFused.Span.Slice(0, fusedGateUpElems));
                    var half = dFF * dModel;
                    ffnGate = Q8Weight.QuantizeRows(gateUpFused.Span.Slice(0, half), dFF, dModel);
                    ffnUp = Q8Weight.QuantizeRows(gateUpFused.Span.Slice(half, half), dFF, dModel);
                    ffnDown = AllocAndLoadResident(reader, $"blk.{l}.ffn_down.weight", dFF, dModel, mmap);
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
                    QNorm = qNorm,
                    KNorm = kNorm,
                    PostAttnNorm = postAttnNorm,
                    PostFfwNorm = postFfwNorm,
                    AttnNormBeta = attnNormBeta,
                    Wq = wq,
                    Bq = bq,
                    Wk = wk,
                    Bk = bk,
                    Wv = wv,
                    Bv = bv,
                    Wo = wo,
                    Bo = bo,
                    WqWhole = wqWhole,
                    WkWhole = wkWhole,
                    WvWhole = wvWhole,
                    WoWhole = woWhole,
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

            // ─── Final norm + LM head ─────────────────────────────────────
            var finalNormGamma = LoadNormGamma("output_norm.weight");
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
                    using var outputRaw = new PooledBuffer<float>(outElems, clearMemory: false);
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
                    using var outputRaw = new PooledBuffer<float>(outElems, clearMemory: false);
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
                throw new OverfitFormatException("Required tensor 'token_embd.weight' missing from GGUF.");
            }
            if (info.ElementCount != (long)vocab * dModel)
            {
                throw new OverfitFormatException(
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
                throw new OverfitFormatException($"Required tensor '{name}' missing from GGUF.");
            }
            if (info.ElementCount != elementCount)
            {
                throw new OverfitFormatException(
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
                throw new OverfitFormatException(
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
                        using var whole = new PooledBuffer<byte>(total, clearMemory: false);

                        if (info.Type == GgmlType.Q4_K)
                        {
                            reader.LoadTensorQ4_KRaw(info, whole.Span);
                        }
                        else
                        {
                            reader.LoadTensorQ6_KRaw(info, whole.Span);
                        }

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
                        using var quants = new PooledBuffer<sbyte>(elems, clearMemory: false);
                        using var scales = new PooledBuffer<float>(blocksPerExpert * expertCount, clearMemory: false);
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
                        using var f32 = new PooledBuffer<float>(perElems, clearMemory: false);
                        for (var e = 0; e < expertCount; e++)
                        {
                            reader.LoadQ5RegionAsF32(info, (long)e * perElems, f32.Span);
                            experts[e] = Q8Weight.QuantizeRows(f32.Span, outDim, inDim);
                        }
                        break;
                    }

                default:
                    throw new OverfitRuntimeException(
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
                throw new OverfitFormatException($"Expert tensor '{info.Name}' must be 3-D, got {info.Dims.Length} dims.");
            }

            var inDim = checked((int)info.Dims[0]);
            var outDim = checked((int)info.Dims[1]);
            var expertCount = checked((int)info.Dims[2]);
            var perExpert = checked(inDim * outDim);
            var total = checked(perExpert * expertCount);

            using var whole = new PooledBuffer<float>(total, clearMemory: false);
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
                throw new OverfitFormatException($"Required tensor '{name}' missing from GGUF.");
            }
            var arr = new float[count];
            reader.LoadTensorAsF32(info, arr);
            return arr;
        }

        internal static float[] LoadRouter(GgufReader reader, GgufTensorInfo info, int dModel, int expertCount)
        {
            var n = checked(dModel * expertCount);
            using var raw = new PooledBuffer<float>(n, clearMemory: false);
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
                throw new OverfitFormatException($"Required tensor '{name}' missing from GGUF.");
            }

            var elementCount = checked((int)((long)inDim * outDim));
            using var raw = new PooledBuffer<float>(elementCount, clearMemory: false);
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
                throw new OverfitFormatException($"Required tensor '{name}' missing from GGUF.");
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
            using var raw = new PooledBuffer<float>(elementCount, clearMemory: false);
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

        // A Q4_K_M GGUF varies the quant per layer (e.g. some attn_v are Q8_0, others Q5_0), so the
        // scratch-renting decision must scan EVERY layer: true if any layer's blk.{l}.{suffix}.weight is not
        // K-quant-native and would therefore take the F32 fallback (which needs the rented scratch).
        private static bool AnyLayerNeedsF32(GgufReader reader, string suffix, int nLayers, int dModel)
        {
            for (var l = 0; l < nLayers; l++)
            {
                if (reader.Tensors.TryGetValue($"blk.{l}.{suffix}.weight", out var info) && !IsKQuantNative(info, dModel))
                {
                    return true;
                }
            }
            return false;
        }

        // Wo per-head can be Q8_0-native; anything else takes the F32 fallback. True if any layer's Wo isn't Q8_0.
        private static bool AnyLayerOutputNeedsF32(GgufReader reader, int nLayers)
        {
            for (var l = 0; l < nLayers; l++)
            {
                if (reader.Tensors.TryGetValue($"blk.{l}.attn_output.weight", out var info) && info.Type != GgmlType.Q8_0)
                {
                    return true;
                }
            }
            return false;
        }

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
            Span<float> f32Scratch, bool quantizable, MemoryMappedModelFile? mmap)
        {
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new OverfitFormatException($"Required tensor '{name}' missing from GGUF.");
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
            LoadTensor(reader, name, f32Scratch.Slice(0, elems));
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
            Span<float> oFullScratch, bool quantizable)
        {
            if (!reader.Tensors.TryGetValue(name, out var info))
            {
                throw new OverfitFormatException($"Required tensor '{name}' missing from GGUF.");
            }

            if (quantizable && info.Type == GgmlType.Q8_0)
            {
                return LoadOutputHeadsQ8(reader, info, nHeads, dModel, headDim);
            }

            // F32 fallback (F16 / F32 / BF16 / Q4_K / Q6_K → LoadTensorAsF32 dequantizes).
            var elems = checked((int)((long)dModel * nHeads * headDim));
            LoadTensor(reader, name, oFullScratch.Slice(0, elems));
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
            using var raw = new PooledBuffer<byte>(totalBytes, clearMemory: false);
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
        /// Builds the WHOLE-matrix Q4_K attention handle (M2 plumbing for the M3 <c>OVERFIT_REPACK_ATTN</c>
        /// decode lever) — a single repacked 8×8 GEMV over Q/K/V/O beats today's per-head Q4_K projections
        /// (measured 2.55× ‖). Returns empty unless the on-disk tensor is Q4_K AND memory-mapped (so the
        /// whole output-row range is one contiguous, zero-copy byte slice) AND the dims are repackable for
        /// the GEMV (<c>inputSize % 256 == 0</c>, <c>outputSize % 8 == 0</c>). It is a SECOND read-only view
        /// of the same mmap bytes the per-head loaders slice — additive, owns nothing, the mmap (disposed
        /// last) outlives it. Empty for fused-QKV (Phi-3), non-mmap, or non-Q4_K tensors → decode keeps the
        /// per-head path. Output-major dims: Q/K/V are [nHeads·headDim, dModel]; O is [dModel, nHeads·headDim].
        /// </summary>
        private static DecodeWeight TryLoadWholeAttnQ4K(
            GgufReader reader, string name, int outputSize, int inputSize, MemoryMappedModelFile? mmap)
        {
            if (mmap is null
                || !reader.Tensors.TryGetValue(name, out var info)
                || info.Type != GgmlType.Q4_K
                || inputSize % Q4KWeight.SuperBlockElements != 0
                || outputSize % Q4KRepack.RowsInterleaved != 0)
            {
                return default;
            }

            var bytesPerRow = inputSize / Q4KWeight.SuperBlockElements * Q4KWeight.SuperBlockBytes;
            var totalBytes = checked((int)((long)outputSize * bytesPerRow));
            var baseOffset = reader.DataStart + (long)info.Offset;
            return new Q4KWeight(mmap.Slice(baseOffset, totalBytes), inputSize, outputSize);
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
            using var raw = new PooledBuffer<byte>(totalBytes, clearMemory: false);
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

            using var quants = new PooledBuffer<sbyte>(elems, clearMemory: false);
            using var scales = new PooledBuffer<float>(totalBlocks, clearMemory: false);
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

            using var quants = new PooledBuffer<sbyte>(elems, clearMemory: false);
            using var scales = new PooledBuffer<float>(totalBlocks, clearMemory: false);
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
                throw new OverfitFormatException($"Required tensor '{name}' missing from GGUF.");
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
            ReadOnlySpan<float> qFull, int nHeads, int dModel, int headDim, bool quantize)
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
                        qFull.Slice(h * headDim * dModel, headDim * dModel), headDim, dModel);
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
            ReadOnlySpan<float> kvFull, int nKvHeads, int dModel, int headDim, bool quantize)
        {
            var wkv = new DecodeWeight[nKvHeads];
            for (var kv = 0; kv < nKvHeads; kv++)
            {
                if (quantize)
                {
                    wkv[kv] = Q8Weight.QuantizeRows(
                        kvFull.Slice(kv * headDim * dModel, headDim * dModel), headDim, dModel);
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
            ReadOnlySpan<float> oFull, int nHeads, int dModel, int headDim, bool quantize)
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
                    using var gather = new PooledBuffer<float>(gatherElems, clearMemory: false);
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

        private static TensorStorage<float>[] SplitBias(ReadOnlySpan<float> biasFull, int nHeads, int headDim)
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
