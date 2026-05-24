// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
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
        /// <param name="path">Path to the GGUF model file.</param>
        /// <param name="quantize">
        /// When true (default) FFN / LM-head / attention weights become Q8_0-resident
        /// where dimensions allow. When false every weight loads as F32 — the
        /// pre-quantization decode path, used as the parity reference (step 2.5).
        /// </param>
        /// <param name="mmap">
        /// When true the verbatim-layout K-quant weights (Q4_K / Q6_K — FFN, LM head, and
        /// per-head attention Q/K/V) are backed by zero-copy slices of a read-only memory map
        /// of the file instead of managed <c>byte[]</c> copies. The map's lifetime is handed
        /// to the returned engine, which disposes it last. Cuts committed RAM for those
        /// weights to zero (the OS pages them in as shared/clean working set). No effect on
        /// Q8_0 (de-interleaved) or F32-fallback weights, which still copy.
        /// </param>
        public static CachedLlamaInferenceEngine Load(string path, bool quantize = true, bool mmap = false)
        {
            using var reader = new GgufReader(path);
            if (!mmap)
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
                    // FFN weights — step 2.3b: resident as Q8_0 (output-major)
                    // when quantization is enabled and both dims are block-aligned;
                    // F32 transposed otherwise.
                    DecodeWeight ffnGate, ffnUp, ffnDown;
                    if (quantize && dModel % Q8DotKernel.BlockSize == 0 && dFF % Q8DotKernel.BlockSize == 0)
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
                    lmHead = Q8Weight.QuantizeRows(embedWeights.AsReadOnlySpan(), vocab, dModel);
                }
                else
                {
                    var outElems = checked((int)((long)vocab * dModel));
                    var outputRaw = ArrayPool<float>.Shared.Rent(outElems);
                    try
                    {
                        LoadTensor(reader, "output.weight", outputRaw.AsSpan(0, outElems));
                        lmHead = Q8Weight.QuantizeRows(outputRaw.AsSpan(0, outElems), vocab, dModel);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(outputRaw);
                    }
                }
            }
            else
            {
                // F32 fallback — transpose [vocab, dModel] → [dModel, vocab].
                var f32LmHead = TensorStorage<float>.Unpooled(checked((int)((long)vocab * dModel)));
                var lmHeadSpan = f32LmHead.AsSpan();
                if (tieWeights)
                {
                    var embSpan = embedWeights.AsReadOnlySpan();
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

                lmHead = f32LmHead;
            }

            return CachedLlamaInferenceEngine.CreateFromBuffers(
                config, embedWeights, finalNormGamma, finalNormBeta, lmHead, layers, mmap);
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
        private static DecodeWeight AllocAndLoadResident(
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
            var raw = ArrayPool<float>.Shared.Rent(elementCount);
            try
            {
                reader.LoadTensorAsF32(info, raw.AsSpan(0, elementCount));
                return Q8Weight.QuantizeRows(raw.AsSpan(0, elementCount), outDim, inDim);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(raw);
            }
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
            var raw = ArrayPool<byte>.Shared.Rent(totalBytes);
            try
            {
                reader.LoadTensorQ4_KRaw(info, raw.AsSpan(0, totalBytes));

                var heads = new DecodeWeight[headCount];
                for (var h = 0; h < headCount; h++)
                {
                    var bytes = new byte[headBytes];
                    raw.AsSpan(h * headBytes, headBytes).CopyTo(bytes);
                    heads[h] = new Q4KWeight(bytes, dModel, headDim);
                }
                return heads;
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(raw);
            }
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
            var raw = ArrayPool<byte>.Shared.Rent(totalBytes);
            try
            {
                reader.LoadTensorQ6_KRaw(info, raw.AsSpan(0, totalBytes));

                var heads = new DecodeWeight[headCount];
                for (var h = 0; h < headCount; h++)
                {
                    var bytes = new byte[headBytes];
                    raw.AsSpan(h * headBytes, headBytes).CopyTo(bytes);
                    heads[h] = new Q6KWeight(bytes, dModel, headDim);
                }
                return heads;
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(raw);
            }
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

            var quants = ArrayPool<sbyte>.Shared.Rent(elems);
            var scales = ArrayPool<float>.Shared.Rent(totalBlocks);
            try
            {
                reader.LoadTensorQ8_0Raw(info, quants.AsSpan(0, elems), scales.AsSpan(0, totalBlocks));

                var heads = new DecodeWeight[nHeads];
                for (var h = 0; h < nHeads; h++)
                {
                    var headQuants = new sbyte[headDim * dModel];
                    var headScales = new float[headDim * blocksPerRow];
                    quants.AsSpan(h * headDim * dModel, headDim * dModel).CopyTo(headQuants);
                    scales.AsSpan(h * headDim * blocksPerRow, headDim * blocksPerRow).CopyTo(headScales);
                    heads[h] = new Q8Weight(headQuants, headScales, dModel, headDim);
                }
                return heads;
            }
            finally
            {
                ArrayPool<sbyte>.Shared.Return(quants);
                ArrayPool<float>.Shared.Return(scales);
            }
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

            var quants = ArrayPool<sbyte>.Shared.Rent(elems);
            var scales = ArrayPool<float>.Shared.Rent(totalBlocks);
            try
            {
                reader.LoadTensorQ8_0Raw(info, quants.AsSpan(0, elems), scales.AsSpan(0, totalBlocks));

                var heads = new DecodeWeight[nHeads];
                for (var h = 0; h < nHeads; h++)
                {
                    var headQuants = new sbyte[dModel * headDim];
                    var headScales = new float[dModel * headBlocks];
                    for (var o = 0; o < dModel; o++)
                    {
                        quants.AsSpan(o * nHeadsHeadDim + h * headDim, headDim)
                            .CopyTo(headQuants.AsSpan(o * headDim, headDim));
                        scales.AsSpan(o * rowBlocks + h * headBlocks, headBlocks)
                            .CopyTo(headScales.AsSpan(o * headBlocks, headBlocks));
                    }
                    heads[h] = new Q8Weight(headQuants, headScales, headDim, dModel);
                }
                return heads;
            }
            finally
            {
                ArrayPool<sbyte>.Shared.Return(quants);
                ArrayPool<float>.Shared.Return(scales);
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
                    var gather = ArrayPool<float>.Shared.Rent(gatherElems);
                    try
                    {
                        for (var o = 0; o < dModel; o++)
                        {
                            for (var i = 0; i < headDim; i++)
                            {
                                gather[o * headDim + i] = oFull[o * nHeadsHeadDim + h * headDim + i];
                            }
                        }
                        wo[h] = Q8Weight.QuantizeRows(gather.AsSpan(0, gatherElems), dModel, headDim);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(gather);
                    }
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
