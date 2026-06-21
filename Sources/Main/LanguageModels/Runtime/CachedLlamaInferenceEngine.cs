// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Inference engine for SLM models loaded from Overfit binary format
    /// (produced by Scripts/convert_llama.py or Scripts/convert_gguf.py).
    ///
    /// Supports:
    ///   - GQA (Llama-3.2, Qwen2.5, Mistral)
    ///   - RoPE (Llama, Qwen, Phi)
    ///   - SwiGLU FFN (all modern SLMs)
    ///   - Standard MHA + GeLU (GPT-1, GPT-2)
    ///
    /// Usage:
    ///   using var engine = CachedLlamaInferenceEngine.Load("test_fixtures/qwen.bin");
    ///   using var session = engine.CreateSession();
    ///   session.Reset(promptTokens);
    ///   var options = SamplingOptions.Greedy;
    ///   int nextToken = session.GenerateNextToken(in options);
    /// </summary>
    public sealed class CachedLlamaInferenceEngine : IDisposable
    {
        private const uint FilemagicExpected = 0x4F565246u;  // "OVRF"
        private const int VersionExpected = 2;

        private readonly GPT1Config _config;
        private readonly RopeTable? _rope;
        private readonly CachedGptStack _stack;

        // Loaded weights — owned by this engine, disposed on Dispose()
        // Embedding table [vocab × dModel] (row = token): F32, or K-quant-resident (Q4_K/Q6_K,
        // possibly mmap-backed) when loaded from a quantized GGUF — the lookup dequantizes per row.
        private readonly DecodeWeight _embedWeights;
        private readonly TensorStorage<float> _finalNormGamma;
        private readonly TensorStorage<float> _finalNormBeta;
        private readonly DecodeWeight _lmHead;
        private readonly LayerWeightBuffers[] _layers;
        private readonly StackWeights _stackWeights;

        // Memory map backing zero-copy mmap-resident weights, if any. Owned by this
        // engine and disposed LAST (after the weights that slice into it) — see Dispose().
        private readonly IDisposable? _backingFile;

        private bool _disposed;

        // ── Internal weight container per layer ───────────────────────────────

        public sealed class LayerWeightBuffers
        {
            public required TensorStorage<float> AttnNormGamma;
            public required TensorStorage<float> AttnNormBeta;

            // Qwen3 per-head RMSNorm on Q and K (over head_dim), applied before RoPE. Null for every other arch.
            public TensorStorage<float>? QNorm;
            public TensorStorage<float>? KNorm;
            public TensorStorage<float>? PostAttnNorm;   // Gemma-2 sandwich norm after attention
            public TensorStorage<float>? PostFfwNorm;    // Gemma-2 sandwich norm after FFN

            public required DecodeWeight[] Wq;
            public required TensorStorage<float>[] Bq;
            public required DecodeWeight[] Wk;
            public required TensorStorage<float>[] Bk;
            public required DecodeWeight[] Wv;
            public required TensorStorage<float>[] Bv;
            public required DecodeWeight[] Wo;
            public required TensorStorage<float>[] Bo;

            // Whole-matrix Q4_K attention handles (M2 plumbing for the OVERFIT_REPACK_ATTN decode lever, M3).
            // Empty (default) unless the on-disk Q/K/V/O tensor is Q4_K + memory-mapped + repackable for the
            // 8×8 GEMV — then each is a SECOND zero-copy view of the same mmap bytes the per-head Wq/Wk/Wv/Wo
            // above slice. The per-head arrays stay the active decode path; M3 consumes these when present.
            public DecodeWeight WqWhole;
            public DecodeWeight WkWhole;
            public DecodeWeight WvWhole;
            public DecodeWeight WoWhole;
            public required TensorStorage<float> FfnNormGamma;
            public required TensorStorage<float> FfnNormBeta;
            public required DecodeWeight FfnGate;
            public required DecodeWeight FfnUp;
            public required DecodeWeight FfnDown;

            // Mixture of Experts (qwen2moe) — null/default for a dense FFN layer.
            public float[]? MoeRouter;
            public DecodeWeight[]? MoeGate;
            public DecodeWeight[]? MoeUp;
            public DecodeWeight[]? MoeDown;
            public DecodeWeight MoeSharedGate;
            public DecodeWeight MoeSharedUp;
            public DecodeWeight MoeSharedDown;
            public float[]? MoeSharedGateInp;
        }

        // ── Constructor ───────────────────────────────────────────────────────

        private CachedLlamaInferenceEngine(
            GPT1Config config,
            DecodeWeight embedWeights,
            TensorStorage<float> finalNormGamma,
            TensorStorage<float> finalNormBeta,
            DecodeWeight lmHead,
            LayerWeightBuffers[] layers,
            IDisposable? backingFile = null)
        {
            _config = config;
            _embedWeights = embedWeights;
            _finalNormGamma = finalNormGamma;
            _finalNormBeta = finalNormBeta;
            _lmHead = lmHead;
            _layers = layers;
            _backingFile = backingFile;

            var headDim = config.AttentionHeadDim;

            // Build RoPE table if required (over head_dim, which Qwen3 sets explicitly ≠ DModel/NHeads).
            // Phi-3 passes longrope per-dim freq factors + attn scaling; other archs leave them null/1.
            _rope = config.UseRoPE
                ? new RopeTable(config.ContextLength, headDim, config.RoPETheta, config.RopeScaling, config.RopeSplitHalf, config.RopeFreqFactors, config.RopeAttnFactor)
                : null;

            _stack = new CachedGptStack(
                config.NLayers,
                config.DModel,
                config.NHeads,
                config.DFF,
                config.VocabSize,
                config.ContextLength,
                layerNormEpsilon: 1e-6f,
                config.FfnActivation,   // SwiGLU for Llama/Qwen/Phi; GeGLU for Gemma
                config.KvHeads,
                config.ExpertCount,
                config.ExpertUsedCount,
                config.ExpertFeedForwardLength,
                config.NormalizeExpertWeights,
                config.HasSharedExpert,
                headDim: headDim,
                attnLogitSoftcap: config.AttnLogitSoftcap,
                finalLogitSoftcap: config.FinalLogitSoftcap);

            _stackWeights = BuildStackWeights();
        }

        // ── Public API ────────────────────────────────────────────────────────

        /// <summary>
        /// Factory used by GgufLlamaLoader to construct an engine from pre-built TensorStorages.
        /// Exposes the otherwise-private constructor without coupling the loader to Runtime internals.
        /// </summary>
        public static CachedLlamaInferenceEngine CreateFromBuffers(
            GPT1Config config,
            DecodeWeight embedWeights,
            TensorStorage<float> finalNormGamma,
            TensorStorage<float> finalNormBeta,
            DecodeWeight lmHead,
            LayerWeightBuffers[] layers,
            IDisposable? backingFile = null)
        {
            return new CachedLlamaInferenceEngine(
                config, embedWeights, finalNormGamma, finalNormBeta, lmHead, layers, backingFile);
        }

        /// <summary>
        /// Loads a model directly from a GGUF file (no Python conversion needed).
        /// Supports F32, F16, BF16 tensors. Quantized formats (Q4_K_M etc.) throw NotSupportedException.
        /// </summary>
        public static CachedLlamaInferenceEngine LoadGguf(string path, bool mmap = true)
        {
            return GgufLlamaLoader.Load(path, quantize: true, mmap: mmap);
        }

        public GPT1Config Config => _config;

        // ── Interpretability (activation capture + logit lens) ────────────────────
        // Pure-managed tensors mean the residual stream at every layer is directly readable — no PyTorch
        // hooks, no ONNX graph surgery, no FFI. Enable capture, decode/generate as usual, then read each
        // layer's hidden and project it through the head with the logit lens to see the prediction form.

        /// <summary>Turns on/off per-layer residual-stream capture for subsequent decodes. Off by default
        /// (zero hot-path cost). See <see cref="GetLayerActivation"/> / <see cref="LogitLens"/>.</summary>
        public void EnableActivationCapture(bool enabled)
        {
            ThrowIfDisposed();
            _stack.EnableActivationCapture(enabled);
        }

        /// <summary>Copies the captured residual stream after transformer <paramref name="layer"/> (0-based,
        /// pre-final-norm) for the most recent decoded token into <paramref name="destination"/> (length DModel).
        /// Requires <see cref="EnableActivationCapture"/>(true) before the decode.</summary>
        public void GetLayerActivation(int layer, Span<float> destination)
        {
            ThrowIfDisposed();
            _stack.GetLayerActivation(layer, destination);
        }

        /// <summary>Logit lens: projects an intermediate hidden (e.g. from <see cref="GetLayerActivation"/>)
        /// through the final norm + LM head into <paramref name="logits"/> (length VocabSize) — the tokens the
        /// model would predict if it stopped at that depth. At the last layer it equals the real logits.</summary>
        public void LogitLens(ReadOnlySpan<float> layerHidden, Span<float> logits)
        {
            ThrowIfDisposed();
            _stack.LogitLensFromHidden(layerHidden, _stackWeights, logits);
        }

        /// <summary>Loads model weights from an Overfit binary file.</summary>
        public static CachedLlamaInferenceEngine Load(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);
            return Load(br);
        }

        /// <summary>Loads model weights from a BinaryReader.</summary>
#pragma warning disable OVERFIT001 // load-time: weight/config arrays built once per model (engine lifetime), not a per-call path
        public static CachedLlamaInferenceEngine Load(BinaryReader reader)
        {
            // ── Header ────────────────────────────────────────────────────────
            var magic = reader.ReadUInt32();
            if (magic != FilemagicExpected)
            {
                throw new OverfitFormatException(
                $"Not an Overfit SLM file. Expected magic 0x{FilemagicExpected:X8}, got 0x{magic:X8}.");
            }

            var version = reader.ReadInt32();
            if (version != VersionExpected)
            {
                throw new OverfitRuntimeException($"Unsupported file version {version}. Expected {VersionExpected}.");
            }

            var nLayers = reader.ReadInt32();
            var dModel = reader.ReadInt32();
            var nHeads = reader.ReadInt32();
            var nKvHeads = reader.ReadInt32();
            var vocabSize = reader.ReadInt32();
            var contextLength = reader.ReadInt32();
            var dFF = reader.ReadInt32();
            var useRope = reader.ReadInt32() != 0;
            var ropeTheta = reader.ReadSingle();
            var ffnActivation = (FeedForwardActivation)reader.ReadInt32();
            var tieWeights = reader.ReadInt32() != 0;

            var config = new GPT1Config
            {
                NLayers = nLayers,
                DModel = dModel,
                NHeads = nHeads,
                NKvHeads = nKvHeads,
                VocabSize = vocabSize,
                ContextLength = contextLength,
                DFF = dFF,
                UseRoPE = useRope,
                RoPETheta = ropeTheta,
                FfnActivation = ffnActivation,
                TieWeights = tieWeights,
            };

            var headDim = dModel / nHeads;

            // ── Weights ───────────────────────────────────────────────────────
            var embedWeights = ReadTensor(reader, vocabSize * dModel);

            var layers = new LayerWeightBuffers[nLayers];
            for (var l = 0; l < nLayers; l++)
            {
                var attnNormGamma = ReadTensor(reader, dModel);
                var attnNormBeta = ReadTensor(reader, dModel);

                // W matrices are DecodeWeight (the binary path stays F32 — Q8 is
                // GGUF-only; TensorStorage converts implicitly). Biases stay F32.
                var wq = new DecodeWeight[nHeads];
                var bq = new TensorStorage<float>[nHeads];
                for (var h = 0; h < nHeads; h++)
                {
                    wq[h] = ReadTensor(reader, dModel * headDim);
                    bq[h] = ReadTensor(reader, headDim);
                }

                var wk = new DecodeWeight[nKvHeads];
                var bk = new TensorStorage<float>[nKvHeads];
                var wv = new DecodeWeight[nKvHeads];
                var bv = new TensorStorage<float>[nKvHeads];
                for (var kv = 0; kv < nKvHeads; kv++)
                {
                    wk[kv] = ReadTensor(reader, dModel * headDim);
                    bk[kv] = ReadTensor(reader, headDim);
                    wv[kv] = ReadTensor(reader, dModel * headDim);
                    bv[kv] = ReadTensor(reader, headDim);
                }

                var wo = new DecodeWeight[nHeads];
                var bo = new TensorStorage<float>[nHeads];
                for (var h = 0; h < nHeads; h++)
                {
                    wo[h] = ReadTensor(reader, headDim * dModel);
                    bo[h] = ReadTensor(reader, dModel);
                }

                var ffnNormGamma = ReadTensor(reader, dModel);
                var ffnNormBeta = ReadTensor(reader, dModel);
                var ffnGate = ReadTensor(reader, dModel * dFF);
                var ffnUp = ReadTensor(reader, dModel * dFF);
                var ffnDown = ReadTensor(reader, dFF * dModel);

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

            var finalNormGamma = ReadTensor(reader, dModel);
            var finalNormBeta = ReadTensor(reader, dModel);

            // LM head in file: [vocabSize, dModel] (same layout as embedding, row = token)
            // Kernel Project needs: [dModel, vocabSize] (input-major: row = hidden dim)
            // Transpose at load time — happens once, no effect on inference speed.
            var lmHeadRaw = ReadTensor(reader, vocabSize * dModel);
            var lmHead = TransposeLmHead(lmHeadRaw, vocabSize, dModel);
#pragma warning disable IDISP017 // Deliberate early dispose, NOT `using`: free the raw [vocab×dModel] buffer
            // right after the transpose — before constructing the engine — to keep the load-time peak at ~1×
            // (a `using` would hold it alive until method end). See the peak-vs-steady RAM discipline.
            lmHeadRaw.Dispose();
#pragma warning restore IDISP017

            return new CachedLlamaInferenceEngine(config, embedWeights, finalNormGamma, finalNormBeta, lmHead, layers);
        }
#pragma warning restore OVERFIT001

        /// <summary>
        /// Creates an inference session. The caller owns the session and must dispose it.
        /// <paramref name="kvCacheDType"/> selects the KV-cache element type: <see cref="KvCacheDType.F32"/>
        /// (default, full precision) or <see cref="KvCacheDType.Q8"/> (per-vector int8 — ~4× less KV RAM and
        /// attention read traffic, for long-context / low-memory decode). When null it falls back to the
        /// <c>OVERFIT_KV_DTYPE</c> env var (<c>q8</c> → Q8, anything else → F32).
        /// </summary>
        public CachedLlamaSession CreateSession(int? maxContextLength = null, KvCacheDType? kvCacheDType = null)
        {
            ThrowIfDisposed();

            var ctx = maxContextLength ?? _config.ContextLength;
            var headDim = _config.AttentionHeadDim;
            var dtype = kvCacheDType ?? ResolveKvDtypeFromEnv();

            var cache = KeyValueCache.Create(
                _config.NLayers,
                kvHeadCount: _config.KvHeads,
                ctx,
                headDim,
                dtype);

            return new CachedLlamaSession(
                _config,
                _stack,
                _stackWeights,
                cache,
                _rope,
                _embedWeights);
        }

        /// <summary>Reads the opt-in <c>OVERFIT_KV_DTYPE</c> env var: <c>q8</c> → <see cref="KvCacheDType.Q8"/>,
        /// otherwise <see cref="KvCacheDType.F32"/> (the default — no behaviour change unless explicitly set).</summary>
        private static KvCacheDType ResolveKvDtypeFromEnv()
        {
            var raw = Environment.GetEnvironmentVariable(OverfitEnvironment.KvDType);
            return string.Equals(raw, "q8", StringComparison.OrdinalIgnoreCase) ? KvCacheDType.Q8 : KvCacheDType.F32;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            _embedWeights.Dispose();
            _finalNormGamma.Dispose();
            _finalNormBeta.Dispose();
            _lmHead.Dispose();

            foreach (var layer in _layers)
            {
                layer.AttnNormGamma.Dispose();
                layer.AttnNormBeta.Dispose();
                layer.FfnNormGamma.Dispose();
                layer.FfnNormBeta.Dispose();
                layer.FfnGate.Dispose();
                layer.FfnUp.Dispose();
                layer.FfnDown.Dispose();

                if (layer.MoeGate is not null)
                {
                    foreach (var w in layer.MoeGate)
                    {
                        w.Dispose();
                    }
                    foreach (var w in layer.MoeUp!)
                    {
                        w.Dispose();
                    }
                    foreach (var w in layer.MoeDown!)
                    {
                        w.Dispose();
                    }
                    layer.MoeSharedGate.Dispose();
                    layer.MoeSharedUp.Dispose();
                    layer.MoeSharedDown.Dispose();
                }

                foreach (var t in layer.Wq)
                {
                    t.Dispose();
                }
                foreach (var t in layer.Bq)
                {
                    t.Dispose();
                }
                foreach (var t in layer.Wk)
                {
                    t.Dispose();
                }
                foreach (var t in layer.Bk)
                {
                    t.Dispose();
                }
                foreach (var t in layer.Wv)
                {
                    t.Dispose();
                }
                foreach (var t in layer.Bv)
                {
                    t.Dispose();
                }
                foreach (var t in layer.Wo)
                {
                    t.Dispose();
                }
                foreach (var t in layer.Bo)
                {
                    t.Dispose();
                }
            }

            // Dispose the memory map LAST: every mmap-resident weight above holds a
            // ReadOnlyMemory slice into it, so the pages must stay valid until they're gone.
            _backingFile?.Dispose();
        }

        // ── Private helpers ───────────────────────────────────────────────────

#pragma warning disable OVERFIT001 // load-time: per-layer weight handle tables built once per engine
        private StackWeights BuildStackWeights()
        {
            var kvCount = _config.KvHeads;
            var useGqa = kvCount < _config.NHeads;

            var blockWeights = new BlockWeights[_config.NLayers];

            for (var l = 0; l < _config.NLayers; l++)
            {
                var layer = _layers[l];

                // Q heads (each has Wq, bq, Wo, bo)
                var heads = new SingleHeadWeights[_config.NHeads];
                for (var h = 0; h < _config.NHeads; h++)
                {
                    if (useGqa)
                    {
                        // GQA: single head has only Wq, bq, Wo — K/V in separate KvHeadWeights
                        heads[h] = new SingleHeadWeights(
                            wq: layer.Wq[h], bq: layer.Bq[h], wo: layer.Wo[h]);
                    }
                    else
                    {
                        // MHA: all weights in SingleHeadWeights
                        var kv = h % kvCount;
                        // Use TensorStorage-based constructor (positional: wq, bq, wo, wk?, bk?, wv?, bv?)
                        heads[h] = new SingleHeadWeights(
                            wq: layer.Wq[h], bq: layer.Bq[h], wo: layer.Wo[h],
                            wk: layer.Wk[kv], bk: layer.Bk[kv], wv: layer.Wv[kv], bv: layer.Bv[kv]);
                    }
                }

                KvHeadWeights[]? kvHeads = null;
                if (useGqa)
                {
                    kvHeads = new KvHeadWeights[kvCount];
                    for (var kv = 0; kv < kvCount; kv++)
                    {
                        kvHeads[kv] = new KvHeadWeights(
                        wk: layer.Wk[kv], wv: layer.Wv[kv],
                        bk: layer.Bk[kv], bv: layer.Bv[kv]);
                    }
                }

                // Zero-copy: pass TensorStorage references directly (no float[] duplication).
                // This avoids doubling the FFN/attention weights (~10 GB for 3B FP32).
                blockWeights[l] = new BlockWeights(
                    heads: heads,
                    kvHeads: kvHeads,
                    qNorm: layer.QNorm,
                    kNorm: layer.KNorm,
                    ln1Gamma: layer.AttnNormGamma,
                    ln1Beta: null,                  // RMSNorm — no beta
                    attentionBias: null,
                    ln2Gamma: layer.FfnNormGamma,
                    ln2Beta: null,                  // RMSNorm — no beta
                    ffnW1: layer.FfnUp,
                    ffnB1: null,
                    ffnW2: layer.FfnDown,
                    ffnB2: null,
                    ffnGate: layer.FfnGate,
                    moeRouter: layer.MoeRouter,
                    moeGate: layer.MoeGate,
                    moeUp: layer.MoeUp,
                    moeDown: layer.MoeDown,
                    moeSharedGate: layer.MoeSharedGate,
                    moeSharedUp: layer.MoeSharedUp,
                    moeSharedDown: layer.MoeSharedDown,
                    moeSharedGateInp: layer.MoeSharedGateInp,
                    postAttnNorm: layer.PostAttnNorm,
                    postFfwNorm: layer.PostFfwNorm,
                    wqWhole: layer.WqWhole,
                    wkWhole: layer.WkWhole,
                    wvWhole: layer.WvWhole,
                    woWhole: layer.WoWhole);
            }

            return new StackWeights(
                blockWeights,
                _finalNormGamma,
                TensorStorage<float>.Unpooled(0),  // RMSNorm — no final norm beta
                _lmHead);
        }
#pragma warning restore OVERFIT001

        /// <summary>
        /// Transposes LM head from file layout [vocabSize, dModel] to kernel layout [dModel, vocabSize].
        /// The Project kernel computes: output[v] += hidden[d] * weights[d * vocabSize + v]
        /// So weights must be stored as [dModel, vocabSize].
        /// </summary>
        private static TensorStorage<float> TransposeLmHead(
            TensorStorage<float> raw, int vocabSize, int dModel)
        {
            var result = TensorStorage<float>.Unpooled(vocabSize * dModel);
            var src = raw.AsReadOnlySpan();
            var dst = result.AsSpan();

            // src[v * dModel + d] = E[v, d]  (row = token)
            // dst[d * vocabSize + v] = E[v, d]  (row = hidden dim)
            for (var d = 0; d < dModel; d++)
            {
                for (var v = 0; v < vocabSize; v++)
                {
                    dst[d * vocabSize + v] = src[v * dModel + d];
                }
            }

            return result;
        }

        /// <summary>
        /// F32 backing of a weight, or a clear failure if it is Q8-resident.
        /// LoRA and the F32-only diagnostics need an F32 <see cref="TensorStorage{T}"/>.
        /// </summary>
        private static TensorStorage<float> RequireF32(in DecodeWeight weight, string context)
            => weight.F32Storage ?? throw new OverfitRuntimeException(
                $"{context} requires F32-resident weights; this model was loaded with Q8_0 " +
                "quantization. Operations on quantized weights are not supported.");

        /// <summary>Reads a float tensor directly into a new TensorStorage.</summary>
        private static TensorStorage<float> ReadTensor(BinaryReader reader, int count)
        {
            // Unpooled: model weights are long-lived. Pool buckets round to pow2 and
            // retain rented arrays after Dispose — wasteful for multi-GB tensors.
            var storage = TensorStorage<float>.Unpooled(count);
            // Stream bytes straight into the destination span — no scratch byte[]
            // allocation. Peak load RAM stays at file size instead of 2× file size.
            // Little-endian-only (same constraint as the previous MemoryMarshal.Cast path).
            var dst = MemoryMarshal.AsBytes(storage.AsSpan());
            reader.BaseStream.ReadExactly(dst);
            return storage;
        }


        // ── DIAGNOSTIC ────────────────────────────────────────────────────────

        /// <summary>
        /// Reads the L2 norm of Q weight via the SAME path inference uses
        /// (through _stackWeights._blocks[l]._heads[h]._wq).
        /// If this differs from LlamaLoRAAdapter.ReadBaseWeightNorm, then
        /// _baseRefs and _stackWeights point to different TensorStorage objects.
        /// </summary>
        /// <summary>Diagnostic hook (M2): true when block <paramref name="layer"/> carries ALL FOUR whole-matrix
        /// Q4_K attention handles. Note Q4_K_M is a MIXED quant (V / O are usually Q6_K) so this is often false on
        /// a real model even when Q/K are present — use <see cref="BlockWholeAttnPresence"/> for per-projection.</summary>
        internal bool BlockHasWholeAttnQ4K(int layer)
        {
            ThrowIfDisposed();
            return _stackWeights.Block(layer).HasWholeAttnQ4K;
        }

        /// <summary>Diagnostic hook (M2): per-projection presence of the whole-matrix Q4_K attention handles
        /// (q, k, v, o) for block <paramref name="layer"/> — each true when that projection was Q4_K + mmap +
        /// repackable. M3 applies the repacked GEMV per-projection (a Q6_K V/O keeps the per-head path).</summary>
        internal (bool Q, bool K, bool V, bool O) BlockWholeAttnPresence(int layer)
        {
            ThrowIfDisposed();
            ref readonly var b = ref _stackWeights.Block(layer);
            return (b.WqWhole.IsQ4K, b.WkWhole.IsQ4K, b.WvWhole.IsQ4K, b.WoWhole.IsQ4K);
        }

        public float ReadInferenceWeightNorm(int layer, int head)
        {
            ThrowIfDisposed();
            var hw = _stackWeights.Block(layer).Head(head);
            var span = RequireF32(hw.Wq, "ReadInferenceWeightNorm").AsReadOnlySpan();
            var sumSq = 0f;
            foreach (var v in span)
            {
                sumSq += v * v;
            }

            return MathF.Sqrt(sumSq);
        }

        /// <summary>
        /// Writes a value directly to Wq[layer, head, index] via the SAME storage
        /// inference uses. If subsequent inference doesn't see this change,
        /// the bug is fundamental in the engine (separate copy somewhere).
        /// </summary>
        public void WriteInferenceWeight(int layer, int head, int index, float value)
        {
            ThrowIfDisposed();
            // _layers[l].Wq[h] is the SAME storage inference reads (zero-copy).
            var span = RequireF32(_layers[layer].Wq[head], "WriteInferenceWeight").AsSpan();
            if (index >= 0 && index < span.Length)
            {
                span[index] = value;
            }
        }

        /// <summary>Number of transformer layers — for the QLoRA training bridge
        /// (<see cref="DevOnBike.Overfit.DeepLearning.TrainableLlamaBlock"/>).</summary>
        internal int TrainableLayerCount
        {
            get
            {
                ThrowIfDisposed();
                return _layers.Length;
            }
        }

        /// <summary>The frozen quantized weights of layer <paramref name="layer"/>, exposed for the
        /// QLoRA training bridge. The returned <see cref="LayerWeightBuffers"/> holds the SAME
        /// (zero-copy) <see cref="DecodeWeight"/> handles inference reads — feed Wq/Wk/Wv/Wo through
        /// <c>ConcatRowsDequantSource</c>/<c>ConcatColsDequantSource</c> and gate/up/down
        /// via <see cref="DecodeWeight.AsRowSource"/> into a trainable block.</summary>
        internal LayerWeightBuffers GetTrainableLayer(int layer)
        {
            ThrowIfDisposed();
            return _layers[layer];
        }

        /// <summary>Frozen token-embedding table <c>[vocab, dModel]</c> (row = token) for the QLoRA
        /// training bridge — look rows up via <see cref="DecodeWeight.DequantizeRow"/>.</summary>
        internal DecodeWeight EmbeddingWeights
        {
            get
            {
                ThrowIfDisposed();
                return _embedWeights;
            }
        }

        /// <summary>Trainable final-RMSNorm gain <c>[dModel]</c> (F32) for the QLoRA training bridge.</summary>
        internal TensorStorage<float> FinalNormGamma
        {
            get
            {
                ThrowIfDisposed();
                return _finalNormGamma;
            }
        }

        /// <summary>Frozen LM-head <c>[vocab, dModel]</c> (separate handle even when tied) for the
        /// QLoRA training bridge — feed via <see cref="DecodeWeight.AsRowSource"/>.</summary>
        internal DecodeWeight LmHeadWeights
        {
            get
            {
                ThrowIfDisposed();
                return _lmHead;
            }
        }

        /// <summary>Read Wq via _layers path (the path LoRA modifies).</summary>
        public float ReadLayerWeight(int layer, int head, int index)
        {
            ThrowIfDisposed();
            var span = RequireF32(_layers[layer].Wq[head], "ReadLayerWeight").AsSpan();
            if (index < 0 || index >= span.Length)
            {
                return float.NaN;
            }

            return span[index];
        }

        /// <summary>True if _layers[l].Wq[h] is the SAME OBJECT as _stackWeights._blocks[l]._heads[h]._wq.</summary>
        public bool AreSameStorage(int layer, int head)
        {
            ThrowIfDisposed();
            // Write a unique value via the _layers path, read it back via the
            // inference (_stackWeights) path — if it surfaces, both alias one storage.
            var span = RequireF32(_layers[layer].Wq[head], "AreSameStorage").AsSpan();
            var saved = span[0];
            span[0] = 12345.6789f;
            var infValue = ReadInferenceWeightAt(layer, head, 0);
            span[0] = saved;  // restore
            return MathF.Abs(infValue - 12345.6789f) < 1e-4f;
        }

        /// <summary>Reads Wq[0] via inference path.</summary>
        public float ReadInferenceWeightAt(int layer, int head, int index)
        {
            ThrowIfDisposed();
            var hw = _stackWeights.Block(layer).Head(head);
            var span = RequireF32(hw.Wq, "ReadInferenceWeightAt").AsReadOnlySpan();
            return index < 0 || index >= span.Length ? float.NaN : span[index];
        }

        // ── LoRA ──────────────────────────────────────────────────────────────

        /// <summary>
        /// Creates a LoRA adapter targeting this engine's weight matrices.
        ///
        /// The adapter holds TensorStorage references — zero weight copying.
        /// Enable()  → W_eff = W_base + scale*(A@B)  (in-place, zero inference overhead)
        /// Disable() → W_base restored exactly
        ///
        /// Example:
        ///   var opts = new LoRAOptions(rank: 16, alpha: 32f, dropout: 0f,
        ///                             LoRATargetModules.Attention);
        ///   using var adapter = engine.CreateLoRAAdapter("v1", opts);
        ///   adapter.Load("v1.lora");
        ///   adapter.Enable();
        ///   session.GenerateNextToken(in sampling);
        ///   adapter.Disable();
        /// </summary>
        public LlamaLoRAAdapter CreateLoRAAdapter(string name, in LoRAOptions options)
        {
            ThrowIfDisposed();

            var headDim = _config.AttentionHeadDim;
            var refs = new Dictionary<(int, LoRATargetModules, int),
                                      TensorStorage<float>>();

            for (var l = 0; l < _config.NLayers; l++)
            {
                var layer = _layers[l];

                for (var h = 0; h < _config.NHeads; h++)
                {
                    refs[(l, LoRATargetModules.Query, h)] = RequireF32(layer.Wq[h], "LoRA on attention");
                    refs[(l, LoRATargetModules.OutputProjection, h)] = RequireF32(layer.Wo[h], "LoRA on attention");
                }

                for (var kv = 0; kv < _config.KvHeads; kv++)
                {
                    refs[(l, LoRATargetModules.Key, kv)] = RequireF32(layer.Wk[kv], "LoRA on attention");
                    refs[(l, LoRATargetModules.Value, kv)] = RequireF32(layer.Wv[kv], "LoRA on attention");
                }

                refs[(l, LoRATargetModules.FeedForwardUp, 0)] = RequireF32(layer.FfnUp, "LoRA on FeedForward");
                refs[(l, LoRATargetModules.FeedForwardDown, 0)] = RequireF32(layer.FfnDown, "LoRA on FeedForward");
            }

            return new LlamaLoRAAdapter(
                name, options,
                _config.NLayers, _config.NHeads, _config.KvHeads,
                _config.DModel, headDim, _config.DFF,
                refs);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CachedLlamaInferenceEngine));
            }
        }
    }
}
