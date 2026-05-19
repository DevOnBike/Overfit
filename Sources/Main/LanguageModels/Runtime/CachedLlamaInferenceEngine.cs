// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Rope;
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
        private readonly TensorStorage<float> _embedWeights;
        private readonly TensorStorage<float> _finalNormGamma;
        private readonly TensorStorage<float> _finalNormBeta;
        private readonly DecodeWeight _lmHead;
        private readonly LayerWeightBuffers[] _layers;
        private readonly StackWeights _stackWeights;

        private bool _disposed;

        // ── Internal weight container per layer ───────────────────────────────

        public sealed class LayerWeightBuffers
        {
            public required TensorStorage<float> AttnNormGamma;
            public required TensorStorage<float> AttnNormBeta;
            public required TensorStorage<float>[] Wq;
            public required TensorStorage<float>[] Bq;
            public required TensorStorage<float>[] Wk;
            public required TensorStorage<float>[] Bk;
            public required TensorStorage<float>[] Wv;
            public required TensorStorage<float>[] Bv;
            public required TensorStorage<float>[] Wo;
            public required TensorStorage<float>[] Bo;
            public required TensorStorage<float> FfnNormGamma;
            public required TensorStorage<float> FfnNormBeta;
            public required TensorStorage<float> FfnGate;
            public required TensorStorage<float> FfnUp;
            public required TensorStorage<float> FfnDown;
        }

        // ── Constructor ───────────────────────────────────────────────────────

        private CachedLlamaInferenceEngine(
            GPT1Config config,
            TensorStorage<float> embedWeights,
            TensorStorage<float> finalNormGamma,
            TensorStorage<float> finalNormBeta,
            DecodeWeight lmHead,
            LayerWeightBuffers[] layers)
        {
            _config = config;
            _embedWeights = embedWeights;
            _finalNormGamma = finalNormGamma;
            _finalNormBeta = finalNormBeta;
            _lmHead = lmHead;
            _layers = layers;

            // Build RoPE table if required
            _rope = config.UseRoPE
                ? new RopeTable(config.ContextLength, config.DModel / config.NHeads, config.RoPETheta)
                : null;

            var headDim = config.DModel / config.NHeads;

            _stack = new CachedGptStack(
                config.NLayers,
                config.DModel,
                config.NHeads,
                config.DFF,
                config.VocabSize,
                config.ContextLength,
                layerNormEpsilon: 1e-6f,
                FeedForwardActivation.SwiGLU,
                config.KvHeads);

            _stackWeights = BuildStackWeights();
        }

        // ── Public API ────────────────────────────────────────────────────────

        /// <summary>
        /// Factory used by GgufLlamaLoader to construct an engine from pre-built TensorStorages.
        /// Exposes the otherwise-private constructor without coupling the loader to Runtime internals.
        /// </summary>
        public static CachedLlamaInferenceEngine CreateFromBuffers(
            GPT1Config config,
            TensorStorage<float> embedWeights,
            TensorStorage<float> finalNormGamma,
            TensorStorage<float> finalNormBeta,
            DecodeWeight lmHead,
            LayerWeightBuffers[] layers)
        {
            return new CachedLlamaInferenceEngine(
                config, embedWeights, finalNormGamma, finalNormBeta, lmHead, layers);
        }

        /// <summary>
        /// Loads a model directly from a GGUF file (no Python conversion needed).
        /// Supports F32, F16, BF16 tensors. Quantized formats (Q4_K_M etc.) throw NotSupportedException.
        /// </summary>
        public static CachedLlamaInferenceEngine LoadGguf(string path)
        {
            return GgufLlamaLoader.Load(path);
        }

        public GPT1Config Config => _config;

        /// <summary>Loads model weights from an Overfit binary file.</summary>
        public static CachedLlamaInferenceEngine Load(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);
            return Load(br);
        }

        /// <summary>Loads model weights from a BinaryReader.</summary>
        public static CachedLlamaInferenceEngine Load(BinaryReader reader)
        {
            // ── Header ────────────────────────────────────────────────────────
            var magic = reader.ReadUInt32();
            if (magic != FilemagicExpected)
            {
                throw new InvalidDataException(
                $"Not an Overfit SLM file. Expected magic 0x{FilemagicExpected:X8}, got 0x{magic:X8}.");
            }

            var version = reader.ReadInt32();
            if (version != VersionExpected)
            {
                throw new NotSupportedException($"Unsupported file version {version}. Expected {VersionExpected}.");
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

                var wq = new TensorStorage<float>[nHeads];
                var bq = new TensorStorage<float>[nHeads];
                for (var h = 0; h < nHeads; h++)
                {
                    wq[h] = ReadTensor(reader, dModel * headDim);
                    bq[h] = ReadTensor(reader, headDim);
                }

                var wk = new TensorStorage<float>[nKvHeads];
                var bk = new TensorStorage<float>[nKvHeads];
                var wv = new TensorStorage<float>[nKvHeads];
                var bv = new TensorStorage<float>[nKvHeads];
                for (var kv = 0; kv < nKvHeads; kv++)
                {
                    wk[kv] = ReadTensor(reader, dModel * headDim);
                    bk[kv] = ReadTensor(reader, headDim);
                    wv[kv] = ReadTensor(reader, dModel * headDim);
                    bv[kv] = ReadTensor(reader, headDim);
                }

                var wo = new TensorStorage<float>[nHeads];
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
            lmHeadRaw.Dispose();

            return new CachedLlamaInferenceEngine(config, embedWeights, finalNormGamma, finalNormBeta, lmHead, layers);
        }

        /// <summary>Creates an inference session. The caller owns the session and must dispose it.</summary>
        public CachedLlamaSession CreateSession(int? maxContextLength = null)
        {
            ThrowIfDisposed();

            var ctx = maxContextLength ?? _config.ContextLength;
            var headDim = _config.DModel / _config.NHeads;

            var cache = KeyValueCache.Create(
                _config.NLayers,
                kvHeadCount: _config.KvHeads,
                ctx,
                headDim);

            return new CachedLlamaSession(
                _config,
                _stack,
                _stackWeights,
                cache,
                _rope,
                _embedWeights.AsReadOnlySpan());
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
        }

        // ── Private helpers ───────────────────────────────────────────────────

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
                    ln1Gamma: layer.AttnNormGamma,
                    ln1Beta: null,                  // RMSNorm — no beta
                    attentionBias: null,
                    ln2Gamma: layer.FfnNormGamma,
                    ln2Beta: null,                  // RMSNorm — no beta
                    ffnW1: layer.FfnUp,
                    ffnB1: null,
                    ffnW2: layer.FfnDown,
                    ffnB2: null,
                    ffnGate: layer.FfnGate);
            }

            return new StackWeights(
                blockWeights,
                _finalNormGamma,
                TensorStorage<float>.Unpooled(0),  // RMSNorm — no final norm beta
                _lmHead);
        }

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
        public float ReadInferenceWeightNorm(int layer, int head)
        {
            ThrowIfDisposed();
            var hw = _stackWeights.Block(layer).Head(head);
            var span = hw.Wq;
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
            var hw = _stackWeights.Block(layer).Head(head);
            // Get mutable span via the underlying TensorStorage
            // Since hw.Wq returns ReadOnlySpan, we need to access _wq directly.
            // _layers[l].Wq[h] is the SAME TensorStorage as hw._wq (zero-copy).
            var span = _layers[layer].Wq[head].AsSpan();
            if (index >= 0 && index < span.Length)
            {
                span[index] = value;
            }
        }

        /// <summary>Read Wq via _layers path (the path LoRA modifies).</summary>
        public float ReadLayerWeight(int layer, int head, int index)
        {
            ThrowIfDisposed();
            var span = _layers[layer].Wq[head].AsSpan();
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
            var fromLayers = _layers[layer].Wq[head];
            var hw = _stackWeights.Block(layer).Head(head);
            // We can't compare _wq directly (private) but we can use a trick:
            // write a unique value to fromLayers, read via inference path, and check
            var span = fromLayers.AsSpan();
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
            var span = hw.Wq;
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

            var headDim = _config.DModel / _config.NHeads;
            var refs = new Dictionary<(int, LoRATargetModules, int),
                                      TensorStorage<float>>();

            for (var l = 0; l < _config.NLayers; l++)
            {
                var layer = _layers[l];

                for (var h = 0; h < _config.NHeads; h++)
                {
                    refs[(l, LoRATargetModules.Query, h)] = layer.Wq[h];
                    refs[(l, LoRATargetModules.OutputProjection, h)] = layer.Wo[h];
                }

                for (var kv = 0; kv < _config.KvHeads; kv++)
                {
                    refs[(l, LoRATargetModules.Key, kv)] = layer.Wk[kv];
                    refs[(l, LoRATargetModules.Value, kv)] = layer.Wv[kv];
                }

                refs[(l, LoRATargetModules.FeedForwardUp, 0)] = layer.FfnUp;
                refs[(l, LoRATargetModules.FeedForwardDown, 0)] = layer.FfnDown;
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
