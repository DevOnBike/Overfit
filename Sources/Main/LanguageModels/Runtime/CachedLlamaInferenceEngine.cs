// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
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
        private readonly TensorStorage<float> _lmHead;
        private readonly LayerWeightBuffers[] _layers;
        private readonly StackWeights _stackWeights;

        private bool _disposed;

        // ── Internal weight container per layer ───────────────────────────────

        private sealed class LayerWeightBuffers
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
            TensorStorage<float> lmHead,
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

                // Convert TensorStorage → float[] for BlockWeights constructor.
                // This happens once at load time, not during inference.
                static float[] Arr(TensorStorage<float> s) => s.AsSpan().ToArray();

                blockWeights[l] = new BlockWeights(
                    heads: heads,
                    kvHeads: kvHeads,
                    ln1Gamma: Arr(layer.AttnNormGamma),
                    ln1Beta: null,  // RMSNorm — no beta
                    attentionBias: null,
                    ln2Gamma: Arr(layer.FfnNormGamma),
                    ln2Beta: null,  // RMSNorm — no beta
                    ffnW1: Arr(layer.FfnUp),
                    ffnB1: null,
                    ffnW2: Arr(layer.FfnDown),
                    ffnB2: null,
                    ffnGate: Arr(layer.FfnGate));
            }

            return new StackWeights(
                blockWeights,
                _finalNormGamma,
                new TensorStorage<float>(0),  // RMSNorm — no final norm beta
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
            var result = new TensorStorage<float>(vocabSize * dModel);
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
            var storage = new TensorStorage<float>(count);
            var span = storage.AsSpan();
            var bytes = new byte[count * 4];
            reader.Read(bytes, 0, bytes.Length);
            System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes).CopyTo(span);
            return storage;
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
                                      Tensors.Core.TensorStorage<float>>();

            for (var l = 0; l < _config.NLayers; l++)
            {
                var layer = _layers[l];

                for (var h = 0; h < _config.NHeads; h++)
                {
                    refs[(l, LoRATargetModules.Query,            h)] = layer.Wq[h];
                    refs[(l, LoRATargetModules.OutputProjection, h)] = layer.Wo[h];
                }

                for (var kv = 0; kv < _config.KvHeads; kv++)
                {
                    refs[(l, LoRATargetModules.Key,   kv)] = layer.Wk[kv];
                    refs[(l, LoRATargetModules.Value, kv)] = layer.Wv[kv];
                }

                refs[(l, LoRATargetModules.FeedForwardUp,   0)] = layer.FfnUp;
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
