// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// LoRA fine-tuning for a <see cref="GPT1Model"/>. The base model is frozen;
    /// only low-rank factors are trained, one A/B pair per targeted weight matrix:
    ///   W_eff = W(frozen) + (A @ B)
    ///   A : [inDim x rank]   (Kaiming-uniform init)
    ///   B : [rank x outDim]  (zero init — adapter starts as the identity)
    ///
    /// Targets are selected with <see cref="LoRATargetModules"/>:
    ///   <see cref="LoRATargetModules.LanguageModelHead"/> — LM head [dModel x vocab]   (Stage 1)
    ///   <see cref="LoRATargetModules.FeedForwardUp"/>      — each block's FFN W1 [dModel x dFF]  (Stage 2)
    ///   <see cref="LoRATargetModules.FeedForwardDown"/>    — each block's FFN W2 [dFF x dModel]  (Stage 2)
    ///   <see cref="LoRATargetModules.FeedForward"/>        — both FFN matrices, every block
    ///   <see cref="LoRATargetModules.Query"/> / Key / Value — per-head attention W [dModel x dHead]  (Stage 3)
    ///   <see cref="LoRATargetModules.OutputProjection"/>    — per-head attention Wo [dHead x dModel]  (Stage 3)
    ///   <see cref="LoRATargetModules.Attention"/>           — all four, every head, every block
    /// Per-head attention targets create one A/B pair per head per block (the
    /// GPT1 attention layer stores Q/K/V/O as separate per-head matrices).
    ///
    /// A and B become <see cref="Parameter"/>s on the autograd graph, so
    /// <c>ComputationGraph.Backward</c> produces their gradients automatically; the
    /// optimizer only ever sees the {A, B} pairs, so base weights are never updated.
    ///
    /// While a fine-tuner is attached, effective weights are injected per-forward
    /// via <c>GPT1Model.LMHeadWeightProvider</c> and
    /// <c>FeedForwardLayer.W1/W2WeightProvider</c>; <see cref="Dispose"/> detaches
    /// every hook. Exported adapters use the multi-entry LoRA .bin format
    /// (<see cref="Gpt1LoRAFile"/>) consumed by <see cref="Gpt1LoRAMergeAdapter"/>.
    /// </summary>
    public sealed class Gpt1LoRAFineTuner : IDisposable
    {
        private const LoRATargetModules SupportedTargets =
            LoRATargetModules.LanguageModelHead
            | LoRATargetModules.FeedForwardUp
            | LoRATargetModules.FeedForwardDown
            | LoRATargetModules.Query
            | LoRATargetModules.Key
            | LoRATargetModules.Value
            | LoRATargetModules.OutputProjection;

        private const int LMHeadLayer = -1;

        private readonly GPT1Model _model;
        private readonly int _dModel;
        private readonly int _dFF;
        private readonly int _vocab;
        private readonly int _nLayers;
        private readonly int _rank;
        private readonly LoRATargetModules _targets;
        private readonly bool _quantizeBase;
        private readonly QLoRABaseFormat _baseFormat;
        private readonly ModuleAdapter[] _adapters;

        private bool _disposed;

        /// <param name="quantizeBase">
        /// QLoRA mode: freeze the base as Q4_K (dequantized on the fly, never updated) instead of
        /// keeping it F32 — ~7× less base RAM in training, plus no base-grad buffer. Implemented for
        /// the <see cref="LoRATargetModules.LanguageModelHead"/> target only (the per-deployment
        /// default); other targets still need their own output-level hooks.
        /// </param>
        public Gpt1LoRAFineTuner(
            GPT1Model model,
            int rank,
            LoRATargetModules targets = LoRATargetModules.LanguageModelHead,
            int seed = 42,
            bool quantizeBase = false,
            QLoRABaseFormat baseFormat = QLoRABaseFormat.Q4K)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);

            if (targets == LoRATargetModules.None)
            {
                throw new ArgumentException("At least one target module must be selected.", nameof(targets));
            }

            const LoRATargetModules qLoRASupported =
                LoRATargetModules.LanguageModelHead | LoRATargetModules.FeedForwardUp | LoRATargetModules.FeedForwardDown;

            if (quantizeBase && (targets & ~qLoRASupported) != 0)
            {
                throw new NotSupportedException(
                    "QLoRA base quantization (quantizeBase) supports LanguageModelHead + FeedForwardUp/Down. " +
                    "Per-head attention is blocked: headDim (= dModel/nHeads) < 256, and Q4_K needs 256-element super-blocks.");
            }

            if (quantizeBase)
            {
                // Q4_K needs 256-element super-blocks; Q8_0 needs only 32-element blocks.
                var mult = baseFormat == QLoRABaseFormat.Q8 ? 32 : 256;

                if (model.Config.DModel % mult != 0)
                {
                    throw new NotSupportedException(
                        $"QLoRA {baseFormat} base requires DModel ({model.Config.DModel}) to be a multiple of {mult}.");
                }

                if (targets.HasFlag(LoRATargetModules.FeedForwardDown) && model.Config.DFF % mult != 0)
                {
                    throw new NotSupportedException(
                        $"QLoRA {baseFormat} FFN-down base requires DFF ({model.Config.DFF}) to be a multiple of {mult}.");
                }
            }

            if ((targets & ~SupportedTargets) != 0)
            {
                throw new NotSupportedException(
                    $"Gpt1LoRAFineTuner supports {SupportedTargets}; attention modules are not implemented.");
            }

            if (targets.HasFlag(LoRATargetModules.LanguageModelHead) && model.Config.TieWeights)
            {
                throw new NotSupportedException(
                    "LoRA on the LM head requires an untied head; the model must be built with TieWeights=false.");
            }

            _model = model;
            _dModel = model.Config.DModel;
            _dFF = model.Config.DFF;
            _vocab = model.Config.VocabSize;
            _nLayers = model.Config.NLayers;
            _rank = rank;
            _targets = targets;
            _quantizeBase = quantizeBase;
            _baseFormat = baseFormat;

            _adapters = BuildAdapters(seed);
            AttachProviders();
        }

        public int Rank => _rank;

        public LoRATargetModules Targets => _targets;

        /// <summary>Number of A/B adapter pairs (one per targeted weight matrix).</summary>
        public int AdapterCount => _adapters.Length;

        public long TrainableParameterCount
        {
            get
            {
                var total = 0L;
                foreach (var adapter in _adapters)
                {
                    total += adapter.A.Shape.Size + adapter.B.Shape.Size;
                }

                return total;
            }
        }

        /// <summary>
        /// Fine-tunes every LoRA factor on a flat token corpus (next-token loss).
        /// Returns the per-step training loss for inspection.
        ///
        /// <para>
        /// <b>Rehearsal-lite (optional).</b> When <paramref name="rehearsalCorpus"/>
        /// is supplied, a <paramref name="rehearsalFraction"/> of training windows
        /// are drawn from it instead of the new-task <paramref name="corpus"/>. This
        /// is the cheapest form of replay: mixing a little of the base regime's data
        /// into the adapter's training keeps it from over-forgetting the general
        /// "normal" while it specialises to the new one. Rehearsal-free continual
        /// learning research (arXiv:2406.09384; NeurIPS 2024 "Continual Multimodal
        /// Pretraining") finds that for practical per-deployment adaptation, simple
        /// LoRA fine-tuning with a sane data mixture is competitive with far more
        /// complex continual-learning machinery — so a single mixing fraction is the
        /// right-sized lever here, not a new algorithm.
        /// </para>
        /// </summary>
        public IReadOnlyList<float> FineTune(
            int[] corpus,
            int steps,
            int contextLength,
            float learningRate,
            int seed = 1234,
            int[]? rehearsalCorpus = null,
            float rehearsalFraction = 0f)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(corpus);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(steps);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextLength);

            if (corpus.Length < contextLength + 1)
            {
                throw new ArgumentException(
                    $"Corpus length {corpus.Length} is too short for contextLength {contextLength}.",
                    nameof(corpus));
            }

            if (rehearsalCorpus is not null)
            {
                if (rehearsalFraction is < 0f or >= 1f)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(rehearsalFraction), rehearsalFraction,
                        "Rehearsal fraction must be in [0, 1).");
                }

                if (rehearsalCorpus.Length < contextLength + 1)
                {
                    throw new ArgumentException(
                        $"Rehearsal corpus length {rehearsalCorpus.Length} is too short for contextLength {contextLength}.",
                        nameof(rehearsalCorpus));
                }
            }

            _model.Eval(); // base is frozen — no dropout / train-mode behaviour wanted

            using var graph = new ComputationGraph(EstimateArenaFloats(contextLength));
            using var optimizer = new Adam(CollectParameters(), learningRate) { UseAdamW = true };

            var rng = new Random(seed);
            var input = new int[contextLength];
            var target = new int[contextLength];
            var history = new float[steps];

            for (var step = 0; step < steps; step++)
            {
                // Rehearsal-lite: a fraction of windows come from the base regime so
                // the adapter doesn't forget the general "normal" while specialising.
                var src = (rehearsalCorpus is not null && rng.NextDouble() < rehearsalFraction)
                    ? rehearsalCorpus
                    : corpus;

                var start = rng.Next(0, src.Length - contextLength - 1);
                src.AsSpan(start, contextLength).CopyTo(input);
                src.AsSpan(start + 1, contextLength).CopyTo(target);

                graph.Reset();
                _model.InvalidateAllCaches();

                foreach (var adapter in _adapters)
                {
                    adapter.A.ZeroGrad();
                    adapter.B.ZeroGrad();
                }

                var logits = _model.Forward(graph, input, batchSize: 1, contextLength);
                history[step] = ComputeLossAndGrad(logits, target, _vocab, writeGrad: true);

                graph.BackwardFromGrad(logits);
                optimizer.Step();
            }

            return history;
        }

        /// <summary>
        /// Cross-entropy next-token loss over one window — no gradient, no model
        /// mutation. Uses the currently attached LoRA factors.
        /// </summary>
        public float EvaluateLoss(int[] corpus, int contextLength, int start)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(corpus);

            if (start < 0 || start + contextLength + 1 > corpus.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(start));
            }

            using var graph = new ComputationGraph(EstimateArenaFloats(contextLength));

            _model.Eval();
            graph.Reset();
            _model.InvalidateAllCaches();

            var input = new int[contextLength];
            var target = new int[contextLength];
            corpus.AsSpan(start, contextLength).CopyTo(input);
            corpus.AsSpan(start + 1, contextLength).CopyTo(target);

            var logits = _model.Forward(graph, input, batchSize: 1, contextLength);
            return ComputeLossAndGrad(logits, target, _vocab, writeGrad: false);
        }

        /// <summary>
        /// Saves the trained LoRA factors in the multi-entry LoRA .bin format —
        /// one entry per targeted weight matrix.
        /// </summary>
        public void Save(string path)
        {
            ThrowIfDisposed();

            var entries = new Gpt1LoRAEntry[_adapters.Length];

            for (var i = 0; i < _adapters.Length; i++)
            {
                var adapter = _adapters[i];
                var weight = new LoRAWeight(adapter.InDim, adapter.OutDim, _rank);

                adapter.A.DataReadOnlySpan.CopyTo(weight.AMutable);
                adapter.B.DataReadOnlySpan.CopyTo(weight.BMutable);

                entries[i] = new Gpt1LoRAEntry(adapter.Layer, adapter.Module, adapter.HeadIndex, weight);
            }

            Gpt1LoRAFile.Save(path, entries);
        }

        /// <summary>Loads LoRA factors previously written by <see cref="Save"/>.</summary>
        public void Load(string path)
        {
            ThrowIfDisposed();

            var entries = Gpt1LoRAFile.Load(path);

            foreach (var adapter in _adapters)
            {
                var matched = false;

                foreach (var entry in entries)
                {
                    if (entry.Layer != adapter.Layer
                        || entry.Module != adapter.Module
                        || entry.HeadIndex != adapter.HeadIndex)
                    {
                        continue;
                    }

                    var weight = entry.Weight;
                    if (weight.InDim != adapter.InDim || weight.OutDim != adapter.OutDim || weight.Rank != _rank)
                    {
                        throw new InvalidDataException(
                            $"LoRA entry [{adapter.Module} layer {adapter.Layer} head {adapter.HeadIndex}] dimensions " +
                            $"[{weight.InDim}x{weight.OutDim} r={weight.Rank}] do not match this fine-tuner " +
                            $"[{adapter.InDim}x{adapter.OutDim} r={_rank}].");
                    }

                    weight.A.CopyTo(adapter.A.DataSpan);
                    weight.B.CopyTo(adapter.B.DataSpan);
                    matched = true;
                    break;
                }

                if (!matched)
                {
                    throw new InvalidDataException(
                        $"LoRA file has no entry for [{adapter.Module} layer {adapter.Layer} head {adapter.HeadIndex}].");
                }
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            foreach (var adapter in _adapters)
            {
                DetachProvider(adapter);

                adapter.ANode.Dispose();
                adapter.BNode.Dispose();
                adapter.WBaseNode?.Dispose();
                adapter.BiasNode?.Dispose();
                adapter.A.Dispose();
                adapter.B.Dispose();
            }
        }

        // ── Private ───────────────────────────────────────────────────────────

        private ModuleAdapter[] BuildAdapters(int seed)
        {
            var list = new List<ModuleAdapter>();

            if (_targets.HasFlag(LoRATargetModules.LanguageModelHead))
            {
                list.Add(CreateAdapter(
                    LMHeadLayer, LoRATargetModules.LanguageModelHead,
                    _dModel, _vocab, _model.LMHead, seed));
            }

            for (var layer = 0; layer < _nLayers; layer++)
            {
                var ffn = _model.Blocks[layer].FFN;

                if (_targets.HasFlag(LoRATargetModules.FeedForwardUp))
                {
                    list.Add(CreateAdapter(
                        layer, LoRATargetModules.FeedForwardUp,
                        _dModel, _dFF, ffn.W1, seed + 1 + (2 * layer), bias: ffn.B1));
                }

                if (_targets.HasFlag(LoRATargetModules.FeedForwardDown))
                {
                    list.Add(CreateAdapter(
                        layer, LoRATargetModules.FeedForwardDown,
                        _dFF, _dModel, ffn.W2, seed + 2 + (2 * layer), bias: ffn.B2));
                }
            }

            // Stage 3 — per-head attention Q/K/V/O. GPT1 stores each head's weights
            // separately, so one A/B pair per head per targeted module per block.
            // Q/K/V are [dModel x dHead]; the output projection Wo is [dHead x dModel].
            var attnSeed = seed + 1000;
            for (var layer = 0; layer < _nLayers; layer++)
            {
                var attn = _model.Blocks[layer].Attention;
                var dHead = attn.DHead;

                for (var h = 0; h < attn.NHeads; h++)
                {
                    if (_targets.HasFlag(LoRATargetModules.Query))
                    {
                        list.Add(CreateAdapter(
                            layer, LoRATargetModules.Query, _dModel, dHead,
                            attn.WqHeads[h], attnSeed++, headIndex: h));
                    }

                    if (_targets.HasFlag(LoRATargetModules.Key))
                    {
                        list.Add(CreateAdapter(
                            layer, LoRATargetModules.Key, _dModel, dHead,
                            attn.WkHeads[h], attnSeed++, headIndex: h));
                    }

                    if (_targets.HasFlag(LoRATargetModules.Value))
                    {
                        list.Add(CreateAdapter(
                            layer, LoRATargetModules.Value, _dModel, dHead,
                            attn.WvHeads[h], attnSeed++, headIndex: h));
                    }

                    if (_targets.HasFlag(LoRATargetModules.OutputProjection))
                    {
                        list.Add(CreateAdapter(
                            layer, LoRATargetModules.OutputProjection, dHead, _dModel,
                            attn.WoHeads[h], attnSeed++, headIndex: h));
                    }
                }
            }

            return list.ToArray();
        }

        private ModuleAdapter CreateAdapter(
            int layer,
            LoRATargetModules module,
            int inDim,
            int outDim,
            Parameter wBase,
            int seed,
            int headIndex = 0,
            Parameter? bias = null)
        {
            var a = new Parameter(new TensorShape(inDim, _rank), requiresGrad: true, clearData: true);
            var b = new Parameter(new TensorShape(_rank, outDim), requiresGrad: true, clearData: true);

            InitializeA(a, seed);
            // B stays zero — standard LoRA init: the adapter starts as the identity.

            var adapter = new ModuleAdapter
            {
                Layer = layer,
                Module = module,
                HeadIndex = headIndex,
                InDim = inDim,
                OutDim = outDim,
                A = a,
                B = b,
                ANode = a.AsNode(),
                BNode = b.AsNode(),
            };

            if (_quantizeBase)
            {
                // QLoRA: freeze the base as Q4_K and inject FrozenQuantizedLinear(x) + bias + LoRA(x)
                // at the output (the 4-bit base can't be a single F32 weight node). The constructor
                // guarantees only LM-head / FFN targets reach here.
                adapter.QuantizedBase = QuantizeWeight(wBase, inDim, outDim);
                adapter.BiasNode = bias?.AsNode();
                adapter.OutputProvider = (graph, x) => BuildQLoRAOutput(graph, x, adapter);
            }
            else
            {
                adapter.WBaseNode = wBase.AsNode();
                adapter.Provider = graph => BuildEffectiveWeight(graph, adapter);
            }

            return adapter;
        }

        // Transpose the input-major weight [inDim, outDim] to output-major [outDim, inDim] and quantize
        // to the chosen frozen-base format (Q4_K = max RAM saving, or Q8_0 = higher fidelity / dim < 256).
        private IDequantRowSource QuantizeWeight(Parameter wBase, int inDim, int outDim)
        {
            var src = wBase.DataReadOnlySpan;
            var transposed = new float[(long)outDim * inDim];
            for (var k = 0; k < inDim; k++)
            {
                for (var o = 0; o < outDim; o++)
                {
                    transposed[o * inDim + k] = src[k * outDim + o];
                }
            }

            return _baseFormat == QLoRABaseFormat.Q8
                ? Q8Weight.QuantizeRows(transposed, outDim, inDim)
                : new Q4KWeight(GgmlQuant.QuantizeQ4_K(transposed, inDim, outDim), inDim, outDim);
        }

        // out = FrozenQuantizedLinear(x, Q4_K base) + bias(frozen) + (x·A)·B  — base frozen, only A/B train.
        private AutogradNode BuildQLoRAOutput(ComputationGraph graph, AutogradNode x, ModuleAdapter adapter)
        {
            var baseOut = graph.FrozenQuantizedLinear(x, adapter.QuantizedBase!);
            if (adapter.BiasNode is not null)
            {
                baseOut = graph.AddBias(baseOut, adapter.BiasNode);
            }

            var xa = graph.Linear(x, adapter.ANode, graph.CreateAuxiliary(new TensorShape(_rank), clearMemory: true));
            var lora = graph.Linear(xa, adapter.BNode, graph.CreateAuxiliary(new TensorShape(adapter.OutDim), clearMemory: true));
            return graph.Add(baseOut, lora);
        }

        private void AttachProviders()
        {
            foreach (var adapter in _adapters)
            {
                switch (adapter.Module)
                {
                    case LoRATargetModules.LanguageModelHead:
                        if (adapter.OutputProvider is not null)
                        {
                            _model.LMHeadOutputProvider = adapter.OutputProvider;
                        }
                        else
                        {
                            _model.LMHeadWeightProvider = adapter.Provider;
                        }

                        break;

                    case LoRATargetModules.FeedForwardUp:
                        if (adapter.OutputProvider is not null)
                        {
                            _model.Blocks[adapter.Layer].FFN.W1OutputProvider = adapter.OutputProvider;
                        }
                        else
                        {
                            _model.Blocks[adapter.Layer].FFN.W1WeightProvider = adapter.Provider;
                        }

                        break;

                    case LoRATargetModules.FeedForwardDown:
                        if (adapter.OutputProvider is not null)
                        {
                            _model.Blocks[adapter.Layer].FFN.W2OutputProvider = adapter.OutputProvider;
                        }
                        else
                        {
                            _model.Blocks[adapter.Layer].FFN.W2WeightProvider = adapter.Provider;
                        }

                        break;

                    case LoRATargetModules.Query:
                        _model.Blocks[adapter.Layer].Attention.SetQueryProvider(adapter.HeadIndex, adapter.Provider);
                        break;

                    case LoRATargetModules.Key:
                        _model.Blocks[adapter.Layer].Attention.SetKeyProvider(adapter.HeadIndex, adapter.Provider);
                        break;

                    case LoRATargetModules.Value:
                        _model.Blocks[adapter.Layer].Attention.SetValueProvider(adapter.HeadIndex, adapter.Provider);
                        break;

                    case LoRATargetModules.OutputProjection:
                        _model.Blocks[adapter.Layer].Attention.SetOutputProvider(adapter.HeadIndex, adapter.Provider);
                        break;
                }
            }
        }

        private void DetachProvider(ModuleAdapter adapter)
        {
            switch (adapter.Module)
            {
                case LoRATargetModules.LanguageModelHead:
                    if (adapter.OutputProvider is not null)
                    {
                        if (ReferenceEquals(_model.LMHeadOutputProvider, adapter.OutputProvider))
                        {
                            _model.LMHeadOutputProvider = null;
                        }
                    }
                    else if (ReferenceEquals(_model.LMHeadWeightProvider, adapter.Provider))
                    {
                        _model.LMHeadWeightProvider = null;
                    }

                    break;

                case LoRATargetModules.FeedForwardUp:
                    {
                        var ffn = _model.Blocks[adapter.Layer].FFN;
                        if (adapter.OutputProvider is not null)
                        {
                            if (ReferenceEquals(ffn.W1OutputProvider, adapter.OutputProvider))
                            {
                                ffn.W1OutputProvider = null;
                            }
                        }
                        else if (ReferenceEquals(ffn.W1WeightProvider, adapter.Provider))
                        {
                            ffn.W1WeightProvider = null;
                        }

                        break;
                    }

                case LoRATargetModules.FeedForwardDown:
                    {
                        var ffn = _model.Blocks[adapter.Layer].FFN;
                        if (adapter.OutputProvider is not null)
                        {
                            if (ReferenceEquals(ffn.W2OutputProvider, adapter.OutputProvider))
                            {
                                ffn.W2OutputProvider = null;
                            }
                        }
                        else if (ReferenceEquals(ffn.W2WeightProvider, adapter.Provider))
                        {
                            ffn.W2WeightProvider = null;
                        }

                        break;
                    }

                case LoRATargetModules.Query:
                    {
                        var attn = _model.Blocks[adapter.Layer].Attention;
                        if (ReferenceEquals(attn.GetQueryProvider(adapter.HeadIndex), adapter.Provider))
                        {
                            attn.SetQueryProvider(adapter.HeadIndex, null);
                        }

                        break;
                    }

                case LoRATargetModules.Key:
                    {
                        var attn = _model.Blocks[adapter.Layer].Attention;
                        if (ReferenceEquals(attn.GetKeyProvider(adapter.HeadIndex), adapter.Provider))
                        {
                            attn.SetKeyProvider(adapter.HeadIndex, null);
                        }

                        break;
                    }

                case LoRATargetModules.Value:
                    {
                        var attn = _model.Blocks[adapter.Layer].Attention;
                        if (ReferenceEquals(attn.GetValueProvider(adapter.HeadIndex), adapter.Provider))
                        {
                            attn.SetValueProvider(adapter.HeadIndex, null);
                        }

                        break;
                    }

                case LoRATargetModules.OutputProjection:
                    {
                        var attn = _model.Blocks[adapter.Layer].Attention;
                        if (ReferenceEquals(attn.GetOutputProvider(adapter.HeadIndex), adapter.Provider))
                        {
                            attn.SetOutputProvider(adapter.HeadIndex, null);
                        }

                        break;
                    }
            }
        }

        private static AutogradNode BuildEffectiveWeight(ComputationGraph graph, ModuleAdapter adapter)
        {
            // W_eff = W_base(frozen) + (A @ B). graph.Linear treats A@B as an
            // ordinary matmul; backward flows W_eff -> A@B -> A and B.
            var zeroBias = graph.CreateAuxiliary(new TensorShape(adapter.OutDim), clearMemory: true);
            var ab = graph.Linear(adapter.ANode, adapter.BNode, zeroBias);   // [inDim, outDim]
            return graph.Add(adapter.WBaseNode, ab);
        }

        private Parameter[] CollectParameters()
        {
            var parameters = new Parameter[_adapters.Length * 2];

            for (var i = 0; i < _adapters.Length; i++)
            {
                parameters[2 * i] = _adapters[i].A;
                parameters[(2 * i) + 1] = _adapters[i].B;
            }

            return parameters;
        }

        private int EstimateArenaFloats(int contextLength)
        {
            // Logits dominate; the FFN LoRA temporaries (A@B + Add per W1/W2 across
            // all blocks) add the third term.
            var perStep = ((long)contextLength * _vocab * 32)
                          + ((long)_dModel * _vocab * 8)
                          + ((long)_nLayers * _dModel * _dFF * 24);

            return (int)Math.Min(256_000_000L, Math.Max(16_000_000L, perStep));
        }

        private static float ComputeLossAndGrad(
            AutogradNode logits,
            ReadOnlySpan<int> targets,
            int vocab,
            bool writeGrad)
        {
            var data = logits.DataView.AsReadOnlySpan();
            var grad = writeGrad ? logits.GradView.AsSpan() : default;
            var seqLen = targets.Length;
            var invSeq = 1f / seqLen;
            var total = 0f;

            for (var t = 0; t < seqLen; t++)
            {
                var off = t * vocab;

                var max = data[off];
                for (var v = 1; v < vocab; v++)
                {
                    if (data[off + v] > max)
                    {
                        max = data[off + v];
                    }
                }

                var sumExp = 0f;
                for (var v = 0; v < vocab; v++)
                {
                    sumExp += MathF.Exp(data[off + v] - max);
                }

                var tgt = targets[t];
                total += max + MathF.Log(sumExp) - data[off + tgt];

                if (writeGrad)
                {
                    for (var v = 0; v < vocab; v++)
                    {
                        var softmax = MathF.Exp(data[off + v] - max) / sumExp;
                        grad[off + v] = (softmax - (v == tgt ? 1f : 0f)) * invSeq;
                    }
                }
            }

            return total / seqLen;
        }

        private void InitializeA(Parameter a, int seed)
        {
            // Kaiming-uniform U(-1/sqrt(rank), 1/sqrt(rank)) — matches LoRAWeight.
            var rng = new Random(seed);
            var bound = 1f / MathF.Sqrt(_rank);
            var data = a.DataSpan;

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = ((float)rng.NextDouble() * 2f - 1f) * bound;
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }

        /// <summary>One trained weight matrix: its LoRA factors and the graph hook.</summary>
        private sealed class ModuleAdapter
        {
            public int Layer;
            public LoRATargetModules Module;
            public int HeadIndex;
            public int InDim;
            public int OutDim;
            public Parameter A = null!;
            public Parameter B = null!;
            public AutogradNode ANode = null!;
            public AutogradNode BNode = null!;
            public AutogradNode? WBaseNode;
            public Func<ComputationGraph, AutogradNode>? Provider;

            // QLoRA: frozen Q4_K base + (optional) frozen bias + the output-level hook (replaces Provider).
            public IDequantRowSource? QuantizedBase;
            public AutogradNode? BiasNode;
            public Func<ComputationGraph, AutogradNode, AutogradNode>? OutputProvider;
        }
    }
}
