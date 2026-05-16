// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// LoRA fine-tuning for a <see cref="GPT1Model"/> — Stage 1: LM head only.
    ///
    /// The base model is frozen; only two low-rank factors are trained:
    ///   W_eff = LMHead(frozen) + (A @ B)
    ///   A : [dModel x rank]   (Kaiming-uniform init)
    ///   B : [rank x vocab]    (zero init — adapter starts as identity)
    ///
    /// A and B become <see cref="Parameter"/>s on the autograd graph, so
    /// <c>ComputationGraph.Backward</c> produces their gradients automatically
    /// (validated by LoRAEffectiveWeightInjectionTests). The optimizer only ever
    /// sees {A, B}, so the base weights are never updated.
    ///
    /// While a fine-tuner is attached, the model's <see cref="GPT1Model"/>
    /// LM-head is LoRA-adapted via <c>GPT1Model.LMHeadWeightProvider</c>;
    /// <see cref="Dispose"/> detaches it.
    ///
    /// Exported adapters use the LoRA .bin format (magic "LORA", one entry keyed
    /// <see cref="LoRATargetModules.LanguageModelHead"/>) shared with
    /// <see cref="LlamaLoRAAdapter"/>.
    /// </summary>
    public sealed class Gpt1LoRAFineTuner : IDisposable
    {
        private readonly GPT1Model _model;
        private readonly int _dModel;
        private readonly int _vocab;
        private readonly int _rank;

        private readonly Parameter _a;          // [dModel, rank]
        private readonly Parameter _b;          // [rank, vocab]
        private readonly AutogradNode _aNode;
        private readonly AutogradNode _bNode;
        private readonly AutogradNode _wBaseNode;   // frozen view of LMHead

        private bool _disposed;

        public Gpt1LoRAFineTuner(GPT1Model model, int rank, int seed = 42)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);

            if (model.Config.TieWeights)
            {
                throw new NotSupportedException(
                    "Gpt1LoRAFineTuner targets the untied LM head; the model must be built with TieWeights=false.");
            }

            _model = model;
            _dModel = model.Config.DModel;
            _vocab = model.Config.VocabSize;
            _rank = rank;

            _a = new Parameter(new TensorShape(_dModel, _rank), requiresGrad: true, clearData: true);
            _b = new Parameter(new TensorShape(_rank, _vocab), requiresGrad: true, clearData: true);

            InitializeA(seed);
            // B stays zero — standard LoRA init: the adapter starts as the identity.

            _aNode = _a.AsNode();
            _bNode = _b.AsNode();
            _wBaseNode = model.LMHead.AsNode();

            // Attach the LoRA-adapted LM head. Detached again in Dispose().
            _model.LMHeadWeightProvider = BuildEffectiveLMHead;
        }

        public int Rank => _rank;

        public long TrainableParameterCount => (long)_a.Shape.Size + _b.Shape.Size;

        /// <summary>
        /// Fine-tunes the LoRA factors on a flat token corpus (next-token loss).
        /// Returns the per-step training loss for inspection.
        /// </summary>
        public IReadOnlyList<float> FineTune(
            int[] corpus,
            int steps,
            int contextLength,
            float learningRate,
            int seed = 1234)
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

            _model.Eval(); // base is frozen — no dropout / train-mode behaviour wanted

            var arenaFloats = (int)Math.Min(
                256_000_000L,
                Math.Max(16_000_000L, (long)contextLength * _vocab * 32 + (long)_dModel * _vocab * 8));

            using var graph = new ComputationGraph(arenaFloats);
            using var optimizer = new Adam(new[] { _a, _b }, learningRate) { UseAdamW = true };

            var rng = new Random(seed);
            var input = new int[contextLength];
            var target = new int[contextLength];
            var history = new float[steps];

            for (var step = 0; step < steps; step++)
            {
                var start = rng.Next(0, corpus.Length - contextLength - 1);
                corpus.AsSpan(start, contextLength).CopyTo(input);
                corpus.AsSpan(start + 1, contextLength).CopyTo(target);

                graph.Reset();
                _model.InvalidateAllCaches();

                _a.ZeroGrad();
                _b.ZeroGrad();

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

            var arenaFloats = (int)Math.Min(
                256_000_000L,
                Math.Max(16_000_000L, (long)contextLength * _vocab * 32 + (long)_dModel * _vocab * 8));

            using var graph = new ComputationGraph(arenaFloats);

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
        /// Saves the trained LoRA factors in the shared LoRA .bin format
        /// (one entry, keyed <see cref="LoRATargetModules.LanguageModelHead"/>).
        /// </summary>
        public void Save(string path)
        {
            ThrowIfDisposed();

            var weight = new LoRAWeight(_dModel, _vocab, _rank);
            _a.DataReadOnlySpan.CopyTo(weight.AMutable);
            _b.DataReadOnlySpan.CopyTo(weight.BMutable);

            Gpt1LoRAFile.SaveLMHead(path, weight);
        }

        /// <summary>Loads LoRA factors previously written by <see cref="Save"/>.</summary>
        public void Load(string path)
        {
            ThrowIfDisposed();

            var weight = Gpt1LoRAFile.LoadLMHead(path);
            if (weight.InDim != _dModel || weight.OutDim != _vocab || weight.Rank != _rank)
            {
                throw new InvalidDataException(
                    $"LoRA dimensions [{weight.InDim}x{weight.OutDim} r={weight.Rank}] " +
                    $"do not match this fine-tuner [{_dModel}x{_vocab} r={_rank}].");
            }

            weight.A.CopyTo(_a.DataSpan);
            weight.B.CopyTo(_b.DataSpan);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            if (ReferenceEquals(_model.LMHeadWeightProvider, (Func<ComputationGraph, AutogradNode>)BuildEffectiveLMHead))
            {
                _model.LMHeadWeightProvider = null;
            }

            _aNode.Dispose();
            _bNode.Dispose();
            _wBaseNode.Dispose();
            _a.Dispose();
            _b.Dispose();
        }

        // ── Private ───────────────────────────────────────────────────────────

        private AutogradNode BuildEffectiveLMHead(ComputationGraph graph)
        {
            // W_eff = W_base(frozen) + (A @ B). graph.Linear treats A@B as an
            // ordinary matmul; backward flows W_eff -> A@B -> A and B.
            var zeroBias = graph.CreateAuxiliary(new TensorShape(_vocab), clearMemory: true);
            var ab = graph.Linear(_aNode, _bNode, zeroBias);   // [dModel, vocab]
            return graph.Add(_wBaseNode, ab);
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

        private void InitializeA(int seed)
        {
            // Kaiming-uniform U(-1/sqrt(rank), 1/sqrt(rank)) — matches LoRAWeight.
            var rng = new Random(seed);
            var bound = 1f / MathF.Sqrt(_rank);
            var data = _a.DataSpan;

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = ((float)rng.NextDouble() * 2f - 1f) * bound;
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
