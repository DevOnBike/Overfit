// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Token embedding lookup table.
    ///
    /// Maps integer token ids to dense embedding vectors.
    /// The embedding table (weight matrix) is learnable: shape [vocabSize, embeddingDim].
    ///
    /// Backward: gradient scatter-add into accessed embedding rows.
    /// Rows not accessed in the current batch receive zero gradient.
    ///
    /// Usage:
    ///   var emb = new EmbeddingLayer(vocabSize: 50257, embeddingDim: 768);
    ///   var output = emb.Forward(graph, tokenIds);  // [seqLen, 768]
    ///
    /// Initialization: N(0, 1) by default (standard practice for embeddings).
    /// For GPT-style models: typically followed by positional embedding addition.
    /// </summary>
    public sealed class EmbeddingLayer : IModule
    {
        private readonly int _vocabSize;
        private readonly int _embeddingDim;

        private AutogradNode? _embNode;

        public EmbeddingLayer(int vocabSize, int embeddingDim)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(embeddingDim);

            _vocabSize    = vocabSize;
            _embeddingDim = embeddingDim;

            Weight = new Parameter(
                new TensorShape(vocabSize, embeddingDim),
                requiresGrad: true,
                clearData: false);

            // Initialise with N(0, 1) — standard for embeddings.
            var span = Weight.DataSpan;
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = MathUtils.NextGaussian();
            }
        }

        /// <summary>The embedding table, shape [vocabSize, embeddingDim].</summary>
        public Parameter Weight { get; }

        public int VocabSize    => _vocabSize;
        public int EmbeddingDim => _embeddingDim;

        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        /// <summary>
        /// Looks up embeddings for the given token ids.
        /// </summary>
        /// <param name="graph">Autograd tape.</param>
        /// <param name="tokenIds">Integer token ids, values in [0, vocabSize).</param>
        /// <returns>AutogradNode of shape [seqLen, embeddingDim].</returns>
        public AutogradNode Forward(ComputationGraph graph, int[] tokenIds)
        {
            _embNode ??= Weight.AsNode();
            return TensorMath.Embedding(graph, tokenIds, _embNode);
        }

        /// <summary>IModule compatibility — not meaningful for Embedding (needs token ids).</summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
            => throw new NotSupportedException(
                "EmbeddingLayer.Forward(ComputationGraph, AutogradNode) is not supported. " +
                "Use Forward(ComputationGraph, int[]) with token id arrays.");

        /// <summary>Single-sample inference lookup.</summary>
        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => throw new NotSupportedException(
                "EmbeddingLayer.ForwardInference(Span) is not supported. " +
                "Use Forward(ComputationGraph, int[]) for inference.");

        /// <summary>Looks up a single token id without autograd.</summary>
        public void LookupInference(int tokenId, Span<float> output)
        {
            if ((uint)tokenId >= (uint)_vocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokenId),
                    $"Token id {tokenId} out of range [0, {_vocabSize}).");
            }

            Weight.DataReadOnlySpan
                  .Slice(tokenId * _embeddingDim, _embeddingDim)
                  .CopyTo(output);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weight.AsNode();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return Weight;
        }

        public void InvalidateParameterCaches()
        {
            _embNode = null;
        }

        public void Save(BinaryWriter bw)
        {
            Weight.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            Weight.Load(br);
        }

        public void Dispose()
        {
            _embNode?.Dispose();
            Weight.Dispose();
        }
    }
}
