// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Wraps an inner <see cref="IModule"/> so its forward pass runs under
    /// <see cref="ComputationGraph.Checkpoint"/> during training: the inner module's activations are NOT
    /// kept on the tape but recomputed in the backward pass — trading one extra forward for a much lower
    /// peak activation footprint. Drop one (or several) into a <see cref="Sequential"/> to checkpoint the
    /// heavy segments of a model. Transparent: identical result (bit-close), all parameters / lifecycle
    /// delegated to the inner module. Inference (<c>graph == null</c> / <see cref="ForwardInference"/>) is
    /// passed straight through (checkpointing is a training-only memory trade).
    /// </summary>
    public sealed class CheckpointedModule : IModule
    {
        private readonly IModule _inner;
        private readonly int _subArenaElements;

        public CheckpointedModule(IModule inner, int subArenaElements = 1 << 20)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(subArenaElements);
            _subArenaElements = subArenaElements;
        }

        public bool IsTraining => _inner.IsTraining;

        public void Train() => _inner.Train();

        public void Eval() => _inner.Eval();

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            // No graph (inference) → run the inner module directly; checkpointing only pays off in training.
            if (graph is null)
            {
                return _inner.Forward(graph, input);
            }

            return graph.Checkpoint((g, x) => _inner.Forward(g, x), input, _subArenaElements);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => _inner.ForwardInference(input, output);

        public IEnumerable<AutogradNode> Parameters() => _inner.Parameters();

        public void Save(BinaryWriter bw) => _inner.Save(bw);

        public void Load(BinaryReader br) => _inner.Load(br);

        public void InvalidateParameterCaches() => _inner.InvalidateParameterCaches();

        public void Dispose() => _inner.Dispose();
    }
}
