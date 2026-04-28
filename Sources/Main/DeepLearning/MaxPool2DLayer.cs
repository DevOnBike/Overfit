// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// 2D max pooling as an <see cref="IModule"/> wrapper around <see cref="TensorMath.MaxPool2D"/>.
    /// Required because <see cref="Sequential"/> composes <see cref="IModule"/>s; non-parametric
    /// pooling functions in <c>TensorMath</c> cannot be inserted into a Sequential pipeline directly.
    /// </summary>
    public sealed class MaxPool2DLayer : IModule
    {
        private readonly int _channels;
        private readonly int _inputH;
        private readonly int _inputW;
        private readonly int _poolSize;

        public bool IsTraining { get; private set; } = true;

        public MaxPool2DLayer(int channels, int inputH, int inputW, int poolSize)
        {
            if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
            if (inputH <= 0) throw new ArgumentOutOfRangeException(nameof(inputH));
            if (inputW <= 0) throw new ArgumentOutOfRangeException(nameof(inputW));
            if (poolSize <= 0) throw new ArgumentOutOfRangeException(nameof(poolSize));
            if (inputH % poolSize != 0) throw new ArgumentException($"inputH ({inputH}) must be divisible by poolSize ({poolSize}).");
            if (inputW % poolSize != 0) throw new ArgumentException($"inputW ({inputW}) must be divisible by poolSize ({poolSize}).");

            _channels = channels;
            _inputH = inputH;
            _inputW = inputW;
            _poolSize = poolSize;
        }

        public int OutputH => _inputH / _poolSize;
        public int OutputW => _inputW / _poolSize;

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.MaxPool2D(graph, input, _channels, _inputH, _inputW, _poolSize);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            // TensorMath.MaxPool2D requires AutogradNode; for single-batch inference we
            // wrap the input in a transient FastTensor + node, run forward, copy result.
            // Allocating temporaries here is acceptable given non-Linear inference rarely
            // appears on the most latency-critical path; revisit if profiling demands.
            using var inTensor = new FastTensor<float>(1, _channels, _inputH, _inputW, clearMemory: false);
            input.CopyTo(inTensor.GetView().AsSpan());
            using var inNode = new AutogradNode(inTensor, requiresGrad: false);

            using var outNode = TensorMath.MaxPool2D(null!, inNode, _channels, _inputH, _inputW, _poolSize);
            outNode.DataView.AsReadOnlySpan().CopyTo(output);
        }

        public IEnumerable<AutogradNode> Parameters() => Array.Empty<AutogradNode>();

        public void Save(BinaryWriter bw) { /* no learnable parameters */ }
        public void Load(BinaryReader br) { /* no learnable parameters */ }

        public void Dispose() { /* nothing to release */ }
    }
}
