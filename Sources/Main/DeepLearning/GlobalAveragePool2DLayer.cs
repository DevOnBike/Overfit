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
    /// Global Average Pooling 2D as an <see cref="IModule"/> wrapper around
    /// <see cref="TensorMath.GlobalAveragePool2D"/>. Reduces spatial dimensions to scalar
    /// per channel (output shape becomes [batch, channels]).
    /// </summary>
    public sealed class GlobalAveragePool2DLayer : IModule
    {
        private readonly int _channels;
        private readonly int _inputH;
        private readonly int _inputW;

        public bool IsTraining { get; private set; } = true;

        public GlobalAveragePool2DLayer(int channels, int inputH, int inputW)
        {
            if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
            if (inputH <= 0) throw new ArgumentOutOfRangeException(nameof(inputH));
            if (inputW <= 0) throw new ArgumentOutOfRangeException(nameof(inputW));

            _channels = channels;
            _inputH = inputH;
            _inputW = inputW;
        }

        public void Train() => IsTraining = true;
        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.GlobalAveragePool2D(graph, input, _channels, _inputH, _inputW);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            using var inTensor = new FastTensor<float>(1, _channels, _inputH, _inputW, clearMemory: false);
            input.CopyTo(inTensor.GetView().AsSpan());
            using var inNode = new AutogradNode(inTensor, requiresGrad: false);

            using var outNode = TensorMath.GlobalAveragePool2D(null!, inNode, _channels, _inputH, _inputW);
            outNode.DataView.AsReadOnlySpan().CopyTo(output);
        }

        public IEnumerable<AutogradNode> Parameters() => Array.Empty<AutogradNode>();

        public void Save(BinaryWriter bw) { /* no learnable parameters */ }
        public void Load(BinaryReader br) { /* no learnable parameters */ }

        public void Dispose() { /* nothing to release */ }
    }
}
