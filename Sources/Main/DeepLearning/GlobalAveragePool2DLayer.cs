// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Global Average Pooling 2D as an <see cref="IModule"/> wrapper around
    /// <see cref="TensorMath.GlobalAveragePool2D"/>. Reduces [batch, C, H, W] to [batch, C].
    /// </summary>
    public sealed class GlobalAveragePool2DLayer : IModule
    {
        private readonly int _channels;
        private readonly int _inputH;
        private readonly int _inputW;

        public bool IsTraining { get; private set; } = true;

        public GlobalAveragePool2DLayer(int channels, int inputH, int inputW)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);

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
            var inputStorage = new TensorStorage<float>(_channels * _inputH * _inputW, clearMemory: false);
            input.CopyTo(inputStorage.AsSpan());

            using var inputNode = new AutogradNode(
                inputStorage,
                new TensorShape(1, _channels, _inputH, _inputW),
                requiresGrad: false);

            using var outputNode = TensorMath.GlobalAveragePool2D(
                null!,
                inputNode,
                _channels, _inputH, _inputW);

            outputNode.DataView.AsReadOnlySpan().CopyTo(output);
        }

        public void InvalidateParameterCaches() { }

        public IEnumerable<AutogradNode> Parameters() => [];

        public void Save(BinaryWriter bw) { }

        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}
