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
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(poolSize);

            if (inputH % poolSize != 0)
            {
                throw new ArgumentException($"inputH ({inputH}) must be divisible by poolSize ({poolSize}).");
            }

            if (inputW % poolSize != 0)
            {
                throw new ArgumentException($"inputW ({inputW}) must be divisible by poolSize ({poolSize}).");
            }

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
            var inputStorage = new TensorStorage<float>(_channels * _inputH * _inputW, clearMemory: false);
            input.CopyTo(inputStorage.AsSpan());

            using var inputNode = new AutogradNode(
                inputStorage,
                new TensorShape(1, _channels, _inputH, _inputW),
                requiresGrad: false);

            using var outputNode = TensorMath.MaxPool2D(
                null!,
                inputNode,
                _channels, _inputH, _inputW, _poolSize);

            outputNode.DataView.AsReadOnlySpan().CopyTo(output);
        }

        public void InvalidateParameterCaches() { }

        public IEnumerable<AutogradNode> Parameters() => [];

        public void Save(BinaryWriter bw) { }

        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}
