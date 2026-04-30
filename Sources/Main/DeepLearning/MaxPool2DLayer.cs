// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// 2D max pooling as <see cref="IModule"/> + <see cref="IInferenceShapeProvider"/>.
    /// Uses <see cref="PoolingKernels.MaxPool2DForwardNchw"/> for zero-allocation inference.
    /// </summary>
    public sealed class MaxPool2DLayer : IModule, IInferenceShapeProvider
    {
        private readonly int _channels;
        private readonly int _inputH;
        private readonly int _inputW;
        private readonly int _poolSize;

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
            _inputH   = inputH;
            _inputW   = inputW;
            _poolSize = poolSize;
        }

        public int OutputH
        {
            get
            {
                return _inputH / _poolSize;
            }
        }

        public int OutputW
        {
            get
            {
                return _inputW / _poolSize;
            }
        }

        public int InferenceInputSize
        {
            get
            {
                return _channels * _inputH * _inputW;
            }
        }

        public int InferenceOutputSize
        {
            get
            {
                return _channels * OutputH * OutputW;
            }
        }

        public void PrepareInference() { }

        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
        }
        public void Eval()
        {
            IsTraining = false;
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return ComputationGraph.MaxPool2DOp(graph, input, _channels, _inputH, _inputW, _poolSize);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            PoolingKernels.MaxPool2DForwardNchw(
                input,
                output,
                _channels,
                _inputH,
                _inputW,
                _poolSize);
        }

        public void InvalidateParameterCaches() { }
        public IEnumerable<AutogradNode> Parameters()
        {
            return [];
        }
        public void Save(BinaryWriter bw) { }
        public void Load(BinaryReader br) { }
        public void Dispose() { }
    }
}