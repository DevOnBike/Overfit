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
    /// Global Average Pooling 2D as <see cref="IModule"/> + <see cref="IInferenceShapeProvider"/>.
    /// Uses <see cref="PoolingKernels.GlobalAveragePool2DForwardNchw"/> for zero-allocation inference.
    /// Reduces [batch, C, H, W] Ã¢â€ â€™ [batch, C].
    /// </summary>
    public sealed class GlobalAveragePool2DLayer : IModule, IInferenceShapeProvider
    {
        private readonly int _channels;
        private readonly int _inputH;
        private readonly int _inputW;

        public GlobalAveragePool2DLayer(int channels, int inputH, int inputW)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);

            _channels = channels;
            _inputH   = inputH;
            _inputW   = inputW;
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
                return _channels;
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
            return ComputationGraph.GlobalAveragePool2DOp(graph, input, _channels, _inputH, _inputW);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            PoolingKernels.GlobalAveragePool2DForwardNchw(
                input,
                output,
                _channels,
                _inputH,
                _inputW);
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