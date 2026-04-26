// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class GlobalAveragePool2DLayer : IModule, IInferenceShapeProvider
    {
        private readonly int _channels;
        private readonly int _h;
        private readonly int _w;
        private readonly int _inputSize;
        private readonly int _outputSize;

        public GlobalAveragePool2DLayer(
            int channels,
            int h,
            int w)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(h);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(w);

            _channels = channels;
            _h = h;
            _w = w;
            _inputSize = channels * h * w;
            _outputSize = channels;
        }

        public bool IsTraining { get; private set; } = true;

        public int InferenceInputSize => _inputSize;

        public int InferenceOutputSize => _outputSize;

        public void Train()
        {
            IsTraining = true;
        }

        public void Eval()
        {
            IsTraining = false;
            PrepareInference();
        }

        public void PrepareInference()
        {
        }

        public AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode input)
        {
            return TensorMath.GlobalAveragePool2D(
                graph,
                input,
                _channels,
                _h,
                _w);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            return [];
        }

        public void Save(BinaryWriter bw)
        {
        }

        public void Load(BinaryReader br)
        {
        }

        public void ForwardInference(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            PoolingKernels.GlobalAveragePool2DForwardNchw(
                input,
                output,
                _channels,
                _h,
                _w);
        }

        public void Dispose()
        {
        }

        public void InvalidateParameterCaches()
        {
        }
    }
}
