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
    public sealed class MaxPool2DLayer : IModule, IInferenceShapeProvider
    {
        private readonly int _channels;
        private readonly int _h;
        private readonly int _w;
        private readonly int _pool;
        private readonly int _outH;
        private readonly int _outW;
        private readonly int _inputSize;
        private readonly int _outputSize;

        public MaxPool2DLayer(
            int channels,
            int h,
            int w,
            int pool)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(h);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(w);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pool);

            if (h % pool != 0 || w % pool != 0)
            {
                throw new ArgumentException("MaxPool2DLayer requires h and w divisible by pool.");
            }

            _channels = channels;
            _h = h;
            _w = w;
            _pool = pool;
            _outH = h / pool;
            _outW = w / pool;
            _inputSize = channels * h * w;
            _outputSize = channels * _outH * _outW;
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
            return TensorMath.MaxPool2D(
                graph,
                input,
                _channels,
                _h,
                _w,
                _pool);
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
            PoolingKernels.MaxPool2DForwardNchw(
                input,
                output,
                _channels,
                _h,
                _w,
                _pool);
        }

        public void Dispose()
        {
        }

        public void InvalidateParameterCaches()
        {
        }
    }
}
