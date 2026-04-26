// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class GlobalAveragePool2DLayer : IModule, IInferenceShapeProvider
    {
        private readonly int _channels;
        private readonly int _h;
        private readonly int _w;
        private readonly int _spatialSize;
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly float _scale;

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
            _spatialSize = h * w;
            _inputSize = channels * h * w;
            _outputSize = channels;
            _scale = 1f / _spatialSize;
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
            if (input.Length % _inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by GlobalAveragePool2DLayer inference input size.",
                    nameof(input));
            }

            var batchSize = input.Length / _inputSize;
            var expectedOutputLength = batchSize * _outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException(
                    "Output span is too small for GlobalAveragePool2DLayer inference.",
                    nameof(output));
            }

            for (var n = 0; n < batchSize; n++)
            {
                ForwardInferenceSingleBatch(
                    input.Slice(n * _inputSize, _inputSize),
                    output.Slice(n * _outputSize, _outputSize));
            }
        }

        private void ForwardInferenceSingleBatch(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            for (var c = 0; c < _channels; c++)
            {
                var channel = input.Slice(c * _spatialSize, _spatialSize);
                output[c] = TensorPrimitives.Sum(channel) * _scale;
            }
        }

        public void Dispose()
        {
        }

        public void InvalidateParameterCaches()
        {
        }
    }
}