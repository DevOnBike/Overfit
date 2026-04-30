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
    /// Windowed average pooling (NCHW layout, 2D, square kernel).
    ///
    /// Supports any kernel size, stride, and symmetric padding.
    /// <see cref="CountIncludePad"/> matches the ONNX attribute of the same name.
    ///
    /// Output dims:
    ///   outH = (inputH + 2*padding - kernelSize) / stride + 1
    ///   outW = (inputW + 2*padding - kernelSize) / stride + 1
    ///
    /// Inference-only: no autograd backward. ONNX AveragePool → this layer.
    /// </summary>
    public sealed class AveragePool2DLayer : IModule, IInferenceShapeProvider
    {
        private readonly int _channels;
        private readonly int _inputH;
        private readonly int _inputW;
        private readonly int _kernelSize;
        private readonly int _padding;
        private readonly int _stride;
        private readonly int _outH;
        private readonly int _outW;

        public AveragePool2DLayer(
            int channels,
            int inputH,
            int inputW,
            int kernelSize,
            int padding = 0,
            int stride = 1,
            bool countIncludePad = false)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kernelSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stride);
            ArgumentOutOfRangeException.ThrowIfNegative(padding);

            _channels       = channels;
            _inputH         = inputH;
            _inputW         = inputW;
            _kernelSize     = kernelSize;
            _padding        = padding;
            _stride         = stride;
            CountIncludePad = countIncludePad;
            _outH = (inputH + 2 * padding - kernelSize) / stride + 1;
            _outW = (inputW + 2 * padding - kernelSize) / stride + 1;
        }

        /// <summary>
        /// When true, padding zeros count toward the average divisor (ONNX default: false).
        /// </summary>
        public bool CountIncludePad { get; }

        public bool IsTraining { get; private set; } = true;

        public int InferenceInputSize  => _channels * _inputH * _inputW;
        public int InferenceOutputSize => _channels * _outH   * _outW;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        public void PrepareInference() { }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
            => throw new NotSupportedException(
                "AveragePool2DLayer.Forward with autograd is not implemented. " +
                "Use ForwardInference for inference-only paths.");

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            PoolingKernels.AveragePool2DForwardNchw(
                input, output,
                batchSize: 1,
                _channels, _inputH, _inputW,
                _kernelSize, _padding, _stride,
                CountIncludePad);
        }

        public IEnumerable<AutogradNode> Parameters() => [];

        public void InvalidateParameterCaches() { }

        public void Save(BinaryWriter bw) { }

        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}
