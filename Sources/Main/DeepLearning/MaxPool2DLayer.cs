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
        private readonly int _stride;
        private readonly int _padding;

        /// <param name="stride">Window step. <c>0</c> (the default) means "equal to <paramref name="poolSize"/>"
        /// — the classic non-overlapping pool. A smaller value gives an overlapping pool (e.g. ResNet's 3x3
        /// stride-2).</param>
        /// <param name="padding">Symmetric zero-padding on each spatial edge (ONNX pads with -inf for max).</param>
        public MaxPool2DLayer(int channels, int inputH, int inputW, int poolSize, int stride = 0, int padding = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(poolSize);
            ArgumentOutOfRangeException.ThrowIfNegative(padding);

            if (stride == 0)
            {
                stride = poolSize;
            }
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stride);

            // The fast non-overlapping path requires exact tiling; the general (overlapping / padded) path uses
            // the ONNX output-size formula and has no divisibility constraint.
            if (IsSimple(poolSize, stride, padding))
            {
                if (inputH % poolSize != 0)
                {
                    throw new ArgumentException($"inputH ({inputH}) must be divisible by poolSize ({poolSize}).");
                }

                if (inputW % poolSize != 0)
                {
                    throw new ArgumentException($"inputW ({inputW}) must be divisible by poolSize ({poolSize}).");
                }
            }

            _channels = channels;
            _inputH = inputH;
            _inputW = inputW;
            _poolSize = poolSize;
            _stride = stride;
            _padding = padding;
        }

        private static bool IsSimple(int poolSize, int stride, int padding)
            => stride == poolSize && padding == 0;

        public int OutputH
        {
            get
            {
                return (_inputH + 2 * _padding - _poolSize) / _stride + 1;
            }
        }

        public int OutputW
        {
            get
            {
                return (_inputW + 2 * _padding - _poolSize) / _stride + 1;
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

        public void PrepareInference()
        {
        }

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
            if (!IsSimple(_poolSize, _stride, _padding))
            {
                throw new OverfitRuntimeException(
                    "Training-graph MaxPool only supports non-overlapping, unpadded pooling (stride == poolSize, "
                    + "padding == 0). The overlapping / padded path is inference-only (ONNX import).");
            }

            return ComputationGraph.MaxPool2DOp(graph, input, _channels, _inputH, _inputW, _poolSize);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (IsSimple(_poolSize, _stride, _padding))
            {
                PoolingKernels.MaxPool2DForwardNchw(
                    input,
                    output,
                    _channels,
                    _inputH,
                    _inputW,
                    _poolSize);
                return;
            }

            var batchSize = input.Length / InferenceInputSize;
            PoolingKernels.MaxPool2DForwardNchw(
                input,
                output,
                batchSize,
                _channels,
                _inputH,
                _inputW,
                kernelSize: _poolSize,
                padding: _padding,
                stride: _stride);
        }

        public void InvalidateParameterCaches()
        {
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
        public void Dispose()
        {
        }
    }
}