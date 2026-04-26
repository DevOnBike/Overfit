// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Kernels
{
    internal static class PoolingKernels
    {
        public static void MaxPool2DForwardNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW,
            int pool)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pool);

            if (inputH % pool != 0 || inputW % pool != 0)
            {
                throw new ArgumentException("MaxPool2D requires inputH and inputW divisible by pool.");
            }

            var outH = inputH / pool;
            var outW = inputW / pool;
            var inputSize = channels * inputH * inputW;
            var outputSize = channels * outH * outW;

            if (input.Length % inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by MaxPool2D input size.",
                    nameof(input));
            }

            var batchSize = input.Length / inputSize;
            var expectedOutputLength = batchSize * outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException(
                    "Output span is too small for MaxPool2D.",
                    nameof(output));
            }

            for (var n = 0; n < batchSize; n++)
            {
                MaxPool2DForwardSingleBatchNchw(
                    input.Slice(n * inputSize, inputSize),
                    output.Slice(n * outputSize, outputSize),
                    channels,
                    inputH,
                    inputW,
                    pool,
                    outH,
                    outW);
            }
        }

        public static void GlobalAveragePool2DForwardNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);

            var spatialSize = inputH * inputW;
            var inputSize = channels * spatialSize;
            var outputSize = channels;

            if (input.Length % inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by GlobalAveragePool2D input size.",
                    nameof(input));
            }

            var batchSize = input.Length / inputSize;
            var expectedOutputLength = batchSize * outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException(
                    "Output span is too small for GlobalAveragePool2D.",
                    nameof(output));
            }

            var scale = 1f / spatialSize;

            for (var n = 0; n < batchSize; n++)
            {
                GlobalAveragePool2DForwardSingleBatchNchw(
                    input.Slice(n * inputSize, inputSize),
                    output.Slice(n * outputSize, outputSize),
                    channels,
                    spatialSize,
                    scale);
            }
        }

        private static void MaxPool2DForwardSingleBatchNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW,
            int pool,
            int outH,
            int outW)
        {
            for (var c = 0; c < channels; c++)
            {
                var inputChannelBase = c * inputH * inputW;
                var outputChannelBase = c * outH * outW;

                for (var oh = 0; oh < outH; oh++)
                {
                    for (var ow = 0; ow < outW; ow++)
                    {
                        var max = float.MinValue;

                        for (var ph = 0; ph < pool; ph++)
                        {
                            var iy = oh * pool + ph;
                            var inputRowBase = inputChannelBase + iy * inputW + ow * pool;

                            for (var pw = 0; pw < pool; pw++)
                            {
                                var value = input[inputRowBase + pw];

                                if (value > max)
                                {
                                    max = value;
                                }
                            }
                        }

                        output[outputChannelBase + oh * outW + ow] = max;
                    }
                }
            }
        }

        private static void GlobalAveragePool2DForwardSingleBatchNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int spatialSize,
            float scale)
        {
            for (var c = 0; c < channels; c++)
            {
                var channel = input.Slice(c * spatialSize, spatialSize);
                output[c] = TensorPrimitives.Sum(channel) * scale;
            }
        }
    }
}
