// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Kernels
{
    /// <summary>
    /// Experimental specialized pooling kernels.
    ///
    /// These are intentionally separate from PoolingKernels until benchmarks prove
    /// that the specialization is worth wiring into the production TensorMath path.
    /// </summary>
    internal static class PoolingFastPathKernels
    {
        /// <summary>
        /// Specialized NCHW max-pool forward path for:
        ///
        /// - pool = 2
        /// - stride = 2
        /// - padding = 0
        /// - input shape [N, C, H, W]
        /// - output shape [N, C, H / 2, W / 2]
        ///
        /// This avoids generic kh/kw loops and performs exactly four reads per output.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void MaxPool2DForwardNchwPool2Stride2(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW)
        {
            if (channels <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(channels));
            }

            if (inputH <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputH));
            }

            if (inputW <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputW));
            }

            if ((inputH & 1) != 0)
            {
                throw new ArgumentException(
                    "Input height must be divisible by 2 for pool=2/stride=2 fast path.",
                    nameof(inputH));
            }

            if ((inputW & 1) != 0)
            {
                throw new ArgumentException(
                    "Input width must be divisible by 2 for pool=2/stride=2 fast path.",
                    nameof(inputW));
            }

            var inputPlaneSize = checked(channels * inputH * inputW);

            if (inputPlaneSize <= 0)
            {
                throw new ArgumentException(
                    "Invalid input dimensions.");
            }

            if (input.Length % inputPlaneSize != 0)
            {
                throw new ArgumentException(
                    $"Input length {input.Length} is not divisible by C*H*W={inputPlaneSize}.",
                    nameof(input));
            }

            var batchSize = input.Length / inputPlaneSize;

            var outputH = inputH / 2;
            var outputW = inputW / 2;
            var expectedOutput = checked(batchSize * channels * outputH * outputW);

            if (output.Length < expectedOutput)
            {
                throw new ArgumentException(
                    $"Output span is too small. Expected at least {expectedOutput}, got {output.Length}.",
                    nameof(output));
            }

            for (var n = 0; n < batchSize; n++)
            {
                var batchInputBase = n * channels * inputH * inputW;
                var batchOutputBase = n * channels * outputH * outputW;

                for (var c = 0; c < channels; c++)
                {
                    var channelInputBase = batchInputBase + (c * inputH * inputW);
                    var channelOutputBase = batchOutputBase + (c * outputH * outputW);

                    for (var oh = 0; oh < outputH; oh++)
                    {
                        var inputRow0 = channelInputBase + ((oh * 2) * inputW);
                        var inputRow1 = inputRow0 + inputW;
                        var outputRow = channelOutputBase + (oh * outputW);

                        for (var ow = 0; ow < outputW; ow++)
                        {
                            var iw = ow * 2;

                            var a = input[inputRow0 + iw];
                            var b = input[inputRow0 + iw + 1];
                            var c0 = input[inputRow1 + iw];
                            var d = input[inputRow1 + iw + 1];

                            var max0 = a > b ? a : b;
                            var max1 = c0 > d ? c0 : d;

                            output[outputRow + ow] = max0 > max1 ? max0 : max1;
                        }
                    }
                }
            }
        }
    }
}