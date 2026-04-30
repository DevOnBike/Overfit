// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Kernels
{
    internal static class PoolingKernels
    {
        // ─────────────────────────────────────────────────────────────────────
        // MaxPool2D forward — inference only (no index tracking)
        // ─────────────────────────────────────────────────────────────────────

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

            if (output.Length < batchSize * outputSize)
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

        // ─────────────────────────────────────────────────────────────────────
        // MaxPool2D forward — training path (with index tracking)
        // Fills output values + flat maxIndices for backward scatter.
        // batchOffset is the flat offset of this batch's input start in the
        // full [B, C, H, W] tensor, so indices are globally addressable.
        // ─────────────────────────────────────────────────────────────────────

        public static void MaxPool2DForwardWithIndicesNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            Span<float> maxIndices,
            int channels,
            int inputH,
            int inputW,
            int pool,
            int batchOffset)
        {
            var outH = inputH / pool;
            var outW = inputW / pool;

            if (pool == 2 && inputW % 2 == 0)
            {
                MaxPool2DForwardWithIndicesPool2(
                    input, output, maxIndices,
                    channels, inputH, inputW, outH, outW, batchOffset);
            }
            else
            {
                MaxPool2DForwardWithIndicesGeneric(
                    input, output, maxIndices,
                    channels, inputH, inputW, pool, outH, outW, batchOffset);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // GlobalAveragePool2D forward
        // ─────────────────────────────────────────────────────────────────────

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

            if (output.Length < batchSize * outputSize)
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

        // ─────────────────────────────────────────────────────────────────────
        // Private: inference single batch
        // ─────────────────────────────────────────────────────────────────────

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
            if (pool == 2 && inputW % 2 == 0)
            {
                MaxPool2DForwardSingleBatchPool2NoIndex(
                    input, output, channels, inputH, inputW, outH, outW);
                return;
            }

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
                            var rowBase = inputChannelBase + (oh * pool + ph) * inputW + ow * pool;
                            for (var pw = 0; pw < pool; pw++)
                            {
                                var value = input[rowBase + pw];
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

        /// <summary>
        /// Pool=2, stride=2 fast path without index recording (inference).
        ///
        /// Two-step approach per output row:
        ///   1) TensorPrimitives.Max(row0, row1, pairMax) — SIMD element-wise
        ///      vertical max across the two input rows (hardware-vectorised by the
        ///      JIT via AVX2/AVX-512 on supported CPUs).
        ///   2) Scalar loop over outW — collapses adjacent horizontal pairs.
        ///      Step 2 is only outW iterations (13 for MNIST), negligible cost.
        ///
        /// Benchmark context: for [64, 8, 26, 26] → [64, 8, 13, 13]:
        ///   - 64 batches × 8 channels × 13 rows = 6656 TensorPrimitives.Max calls
        ///     each on inputW=26 floats → fully vectorised
        ///   - eliminates all branching in the hot path
        ///   - pairMax is stackalloc'd (104 bytes for inputW=26) — zero heap alloc
        /// </summary>
        private static void MaxPool2DForwardSingleBatchPool2NoIndex(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW,
            int outH,
            int outW)
        {
            Span<float> pairMax = inputW <= 128
                ? stackalloc float[inputW]
                : new float[inputW];

            for (var c = 0; c < channels; c++)
            {
                var inputChannelBase = c * inputH * inputW;
                var outputChannelBase = c * outH * outW;

                for (var oh = 0; oh < outH; oh++)
                {
                    var row0 = input.Slice(inputChannelBase + oh * 2 * inputW, inputW);
                    var row1 = input.Slice(inputChannelBase + (oh * 2 + 1) * inputW, inputW);

                    // Vertical max: for each column, keep the larger of the two rows.
                    TensorPrimitives.Max(row0, row1, pairMax);

                    // Horizontal max: collapse adjacent pairs → output pixels.
                    var outRowBase = outputChannelBase + oh * outW;
                    for (var ow = 0; ow < outW; ow++)
                    {
                        var a = pairMax[ow * 2];
                        var b = pairMax[ow * 2 + 1];
                        output[outRowBase + ow] = a > b ? a : b;
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Private: training paths (with index tracking)
        // ─────────────────────────────────────────────────────────────────────

        private static void MaxPool2DForwardWithIndicesPool2(
            ReadOnlySpan<float> input,
            Span<float> output,
            Span<float> maxIndices,
            int channels,
            int inputH,
            int inputW,
            int outH,
            int outW,
            int batchOffset)
        {
            Span<float> pairMax = inputW <= 128
                ? stackalloc float[inputW]
                : new float[inputW];

            for (var c = 0; c < channels; c++)
            {
                var inputChannelBase = c * inputH * inputW;
                var outputChannelBase = c * outH * outW;

                for (var oh = 0; oh < outH; oh++)
                {
                    var row0Start = inputChannelBase + oh * 2 * inputW;
                    var row1Start = inputChannelBase + (oh * 2 + 1) * inputW;

                    var row0 = input.Slice(row0Start, inputW);
                    var row1 = input.Slice(row1Start, inputW);

                    TensorPrimitives.Max(row0, row1, pairMax);

                    var outRowBase = outputChannelBase + oh * outW;

                    for (var ow = 0; ow < outW; ow++)
                    {
                        float a = pairMax[ow * 2];
                        float b = pairMax[ow * 2 + 1];

                        float maxVal;
                        int maxIdx;

                        if (a >= b)
                        {
                            // Horizontal winner is left column (ow*2).
                            // Vertical winner: whichever row had the larger value.
                            var idxInRow0 = row0Start + ow * 2;
                            var idxInRow1 = row1Start + ow * 2;
                            maxVal = a;
                            maxIdx = input[idxInRow0] >= input[idxInRow1] ? idxInRow0 : idxInRow1;
                        }
                        else
                        {
                            var idxInRow0 = row0Start + ow * 2 + 1;
                            var idxInRow1 = row1Start + ow * 2 + 1;
                            maxVal = b;
                            maxIdx = input[idxInRow0] >= input[idxInRow1] ? idxInRow0 : idxInRow1;
                        }

                        output[outRowBase + ow] = maxVal;
                        maxIndices[outRowBase + ow] = batchOffset + maxIdx;
                    }
                }
            }
        }

        private static void MaxPool2DForwardWithIndicesGeneric(
            ReadOnlySpan<float> input,
            Span<float> output,
            Span<float> maxIndices,
            int channels,
            int inputH,
            int inputW,
            int pool,
            int outH,
            int outW,
            int batchOffset)
        {
            for (var c = 0; c < channels; c++)
            {
                var inputChannelBase = c * inputH * inputW;
                var outputChannelBase = c * outH * outW;

                for (var oh = 0; oh < outH; oh++)
                {
                    for (var ow = 0; ow < outW; ow++)
                    {
                        var maxVal = float.MinValue;
                        var maxIdx = 0;

                        for (var ph = 0; ph < pool; ph++)
                        {
                            var rowBase = inputChannelBase + (oh * pool + ph) * inputW + ow * pool;
                            for (var pw = 0; pw < pool; pw++)
                            {
                                var idx = rowBase + pw;
                                var val = input[idx];
                                if (val > maxVal) { maxVal = val; maxIdx = idx; }
                            }
                        }

                        var outIdx = outputChannelBase + oh * outW + ow;
                        output[outIdx] = maxVal;
                        maxIndices[outIdx] = batchOffset + maxIdx;
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
                output[c] = TensorPrimitives.Sum(input.Slice(c * spatialSize, spatialSize)) * scale;
            }
        }
    }
}
