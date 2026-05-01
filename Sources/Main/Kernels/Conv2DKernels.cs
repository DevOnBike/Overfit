// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;

namespace DevOnBike.Overfit.Kernels
{
    internal static class Conv2DKernels
    {
        public static void ForwardValidNchw(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kernelSize);

            if (kernelSize > inputH || kernelSize > inputW)
            {
                throw new ArgumentException("Kernel size cannot be larger than input spatial dimensions.");
            }

            var outH = inputH - kernelSize + 1;
            var outW = inputW - kernelSize + 1;

            var inputSize = inChannels * inputH * inputW;
            var outputSize = outChannels * outH * outW;
            var kernelSizePerOutput = inChannels * kernelSize * kernelSize;

            if (input.Length % inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by Conv2D input size.",
                    nameof(input));
            }

            if (kernels.Length < outChannels * kernelSizePerOutput)
            {
                throw new ArgumentException(
                    "Kernel span is too small.",
                    nameof(kernels));
            }

            var batchSize = input.Length / inputSize;
            var expectedOutputLength = batchSize * outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException(
                    "Output span is too small for Conv2D inference.",
                    nameof(output));
            }

            for (var n = 0; n < batchSize; n++)
            {
                var inputBatch = input.Slice(n * inputSize, inputSize);
                var outputBatch = output.Slice(n * outputSize, outputSize);

                if (inChannels == 1 && kernelSize == 3)
                {
                    ForwardValidSingleChannel3x3(
                        inputBatch,
                        kernels,
                        outputBatch,
                        outChannels,
                        inputH,
                        inputW,
                        outH,
                        outW);
                }
                else
                {
                    ForwardValidGenericSingleBatch(
                        inputBatch,
                        kernels,
                        outputBatch,
                        inChannels,
                        outChannels,
                        inputH,
                        inputW,
                        kernelSize,
                        outH,
                        outW,
                        kernelSizePerOutput);
                }
            }
        }

        private static void ForwardValidSingleChannel3x3(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int outChannels,
            int inputH,
            int inputW,
            int outH,
            int outW)
        {
            if (Vector.IsHardwareAccelerated && outW >= Vector<float>.Count)
            {
                ForwardValidSingleChannel3x3Vectorized(
                    input,
                    kernels,
                    output,
                    outChannels,
                    inputW,
                    outH,
                    outW);

                return;
            }

            ForwardValidSingleChannel3x3Scalar(
                input,
                kernels,
                output,
                outChannels,
                inputW,
                outH,
                outW);
        }

        private static void ForwardValidSingleChannel3x3Vectorized(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int outChannels,
            int inputW,
            int outH,
            int outW)
        {
            var vectorWidth = Vector<float>.Count;

            for (var oc = 0; oc < outChannels; oc++)
            {
                var kernelBase = oc * 9;
                var outputChannelBase = oc * outH * outW;

                var k00 = new Vector<float>(kernels[kernelBase + 0]);
                var k01 = new Vector<float>(kernels[kernelBase + 1]);
                var k02 = new Vector<float>(kernels[kernelBase + 2]);
                var k10 = new Vector<float>(kernels[kernelBase + 3]);
                var k11 = new Vector<float>(kernels[kernelBase + 4]);
                var k12 = new Vector<float>(kernels[kernelBase + 5]);
                var k20 = new Vector<float>(kernels[kernelBase + 6]);
                var k21 = new Vector<float>(kernels[kernelBase + 7]);
                var k22 = new Vector<float>(kernels[kernelBase + 8]);

                for (var oy = 0; oy < outH; oy++)
                {
                    var inputRow0 = oy * inputW;
                    var inputRow1 = (oy + 1) * inputW;
                    var inputRow2 = (oy + 2) * inputW;
                    var outputRow = outputChannelBase + oy * outW;

                    var ox = 0;

                    for (; ox <= outW - vectorWidth; ox += vectorWidth)
                    {
                        var acc =
                            new Vector<float>(input.Slice(inputRow0 + ox, vectorWidth)) * k00 +
                            new Vector<float>(input.Slice(inputRow0 + ox + 1, vectorWidth)) * k01 +
                            new Vector<float>(input.Slice(inputRow0 + ox + 2, vectorWidth)) * k02 +
                            new Vector<float>(input.Slice(inputRow1 + ox, vectorWidth)) * k10 +
                            new Vector<float>(input.Slice(inputRow1 + ox + 1, vectorWidth)) * k11 +
                            new Vector<float>(input.Slice(inputRow1 + ox + 2, vectorWidth)) * k12 +
                            new Vector<float>(input.Slice(inputRow2 + ox, vectorWidth)) * k20 +
                            new Vector<float>(input.Slice(inputRow2 + ox + 1, vectorWidth)) * k21 +
                            new Vector<float>(input.Slice(inputRow2 + ox + 2, vectorWidth)) * k22;

                        acc.CopyTo(output.Slice(outputRow + ox, vectorWidth));
                    }

                    for (; ox < outW; ox++)
                    {
                        output[outputRow + ox] =
                            input[inputRow0 + ox] * kernels[kernelBase + 0] +
                            input[inputRow0 + ox + 1] * kernels[kernelBase + 1] +
                            input[inputRow0 + ox + 2] * kernels[kernelBase + 2] +
                            input[inputRow1 + ox] * kernels[kernelBase + 3] +
                            input[inputRow1 + ox + 1] * kernels[kernelBase + 4] +
                            input[inputRow1 + ox + 2] * kernels[kernelBase + 5] +
                            input[inputRow2 + ox] * kernels[kernelBase + 6] +
                            input[inputRow2 + ox + 1] * kernels[kernelBase + 7] +
                            input[inputRow2 + ox + 2] * kernels[kernelBase + 8];
                    }
                }
            }
        }

        private static void ForwardValidSingleChannel3x3Scalar(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int outChannels,
            int inputW,
            int outH,
            int outW)
        {
            for (var oc = 0; oc < outChannels; oc++)
            {
                var kernelBase = oc * 9;
                var outputChannelBase = oc * outH * outW;

                var k00 = kernels[kernelBase + 0];
                var k01 = kernels[kernelBase + 1];
                var k02 = kernels[kernelBase + 2];
                var k10 = kernels[kernelBase + 3];
                var k11 = kernels[kernelBase + 4];
                var k12 = kernels[kernelBase + 5];
                var k20 = kernels[kernelBase + 6];
                var k21 = kernels[kernelBase + 7];
                var k22 = kernels[kernelBase + 8];

                for (var oy = 0; oy < outH; oy++)
                {
                    var inputRow0 = oy * inputW;
                    var inputRow1 = (oy + 1) * inputW;
                    var inputRow2 = (oy + 2) * inputW;
                    var outputRow = outputChannelBase + oy * outW;

                    for (var ox = 0; ox < outW; ox++)
                    {
                        output[outputRow + ox] =
                            input[inputRow0 + ox] * k00 +
                            input[inputRow0 + ox + 1] * k01 +
                            input[inputRow0 + ox + 2] * k02 +
                            input[inputRow1 + ox] * k10 +
                            input[inputRow1 + ox + 1] * k11 +
                            input[inputRow1 + ox + 2] * k12 +
                            input[inputRow2 + ox] * k20 +
                            input[inputRow2 + ox + 1] * k21 +
                            input[inputRow2 + ox + 2] * k22;
                    }
                }
            }
        }

        private static void ForwardValidGenericSingleBatch(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize,
            int outH,
            int outW,
            int kernelSizePerOutput)
        {
            for (var oc = 0; oc < outChannels; oc++)
            {
                var kernelBase = oc * kernelSizePerOutput;
                var outputChannelBase = oc * outH * outW;

                for (var oy = 0; oy < outH; oy++)
                {
                    for (var ox = 0; ox < outW; ox++)
                    {
                        var sum = 0f;

                        for (var ic = 0; ic < inChannels; ic++)
                        {
                            var inputChannelBase = ic * inputH * inputW;
                            var kernelChannelBase = kernelBase + ic * kernelSize * kernelSize;

                            for (var ky = 0; ky < kernelSize; ky++)
                            {
                                var inputRowBase = inputChannelBase + (oy + ky) * inputW + ox;
                                var kernelRowBase = kernelChannelBase + ky * kernelSize;

                                for (var kx = 0; kx < kernelSize; kx++)
                                {
                                    sum += input[inputRowBase + kx] * kernels[kernelRowBase + kx];
                                }
                            }
                        }

                        output[outputChannelBase + oy * outW + ox] = sum;
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // ForwardNchw — general Conv2D with padding and stride (ONNX-4)
        //
        // Replaces ForwardValidNchw for ONNX-imported models that use padding/stride.
        // ForwardValidNchw (padding=0, stride=1) is kept as-is for training path
        // backward compatibility.
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// General 2-D convolution forward pass (NCHW layout).
        /// Supports symmetric zero-padding and stride ≥ 1.
        /// Square kernel assumed (kH == kW == kernelSize).
        ///
        /// Output dimensions:
        ///   outH = (inputH + 2*padding - kernelSize) / stride + 1
        ///   outW = (inputW + 2*padding - kernelSize) / stride + 1
        /// </summary>
        public static void ForwardNchw(
            ReadOnlySpan<float> input,   // [batch, inC, inputH, inputW]
            ReadOnlySpan<float> kernels, // [outC, inC, kSize, kSize]
            Span<float> output,          // [batch, outC, outH, outW]
            int batchSize,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride)
        {
            var outH = (inputH + 2 * padding - kernelSize) / stride + 1;
            var outW = (inputW + 2 * padding - kernelSize) / stride + 1;
            var inputPlaneSize  = inChannels  * inputH * inputW;
            var outputPlaneSize = outChannels * outH   * outW;
            var kernelSizePerOutput = inChannels * kernelSize * kernelSize;

            output.Clear();

            for (var n = 0; n < batchSize; n++)
            {
                var inputBatch  = input.Slice(n * inputPlaneSize,  inputPlaneSize);
                var outputBatch = output.Slice(n * outputPlaneSize, outputPlaneSize);

                ForwardNchwSingleBatch(
                    inputBatch, kernels, outputBatch,
                    inChannels, outChannels,
                    inputH, inputW,
                    kernelSize, outH, outW,
                    kernelSizePerOutput, padding, stride);
            }
        }

        private static void ForwardNchwSingleBatch(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize,
            int outH,
            int outW,
            int kernelSizePerOutput,
            int padding,
            int stride)
        {
            for (var oc = 0; oc < outChannels; oc++)
            {
                var kernelBase      = oc * kernelSizePerOutput;
                var outputChanBase  = oc * outH * outW;

                for (var oy = 0; oy < outH; oy++)
                {
                    var inputYBase = oy * stride - padding;

                    for (var ox = 0; ox < outW; ox++)
                    {
                        var inputXBase = ox * stride - padding;
                        var sum = 0f;

                        for (var ic = 0; ic < inChannels; ic++)
                        {
                            var inputChanBase  = ic * inputH * inputW;
                            var kernelChanBase = kernelBase + ic * kernelSize * kernelSize;

                            for (var ky = 0; ky < kernelSize; ky++)
                            {
                                var iy = inputYBase + ky;
                                if ((uint)iy >= (uint)inputH)
                                {
                                    continue; // zero-pad: skip out-of-bounds rows
                                }

                                var kernelRowBase = kernelChanBase + ky * kernelSize;
                                var inputRowBase  = inputChanBase  + iy * inputW;

                                for (var kx = 0; kx < kernelSize; kx++)
                                {
                                    var ix = inputXBase + kx;
                                    if ((uint)ix >= (uint)inputW)
                                    {
                                        continue; // zero-pad: skip out-of-bounds cols
                                    }

                                    sum += input[inputRowBase + ix] * kernels[kernelRowBase + kx];
                                }
                            }
                        }

                        output[outputChanBase + oy * outW + ox] = sum;
                    }
                }
            }
        }

    }
}