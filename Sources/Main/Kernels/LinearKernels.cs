// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Kernels
{
    internal static class LinearKernels
    {
        private const int InputMajorVectorizedOutputThreshold = 32;

        public static void TransposeInputOutputToOutputInput(
            ReadOnlySpan<float> sourceInputOutput,
            Span<float> destinationOutputInput,
            int inputSize,
            int outputSize)
        {
            if (sourceInputOutput.Length < inputSize * outputSize)
            {
                throw new ArgumentException("Source weights span is too small.", nameof(sourceInputOutput));
            }

            if (destinationOutputInput.Length < inputSize * outputSize)
            {
                throw new ArgumentException("Destination weights span is too small.", nameof(destinationOutputInput));
            }

            for (var i = 0; i < inputSize; i++)
            {
                var srcBase = i * outputSize;

                for (var j = 0; j < outputSize; j++)
                {
                    destinationOutputInput[j * inputSize + i] = sourceInputOutput[srcBase + j];
                }
            }
        }

        public static void Forward(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> weightsOutputInput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (inputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputSize));
            }

            if (outputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputSize));
            }

            if (input.Length % inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by layer input size.",
                    nameof(input));
            }

            if (weightsInputOutput.Length < inputSize * outputSize)
            {
                throw new ArgumentException(
                    "Input-major weights span is too small.",
                    nameof(weightsInputOutput));
            }

            if (weightsOutputInput.Length < inputSize * outputSize)
            {
                throw new ArgumentException(
                    "Output-major weights span is too small.",
                    nameof(weightsOutputInput));
            }

            if (bias.Length < outputSize)
            {
                throw new ArgumentException(
                    "Bias span is too small.",
                    nameof(bias));
            }

            var batchSize = input.Length / inputSize;
            var expectedOutputLength = batchSize * outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException(
                    "Output span is too small for Linear inference.",
                    nameof(output));
            }

            for (var b = 0; b < batchSize; b++)
            {
                var inSlice = input.Slice(b * inputSize, inputSize);
                var outSlice = output.Slice(b * outputSize, outputSize);

                if (outputSize >= InputMajorVectorizedOutputThreshold)
                {
                    ForwardInputMajorVector4(
                        inSlice,
                        weightsInputOutput,
                        bias,
                        outSlice,
                        inputSize,
                        outputSize);
                }
                else
                {
                    ForwardOutputMajorDot(
                        inSlice,
                        weightsOutputInput,
                        bias,
                        outSlice);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ForwardOutputMajorDot(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsOutputInput,
            ReadOnlySpan<float> bias,
            Span<float> output)
        {
            var inputSize = input.Length;
            var outputSize = output.Length;

            for (var j = 0; j < outputSize; j++)
            {
                var wRow = weightsOutputInput.Slice(j * inputSize, inputSize);
                output[j] = TensorPrimitives.Dot(input, wRow) + bias[j];
            }
        }

        private static void ForwardInputMajorVector4(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (!Vector.IsHardwareAccelerated ||
                outputSize < Vector<float>.Count * 4)
            {
                ForwardInputMajorVector1(
                    input,
                    weightsInputOutput,
                    bias,
                    output,
                    inputSize,
                    outputSize);

                return;
            }

            var vectorWidth = Vector<float>.Count;
            var blockWidth = vectorWidth * 4;

            var j = 0;

            for (; j <= outputSize - blockWidth; j += blockWidth)
            {
                var acc0 = new Vector<float>(bias.Slice(j, vectorWidth));
                var acc1 = new Vector<float>(bias.Slice(j + vectorWidth, vectorWidth));
                var acc2 = new Vector<float>(bias.Slice(j + vectorWidth * 2, vectorWidth));
                var acc3 = new Vector<float>(bias.Slice(j + vectorWidth * 3, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    var x = new Vector<float>(input[i]);
                    var rowBase = i * outputSize + j;

                    acc0 += x * new Vector<float>(weightsInputOutput.Slice(rowBase, vectorWidth));
                    acc1 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth, vectorWidth));
                    acc2 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth * 2, vectorWidth));
                    acc3 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth * 3, vectorWidth));
                }

                acc0.CopyTo(output.Slice(j, vectorWidth));
                acc1.CopyTo(output.Slice(j + vectorWidth, vectorWidth));
                acc2.CopyTo(output.Slice(j + vectorWidth * 2, vectorWidth));
                acc3.CopyTo(output.Slice(j + vectorWidth * 3, vectorWidth));
            }

            for (; j <= outputSize - vectorWidth; j += vectorWidth)
            {
                var acc = new Vector<float>(bias.Slice(j, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    acc += new Vector<float>(input[i]) *
                           new Vector<float>(weightsInputOutput.Slice(i * outputSize + j, vectorWidth));
                }

                acc.CopyTo(output.Slice(j, vectorWidth));
            }

            for (; j < outputSize; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputOutput[i * outputSize + j];
                }

                output[j] = sum;
            }
        }

        private static void ForwardInputMajorVector1(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (!Vector.IsHardwareAccelerated ||
                outputSize < Vector<float>.Count)
            {
                ForwardInputMajorScalar(
                    input,
                    weightsInputOutput,
                    bias,
                    output,
                    inputSize,
                    outputSize);

                return;
            }

            var vectorWidth = Vector<float>.Count;
            var j = 0;

            for (; j <= outputSize - vectorWidth; j += vectorWidth)
            {
                var acc = new Vector<float>(bias.Slice(j, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    var x = new Vector<float>(input[i]);
                    var w = new Vector<float>(weightsInputOutput.Slice(i * outputSize + j, vectorWidth));

                    acc += x * w;
                }

                acc.CopyTo(output.Slice(j, vectorWidth));
            }

            for (; j < outputSize; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputOutput[i * outputSize + j];
                }

                output[j] = sum;
            }
        }

        private static void ForwardInputMajorScalar(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            bias.Slice(0, outputSize).CopyTo(output);

            for (var i = 0; i < inputSize; i++)
            {
                var x = input[i];
                var wBase = i * outputSize;

                for (var j = 0; j < outputSize; j++)
                {
                    output[j] += x * weightsInputOutput[wBase + j];
                }
            }
        }
    }
}