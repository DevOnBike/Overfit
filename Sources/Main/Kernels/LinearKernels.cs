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

        // ─────────────────────────────────────────────────────────────────────
        // Backward kernels — span-only, no AutogradNode dependency
        //
        // ParallelThreshold: 1_048_576 (1M ops). Much higher than TensorMath's 4096.
        // Rationale: Parallel.For has ~2-5 µs overhead + TPL allocation (~1-3 KB).
        // For small matrices (e.g., Linear(8,10) backward: 64×10×8 = 5120 ops)
        // sequential SIMD via TensorPrimitives.Dot is 10-50× faster than Parallel.For.
        // For large matrices (e.g., Linear(1352,64) backward: 5.5M ops) parallelism wins.
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Computes the input gradient:
        ///   gradInput[b, i] += sum_j(gradOutput[b, j] * weights[i, j])
        ///
        /// Equivalent to: gradInput += gradOutput @ weights^T
        /// Layout: weights[inputSize, outputSize] (input-major, same as training path).
        /// </summary>
        public static void BackwardInput(
            ReadOnlySpan<float> gradOutput,    // [batchSize, outputSize]
            ReadOnlySpan<float> weights,        // [inputSize, outputSize]
            Span<float> gradInput,              // [batchSize, inputSize]  — accumulated in-place
            int batchSize,
            int inputSize,
            int outputSize)
        {
            // Sequential SIMD: Span<T> is a ref struct and cannot be captured in Parallel.For lambdas.
            // TensorPrimitives.Dot inside BackwardInputRow uses AVX-512 on supported hardware,
            // so sequential is already highly vectorised. Parallelism could be added via unsafe
            // fixed pointers if profiling shows sequential to be a bottleneck.
            for (var b = 0; b < batchSize; b++)
            {
                BackwardInputRow(
                    gradOutput.Slice(b * outputSize, outputSize),
                    weights,
                    gradInput.Slice(b * inputSize, inputSize),
                    inputSize,
                    outputSize);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void BackwardInputRow(
            ReadOnlySpan<float> gradOutputRow,   // [outputSize]
            ReadOnlySpan<float> weights,          // [inputSize, outputSize]
            Span<float> gradInputRow,             // [inputSize]
            int inputSize,
            int outputSize)
        {
            // gradInput[i] += Dot(gradOutput, weights[i, :])
            // weights[i, :] = weights.Slice(i * outputSize, outputSize)
            for (var i = 0; i < inputSize; i++)
            {
                gradInputRow[i] += TensorPrimitives.Dot(
                    gradOutputRow,
                    weights.Slice(i * outputSize, outputSize));
            }
        }

        /// <summary>
        /// Accumulates weight gradients:
        ///   gradWeights[i, j] += sum_b(input[b, i] * gradOutput[b, j])
        ///
        /// Equivalent to: gradWeights += input^T @ gradOutput
        /// Layout: gradWeights[inputSize, outputSize] (input-major).
        /// </summary>
        public static void AccumulateWeightGrad(
            ReadOnlySpan<float> input,         // [batchSize, inputSize]
            ReadOnlySpan<float> gradOutput,    // [batchSize, outputSize]
            Span<float> gradWeights,           // [inputSize, outputSize]  — accumulated in-place
            int batchSize,
            int inputSize,
            int outputSize)
        {
            // Outer-product accumulation: for each batch element, for each input dim,
            // add input[b,i] * gradOutput[b,:] to gradWeights[i,:].
            // Sequential is almost always correct here — parallelising requires per-row
            // reduction which adds complexity. For the common case (inputSize=1352,
            // outputSize=64, batchSize=64) the work is 5.5M adds — fast enough sequentially.
            // Adding a parallel path would require separate partial gradient buffers per thread
            // (as Conv2DWorkspace does) — out of scope for PERF-1.
            for (var b = 0; b < batchSize; b++)
            {
                var inRow = input.Slice(b * inputSize, inputSize);
                var gO = gradOutput.Slice(b * outputSize, outputSize);

                for (var i = 0; i < inputSize; i++)
                {
                    var xi = inRow[i];
                    var wRow = gradWeights.Slice(i * outputSize, outputSize);

                    // gradWeights[i, :] += xi * gradOutput[b, :]
                    TensorPrimitives.MultiplyAdd(gO, xi, wRow, wRow);
                }
            }
        }

        /// <summary>
        /// Accumulates bias gradients:
        ///   gradBias[j] += sum_b(gradOutput[b, j])
        ///
        /// Always sequential — bias is small (outputSize elements).
        /// </summary>
        public static void AccumulateBiasGrad(
            ReadOnlySpan<float> gradOutput,    // [batchSize, outputSize]
            Span<float> gradBias,              // [outputSize]  — accumulated in-place
            int batchSize,
            int outputSize)
        {
            for (var b = 0; b < batchSize; b++)
            {
                TensorPrimitives.Add(
                    gradOutput.Slice(b * outputSize, outputSize),
                    gradBias,
                    gradBias);
            }
        }


        // ─────────────────────────────────────────────────────────────────────
        // ForwardBatched — weight-stationary outer-product, no zero-skipping
        //
        // Layout: weights[inputSize, outputSize] (input-major).
        //
        // Algorithm: for each input feature k, broadcast W[k,:] across all batch
        // rows simultaneously. W[k,:] stays in L1 cache while all B rows are
        // processed. Eliminates zero-skipping branch mispredictions (post-ReLU
        // activations are ~50% zero → ~50% misprediction rate on the skip branch).
        //
        // Routing in TensorMath.Linear:
        //   batchSize * inputSize * outputSize < ForwardBatchedThreshold
        //     → ForwardBatched sequential (weight-stationary)
        //   otherwise
        //     → ForwardBatched + Parallel.For over K-chunks (TODO) or old path
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Threshold above which <see cref="ForwardBatched"/> switches to the
        /// parallel path in <see cref="DevOnBike.Overfit.Ops.TensorMath"/>.
        /// 500_000 chosen so that:
        ///   - Linear(64,10)   batch=64: 64*64*10 = 40K → sequential ✓
        ///   - Linear(1352,64) batch=64: 64*1352*64 = 5.5M → parallel ✓
        /// </summary>
        internal const long ForwardBatchedThreshold = 500_000;

        /// <summary>
        /// Batched forward pass without zero-skipping.
        /// Caller is responsible for initialising <paramref name="output"/> with
        /// the bias before calling (use <see cref="InitWithBias"/>).
        /// </summary>
        public static void ForwardBatched(
            ReadOnlySpan<float> input,    // [batchSize, inputSize]
            ReadOnlySpan<float> weights,  // [inputSize, outputSize] input-major
            Span<float> output,           // [batchSize, outputSize]  — bias pre-filled
            int batchSize,
            int inputSize,
            int outputSize)
        {
            // Weight-stationary outer product:
            //   for each input feature k:
            //     for each batch row b:
            //       output[b,:] += input[b,k] * W[k,:]
            //
            // W[k,:] (outputSize floats) is loaded once per k and reused for all B rows.
            // For outputSize=64 (256 bytes) this fits in L1 cache across all B=64 iterations.
            for (var k = 0; k < inputSize; k++)
            {
                var wRow = weights.Slice(k * outputSize, outputSize);

                for (var b = 0; b < batchSize; b++)
                {
                    var xi = input[b * inputSize + k];
                    TensorPrimitives.MultiplyAdd(
                        wRow,
                        xi,
                        output.Slice(b * outputSize, outputSize),
                        output.Slice(b * outputSize, outputSize));
                }
            }
        }

        /// <summary>
        /// Initialises the output buffer with bias, broadcasted across batchSize rows.
        /// Call before <see cref="ForwardBatched"/>.
        /// </summary>
        public static void InitWithBias(
            Span<float> output,
            ReadOnlySpan<float> bias,
            int batchSize,
            int outputSize)
        {
            for (var b = 0; b < batchSize; b++)
            {
                bias.CopyTo(output.Slice(b * outputSize, outputSize));
            }
        }

    }
}