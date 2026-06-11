// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Batched F32 projection (prefill Phase 1): <c>output[N×O] = bias + input[N×I] @ W[I×O]</c>,
    /// the multi-token counterpart of <see cref="SingleTokenProjectionKernel"/>.
    ///
    /// <para>
    /// Weight layout is input-major (<c>W[i*O + o]</c>) — identical to the
    /// single-token kernel. The loop order is <b>output-tile → input → row</b>: a
    /// weight tile is loaded once and reused across all <c>N</c> rows before the
    /// next input element. That amortises the weight-read bandwidth by <c>N</c> —
    /// the whole point of batched prefill (decode reads the weight set once per
    /// token; prefill of <c>N</c> prompt tokens now reads it once per <c>N</c>).
    /// </para>
    ///
    /// <para>
    /// Per output element the accumulation order is input-ascending and uses the
    /// same <c>TensorPrimitives.MultiplyAdd</c> + <c>x == 0</c> skip as the
    /// single-token kernel, so each output row is <b>bit-identical</b> to
    /// <c>SingleTokenProjectionKernel.Project</c> of that input row. (Verified by
    /// <c>BatchedProjectionKernelTests</c>.)
    /// </para>
    /// </summary>
    public static unsafe class BatchedProjectionKernel
    {
        /// <summary>
        /// L1-resident output-tile width (floats). Matches the single-token kernel
        /// so blocking does not change the accumulation order. 2048 floats = 8 KB.
        /// </summary>
        private const int OutputTile = 2048;

        /// <summary>
        /// Sequential batched projection. <paramref name="input"/> is row-major
        /// <c>[rows × inputSize]</c>, <paramref name="output"/> row-major
        /// <c>[rows × outputSize]</c>, <paramref name="weights"/> input-major
        /// <c>[inputSize × outputSize]</c>. <paramref name="bias"/> may be empty.
        /// </summary>
        public static void Project(
            ReadOnlySpan<float> input,
            int rows,
            ReadOnlySpan<float> weights,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            Validate(input, rows, weights, bias, output, inputSize, outputSize);

            for (var tile = 0; tile < outputSize; tile += OutputTile)
            {
                var len = Math.Min(OutputTile, outputSize - tile);
                ProjectTile(input, rows, weights, bias, output, inputSize, outputSize, tile, len);
            }
        }

        /// <summary>
        /// Parallel batched projection — the output dimension is split across the
        /// zero-allocation <c>OverfitParallel</c> pool (each worker owns a
        /// disjoint band of output columns for every row, so there are no
        /// cross-worker writes). Bit-identical to <see cref="Project"/>.
        /// </summary>
        public static void ProjectParallel(
            ReadOnlySpan<float> input,
            int rows,
            ReadOnlySpan<float> weights,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            Validate(input, rows, weights, bias, output, inputSize, outputSize);

            fixed (float* inputPtr = input)
            fixed (float* weightsPtr = weights)
            fixed (float* biasPtr = bias)
            fixed (float* outputPtr = output)
            {
                var context = new BatchedContext
                {
                    Input = inputPtr,
                    Weights = weightsPtr,
                    Bias = biasPtr,
                    BiasLength = bias.Length,
                    Output = outputPtr,
                    Rows = rows,
                    InputSize = inputSize,
                    OutputSize = outputSize,
                };

                OverfitParallel.For(0, outputSize, &ProjectColumnRange, &context);
            }
        }

        /// <summary>Worker body — one disjoint band <c>[outStart, outEnd)</c> of output columns, all rows.</summary>
        private static void ProjectColumnRange(int outStart, int outEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<BatchedContext>(context);
            var outputSize = ctx.OutputSize;
            var inputSize = ctx.InputSize;
            var rows = ctx.Rows;

            // Tile the worker's band into L1-resident chunks (same width as the
            // sequential path) so the weight tile stays hot across the row loop.
            for (var tile = outStart; tile < outEnd; tile += OutputTile)
            {
                var len = Math.Min(OutputTile, outEnd - tile);

                var biasSpan = ctx.BiasLength == 0
                    ? ReadOnlySpan<float>.Empty
                    : new ReadOnlySpan<float>(ctx.Bias, ctx.BiasLength);

                var input = new ReadOnlySpan<float>(ctx.Input, rows * inputSize);
                var weights = new ReadOnlySpan<float>(ctx.Weights, inputSize * outputSize);
                var output = new Span<float>(ctx.Output, rows * outputSize);

                ProjectTile(input, rows, weights, biasSpan, output, inputSize, outputSize, tile, len);
            }
        }

        /// <summary>
        /// Projects one output tile <c>[tile, tile+len)</c> for every row. Loads each
        /// input element's weight tile once and applies it across all rows.
        /// </summary>
        private static void ProjectTile(
            ReadOnlySpan<float> input,
            int rows,
            ReadOnlySpan<float> weights,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize,
            int tile,
            int len)
        {
            // Initialise each row's output tile to bias (or zero).
            for (var n = 0; n < rows; n++)
            {
                var outRow = output.Slice(n * outputSize + tile, len);
                if (bias.IsEmpty)
                {
                    outRow.Clear();
                }
                else
                {
                    bias.Slice(tile, len).CopyTo(outRow);
                }
            }

            for (var i = 0; i < inputSize; i++)
            {
                var weightTile = weights.Slice(i * outputSize + tile, len);

                for (var n = 0; n < rows; n++)
                {
                    var x = input[n * inputSize + i];
                    if (x == 0f)
                    {
                        continue;
                    }

                    var outRow = output.Slice(n * outputSize + tile, len);
                    TensorPrimitives.MultiplyAdd(weightTile, x, outRow, outRow);
                }
            }
        }

        private static void Validate(
            ReadOnlySpan<float> input,
            int rows,
            ReadOnlySpan<float> weights,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            if (input.Length < (long)rows * inputSize)
            {
                throw new ArgumentException($"Input span {input.Length} < rows*inputSize {(long)rows * inputSize}.", nameof(input));
            }
            if (weights.Length < (long)inputSize * outputSize)
            {
                throw new ArgumentException($"Weights span {weights.Length} < inputSize*outputSize {(long)inputSize * outputSize}.", nameof(weights));
            }
            if (output.Length < (long)rows * outputSize)
            {
                throw new ArgumentException($"Output span {output.Length} < rows*outputSize {(long)rows * outputSize}.", nameof(output));
            }
            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException("Bias span is smaller than outputSize.", nameof(bias));
            }
        }

        private struct BatchedContext
        {
            public float* Input;
            public float* Weights;
            public float* Bias;
            public int BiasLength;
            public float* Output;
            public int Rows;
            public int InputSize;
            public int OutputSize;
        }
    }
}
