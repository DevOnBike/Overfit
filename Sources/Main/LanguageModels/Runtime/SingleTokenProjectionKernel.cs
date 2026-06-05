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
    /// Allocation-free single-token linear projection kernel for autoregressive SLM decode.
    ///
    /// This is the building block needed for cached GPT-style inference:
    ///
    /// - Q projection:  x[token] @ Wq
    /// - K projection:  x[token] @ Wk
    /// - V projection:  x[token] @ Wv
    /// - O projection:  attentionOutput @ Wo
    /// - LM head:       hidden @ W_vocab
    ///
    /// Layout:
    ///
    /// input:   [inputSize]
    /// weights: [inputSize, outputSize] in input-major order
    /// bias:    [outputSize], optional
    /// output:  [outputSize]
    ///
    /// The caller owns all buffers. This kernel does not allocate.
    ///
    /// Implementation note:
    /// This version uses TensorPrimitives.MultiplyAdd over output rows:
    ///
    /// output[:] += input[i] * weights[i, :]
    ///
    /// That keeps the public API unchanged, but lets the runtime use vectorized
    /// span primitives for the large output dimension cases that matter for
    /// cached decode, especially:
    ///
    /// - 768 -> 768 projections,
    /// - 768 -> 40478 LM head.
    /// </summary>
    public static unsafe class SingleTokenProjectionKernel
    {
        public static void Project(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            ValidateArguments(
                input,
                weightsInputOutput,
                output,
                inputSize,
                outputSize);

            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException("Bias span is smaller than outputSize.", nameof(bias));
            }

            if (bias.IsEmpty)
            {
                output.Slice(0, outputSize).Clear();
            }
            else
            {
                bias.Slice(0, outputSize).CopyTo(output);
            }

            Accumulate(
                input,
                weightsInputOutput,
                output,
                inputSize,
                outputSize);
        }

        public static void ProjectWithoutBias(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            Project(
                input,
                weightsInputOutput,
                [],
                output,
                inputSize,
                outputSize);
        }

        /// <summary>
        /// Output-tile width (in floats) for the blocked accumulate. Large output
        /// dimensions (FFN, LM head) are processed one L1-resident tile at a time so
        /// the tile stays hot across the whole input loop — instead of the entire
        /// output vector being streamed through L2/L3 once per input element.
        /// Outputs at or below this size take a single tile, identical to an
        /// unblocked accumulate. 2048 floats = 8 KB; tunable.
        /// </summary>
        private const int AccumulateOutputTile = 2048;

        public static void Accumulate(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            ValidateArguments(
                input,
                weightsInputOutput,
                output,
                inputSize,
                outputSize);

            // Blocked over the output dimension: for each output tile, run the full
            // input loop while the tile stays resident in L1. The per-element
            // accumulation order (input ascending) is unchanged, so the result is
            // bit-identical to an unblocked accumulate.
            for (var tile = 0; tile < outputSize; tile += AccumulateOutputTile)
            {
                var tileLength = Math.Min(AccumulateOutputTile, outputSize - tile);
                var outputTile = output.Slice(tile, tileLength);

                for (var i = 0; i < inputSize; i++)
                {
                    var x = input[i];

                    if (x == 0f)
                    {
                        continue;
                    }

                    var weightTile = weightsInputOutput.Slice(
                        i * outputSize + tile,
                        tileLength);

                    TensorPrimitives.MultiplyAdd(
                        weightTile,
                        x,
                        outputTile,
                        outputTile);
                }
            }
        }

        /// <summary>
        /// Minimum matmul work (<c>inputSize × outputSize</c>) for the parallel
        /// path to pay off. Below this the <see cref="OverfitParallelFor"/>
        /// dispatch (~5 µs) outweighs the gain and the sequential
        /// <see cref="Project"/> runs instead. Tuned so FFN / LM-head matmuls
        /// parallelise while the small per-head attention projections stay
        /// sequential (they are better parallelised head-wise, one level up).
        /// </summary>
        public const long ParallelWorkThreshold = 1_000_000;

        /// <summary>
        /// Parallel projection for large matmuls (FFN, LM head). Splits the
        /// output dimension into one contiguous band per worker via the
        /// zero-allocation <see cref="OverfitParallelFor"/> dispatcher — each
        /// worker streams its own slice of the weight matrix from DRAM, so the
        /// decode matmul uses aggregate multi-core memory bandwidth instead of a
        /// single core. Output bands are disjoint: no reduction, no false
        /// sharing.
        ///
        /// Bit-identical to <see cref="Project"/>: every output element is
        /// accumulated by exactly one worker in input-ascending order.
        ///
        /// Below <see cref="ParallelWorkThreshold"/> (or with a single worker)
        /// it falls back to the sequential <see cref="Project"/> path. The happy
        /// path allocates 0 managed bytes.
        /// </summary>
        public static void ProjectParallel(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            ValidateArguments(input, weightsInputOutput, output, inputSize, outputSize);

            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException("Bias span is smaller than outputSize.", nameof(bias));
            }

            if ((long)inputSize * outputSize < ParallelWorkThreshold
                || OverfitParallelFor.WorkerCount <= 1)
            {
                Project(input, weightsInputOutput, bias, output, inputSize, outputSize);
                return;
            }

            fixed (float* pInput = input)
            fixed (float* pWeights = weightsInputOutput)
            fixed (float* pBias = bias)
            fixed (float* pOutput = output)
            {
                var context = new ProjectChunkContext
                {
                    Input = pInput,
                    Weights = pWeights,
                    Bias = pBias,
                    BiasLength = bias.Length,
                    Output = pOutput,
                    InputSize = inputSize,
                    OutputSize = outputSize,
                };

                // For() is synchronous — it blocks until every band completes,
                // so the fixed pointers stay valid for the whole dispatch.
                OverfitParallelFor.For(0, outputSize, &ProjectChunk, &context);
            }
        }

        /// <summary>
        /// Worker body for <see cref="ProjectParallel"/>: computes the output
        /// band <c>[chunkStart, chunkEnd)</c> — bias/zero init then a blocked
        /// accumulate over the band, mirroring <see cref="Accumulate"/>.
        /// </summary>
        private static void ProjectChunk(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<ProjectChunkContext>(context);

            var count = chunkEnd - chunkStart;
            var inputSize = ctx.InputSize;
            var outputSize = ctx.OutputSize;

            var outputBand = new Span<float>(ctx.Output + chunkStart, count);

            if (ctx.BiasLength == 0)
            {
                outputBand.Clear();
            }
            else
            {
                new ReadOnlySpan<float>(ctx.Bias + chunkStart, count).CopyTo(outputBand);
            }

            for (var tile = 0; tile < count; tile += AccumulateOutputTile)
            {
                var tileLength = Math.Min(AccumulateOutputTile, count - tile);
                var outputTile = outputBand.Slice(tile, tileLength);

                for (var i = 0; i < inputSize; i++)
                {
                    var x = ctx.Input[i];

                    if (x == 0f)
                    {
                        continue;
                    }

                    var weightTile = new ReadOnlySpan<float>(
                        ctx.Weights + (long)i * outputSize + chunkStart + tile,
                        tileLength);

                    TensorPrimitives.MultiplyAdd(weightTile, x, outputTile, outputTile);
                }
            }
        }

        /// <summary>Pinned-pointer state passed to <see cref="ProjectChunk"/> workers.</summary>
        private struct ProjectChunkContext
        {
            public float* Input;
            public float* Weights;
            public float* Bias;
            public int BiasLength;
            public float* Output;
            public int InputSize;
            public int OutputSize;
        }

        public static void ProjectSlice(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int fullOutputSize,
            int outputOffset,
            int outputCount)
        {
            ValidateSliceArguments(
                input,
                weightsInputOutput,
                output,
                inputSize,
                fullOutputSize,
                outputOffset,
                outputCount);

            if (!bias.IsEmpty && bias.Length < outputOffset + outputCount)
            {
                throw new ArgumentException("Bias span is too small for the requested output slice.", nameof(bias));
            }

            var outputSlice = output.Slice(0, outputCount);

            if (bias.IsEmpty)
            {
                outputSlice.Clear();
            }
            else
            {
                bias.Slice(outputOffset, outputCount).CopyTo(outputSlice);
            }

            for (var i = 0; i < inputSize; i++)
            {
                var x = input[i];

                if (x == 0f)
                {
                    continue;
                }

                var weightRow = weightsInputOutput.Slice(
                    i * fullOutputSize + outputOffset,
                    outputCount);

                TensorPrimitives.MultiplyAdd(
                    weightRow,
                    x,
                    outputSlice,
                    outputSlice);
            }
        }

        public static void CopyToCachePosition(
            ReadOnlySpan<float> source,
            KeyValueCache cache,
            int layerIndex,
            int headIndex,
            int position,
            bool copyToKey,
            bool copyToValue)
        {
            if (cache is null)
            {
                throw new ArgumentNullException(nameof(cache));
            }

            if (!copyToKey && !copyToValue)
            {
                return;
            }

            var headDimension = cache.Shape.HeadDimension;

            if (source.Length < headDimension)
            {
                throw new ArgumentException("Source span is smaller than cache head dimension.", nameof(source));
            }

            // Mode-agnostic write: F32 copies, Q8 quantizes — so this kernel works with either cache dtype.
            if (copyToKey)
            {
                cache.WriteKey(layerIndex, headIndex, position, source.Slice(0, headDimension));
            }

            if (copyToValue)
            {
                cache.WriteValue(layerIndex, headIndex, position, source.Slice(0, headDimension));
            }
        }

        private static void ValidateArguments(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            if (input.Length < inputSize)
            {
                throw new ArgumentException("Input span is smaller than inputSize.", nameof(input));
            }

            if (weightsInputOutput.Length < inputSize * outputSize)
            {
                throw new ArgumentException("Weights span is smaller than inputSize * outputSize.", nameof(weightsInputOutput));
            }

            if (output.Length < outputSize)
            {
                throw new ArgumentException("Output span is smaller than outputSize.", nameof(output));
            }
        }

        private static void ValidateSliceArguments(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            Span<float> output,
            int inputSize,
            int fullOutputSize,
            int outputOffset,
            int outputCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fullOutputSize);

            ArgumentOutOfRangeException.ThrowIfNegative(outputOffset);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputCount);

            ArgumentOutOfRangeException.ThrowIfLessThan(fullOutputSize, outputOffset + outputCount);

            if (input.Length < inputSize)
            {
                throw new ArgumentException("Input span is smaller than inputSize.", nameof(input));
            }

            if (weightsInputOutput.Length < inputSize * fullOutputSize)
            {
                throw new ArgumentException("Weights span is smaller than inputSize * fullOutputSize.", nameof(weightsInputOutput));
            }

            if (output.Length < outputCount)
            {
                throw new ArgumentException("Output span is smaller than outputCount.", nameof(output));
            }
        }
    }
}
