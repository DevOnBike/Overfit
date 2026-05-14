// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

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

            var outputSlice = output.Slice(0, outputSize);

            for (var i = 0; i < inputSize; i++)
            {
                var x = input[i];

                if (x == 0f)
                {
                    continue;
                }

                var weightRow = weightsInputOutput.Slice(
                    i * outputSize,
                    outputSize);

                TensorPrimitives.MultiplyAdd(
                    weightRow,
                    x,
                    outputSlice,
                    outputSlice);
            }
        }

        /// <summary>
        /// Parallel projection for large output dimensions (LM head, FFN W1).
        ///
        /// Splits the output dimension into chunks and runs each chunk on a
        /// separate thread via <see cref="Parallel.For"/>. Uses <c>unsafe</c>
        /// MemoryMarshal.GetReference + Unsafe.Add to cross the managed/Parallel boundary without
        /// allocating delegates or closures per call.
        ///
        /// Threshold: only parallelises when <c>outputSize</c> exceeds
        /// <see cref="ParallelThreshold"/>. Below the threshold falls back
        /// to the sequential <see cref="Project"/> path.
        ///
        /// Access pattern: each thread reads ALL input values (small — fits
        /// in L1) and a contiguous slice of each weight row (cache-friendly
        /// within the slice). No output sharing between threads.
        /// </summary>
        public const int ParallelThreshold = 10_000;

        public static unsafe void ProjectParallel(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (outputSize < ParallelThreshold)
            {
                Project(input, weightsInputOutput, bias, output, inputSize, outputSize);
                return;
            }

            ValidateArguments(input, weightsInputOutput, output, inputSize, outputSize);

            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException("Bias span is smaller than outputSize.", nameof(bias));
            }

            // Sequential init: fill output with bias or zero.
            // Cheaper than parallelising (one pass, no synchronisation).
            if (!bias.IsEmpty)
            {
                bias.Slice(0, outputSize).CopyTo(output);
            }
            else
            {
                output.Slice(0, outputSize).Clear();
            }

            var processorCount = Environment.ProcessorCount;
            var chunkSize      = Math.Max(512, (outputSize + processorCount - 1) / processorCount);
            var chunkCount     = (outputSize + chunkSize - 1) / chunkSize;

            // CS1764/CS8175: fixed and ref locals cannot be captured in lambdas.
            // Safe workaround: convert to nint (value type) before Parallel.For.
            // Parallel.For is synchronous — it blocks until ALL worker threads finish,
            // so the fixed block is guaranteed to be active for the entire duration.
            // The nint values are addresses captured by value in the lambda closure.
            fixed (float* pInput   = input)
            fixed (float* pWeights = weightsInputOutput)
            fixed (float* pOutput  = output)
            {
                var ip = (nint)pInput;
                var wp = (nint)pWeights;
                var op = (nint)pOutput;

                Parallel.For(0, chunkCount, chunk =>
                {
                    var start = chunk * chunkSize;
                    var count = Math.Min(chunkSize, outputSize - start);

                    // Cast nint back to pointer — safe because Parallel.For is
                    // synchronous and the fixed block is still active.
                    var outputChunk = new Span<float>((float*)op + start, count);

                    // Accumulate: output[start:start+count] += input[i] * W[i, start:start+count]
                    // output was already initialised with bias (or zero) above.
                    for (var i = 0; i < inputSize; i++)
                    {
                        var x = ((float*)ip)[i];
                        if (x == 0f)
                        {
                            continue;
                        }

                        var weightSlice = new ReadOnlySpan<float>(
                            (float*)wp + (long)i * outputSize + start,
                            count);

                        TensorPrimitives.MultiplyAdd(weightSlice, x, outputChunk, outputChunk);
                    }
                });
            }
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

            if (copyToKey)
            {
                source
                    .Slice(0, headDimension)
                    .CopyTo(cache.GetKeyWriteSpan(layerIndex, headIndex, position));
            }

            if (copyToValue)
            {
                source
                    .Slice(0, headDimension)
                    .CopyTo(cache.GetValueWriteSpan(layerIndex, headIndex, position));
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
