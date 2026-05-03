// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
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
    public static class SingleTokenProjectionKernel
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
                ReadOnlySpan<float>.Empty,
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
            if (inputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputSize));
            }

            if (outputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputSize));
            }

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
            if (inputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(inputSize));
            }

            if (fullOutputSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(fullOutputSize));
            }

            if (outputOffset < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputOffset));
            }

            if (outputCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(outputCount));
            }

            if (outputOffset + outputCount > fullOutputSize)
            {
                throw new ArgumentOutOfRangeException(nameof(outputCount));
            }

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
