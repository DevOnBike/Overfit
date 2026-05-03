// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Allocation-free LayerNorm kernel for a single transformer token.
    ///
    /// Computes:
    ///
    /// mean = average(input)
    /// variance = average((input - mean)^2)
    /// output[i] = ((input[i] - mean) / sqrt(variance + epsilon)) * gamma[i] + beta[i]
    ///
    /// Scope:
    ///
    /// - batch = 1,
    /// - one token,
    /// - FP32,
    /// - caller-owned output buffer,
    /// - optional gamma and beta.
    ///
    /// This is the missing normalization primitive needed before assembling
    /// CachedTransformerBlock:
    ///
    /// Pre-LN transformer block:
    ///   x1 = LayerNorm(x)
    ///   x  = x + Attention(x1)
    ///   x2 = LayerNorm(x)
    ///   x  = x + FFN(x2)
    /// </summary>
    public static class SingleTokenLayerNormKernel
    {
        public static void Normalize(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> gamma,
            ReadOnlySpan<float> beta,
            Span<float> output,
            int size,
            float epsilon)
        {
            ValidateArguments(
                input,
                gamma,
                beta,
                output,
                size,
                epsilon);

            var mean = 0f;

            for (var i = 0; i < size; i++)
            {
                mean += input[i];
            }

            mean /= size;

            var variance = 0f;

            for (var i = 0; i < size; i++)
            {
                var centered = input[i] - mean;
                variance += centered * centered;
            }

            variance /= size;

            var invStd = 1f / MathF.Sqrt(variance + epsilon);

            if (gamma.IsEmpty && beta.IsEmpty)
            {
                for (var i = 0; i < size; i++)
                {
                    output[i] = (input[i] - mean) * invStd;
                }

                return;
            }

            if (gamma.IsEmpty)
            {
                for (var i = 0; i < size; i++)
                {
                    output[i] = ((input[i] - mean) * invStd) + beta[i];
                }

                return;
            }

            if (beta.IsEmpty)
            {
                for (var i = 0; i < size; i++)
                {
                    output[i] = ((input[i] - mean) * invStd) * gamma[i];
                }

                return;
            }

            for (var i = 0; i < size; i++)
            {
                output[i] = ((input[i] - mean) * invStd * gamma[i]) + beta[i];
            }
        }

        public static void NormalizeWithoutAffine(
            ReadOnlySpan<float> input,
            Span<float> output,
            int size,
            float epsilon)
        {
            Normalize(
                input,
                ReadOnlySpan<float>.Empty,
                ReadOnlySpan<float>.Empty,
                output,
                size,
                epsilon);
        }

        public static void AddResidual(
            ReadOnlySpan<float> residual,
            ReadOnlySpan<float> update,
            Span<float> output,
            int size)
        {
            if (size <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(size));
            }

            if (residual.Length < size)
            {
                throw new ArgumentException("Residual span is smaller than size.", nameof(residual));
            }

            if (update.Length < size)
            {
                throw new ArgumentException("Update span is smaller than size.", nameof(update));
            }

            if (output.Length < size)
            {
                throw new ArgumentException("Output span is smaller than size.", nameof(output));
            }

            for (var i = 0; i < size; i++)
            {
                output[i] = residual[i] + update[i];
            }
        }

        public static void AddResidualInPlace(
            Span<float> destination,
            ReadOnlySpan<float> update,
            int size)
        {
            if (size <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(size));
            }

            if (destination.Length < size)
            {
                throw new ArgumentException("Destination span is smaller than size.", nameof(destination));
            }

            if (update.Length < size)
            {
                throw new ArgumentException("Update span is smaller than size.", nameof(update));
            }

            for (var i = 0; i < size; i++)
            {
                destination[i] += update[i];
            }
        }

        private static void ValidateArguments(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> gamma,
            ReadOnlySpan<float> beta,
            Span<float> output,
            int size,
            float epsilon)
        {
            if (size <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(size));
            }

            if (epsilon <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(epsilon));
            }

            if (input.Length < size)
            {
                throw new ArgumentException("Input span is smaller than size.", nameof(input));
            }

            if (!gamma.IsEmpty && gamma.Length < size)
            {
                throw new ArgumentException("Gamma span is smaller than size.", nameof(gamma));
            }

            if (!beta.IsEmpty && beta.Length < size)
            {
                throw new ArgumentException("Beta span is smaller than size.", nameof(beta));
            }

            if (output.Length < size)
            {
                throw new ArgumentException("Output span is smaller than size.", nameof(output));
            }
        }
    }
}
