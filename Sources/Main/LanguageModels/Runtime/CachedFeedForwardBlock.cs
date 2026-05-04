// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Cached single-token feed-forward block for transformer decode.
    ///
    /// Computes:
    ///
    /// hidden -> Linear(dModel, dFF) -> activation -> Linear(dFF, dModel)
    ///
    /// Scope:
    /// - batch = 1,
    /// - one token,
    /// - FP32,
    /// - caller-owned output buffer,
    /// - no residual connection,
    /// - no layer normalization.
    ///
    /// Layout:
    ///
    /// w1: [dModel, dFF] in input-major order
    /// b1: [dFF], optional
    /// w2: [dFF, dModel] in input-major order
    /// b2: [dModel], optional
    /// </summary>
    public sealed class CachedFeedForwardBlock
    {
        private readonly float[] _intermediate;

        public CachedFeedForwardBlock(
            int dModel,
            int dFF,
            FeedForwardActivation activation = FeedForwardActivation.GeLU)
        {
            if (dModel <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(dModel));
            }

            if (dFF <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(dFF));
            }

            DModel = dModel;
            DFF = dFF;
            Activation = activation;

            _intermediate = new float[dFF];
        }

        public int DModel { get; }

        public int DFF { get; }

        public FeedForwardActivation Activation { get; }

        public void Decode(
            ReadOnlySpan<float> hidden,
            ReadOnlySpan<float> w1,
            ReadOnlySpan<float> b1,
            ReadOnlySpan<float> w2,
            ReadOnlySpan<float> b2,
            Span<float> output)
        {
            ValidateDecodeArguments(
                hidden,
                w1,
                b1,
                w2,
                b2,
                output);

            SingleTokenProjectionKernel.Project(
                hidden,
                w1,
                b1,
                _intermediate,
                DModel,
                DFF);

            ApplyActivation(_intermediate, Activation);

            SingleTokenProjectionKernel.Project(
                _intermediate,
                w2,
                b2,
                output,
                DFF,
                DModel);
        }

        public void DecodeWithoutBias(
            ReadOnlySpan<float> hidden,
            ReadOnlySpan<float> w1,
            ReadOnlySpan<float> w2,
            Span<float> output)
        {
            Decode(
                hidden,
                w1,
                ReadOnlySpan<float>.Empty,
                w2,
                ReadOnlySpan<float>.Empty,
                output);
        }

        public void GetLastIntermediate(Span<float> destination)
        {
            if (destination.Length < DFF)
            {
                throw new ArgumentException("Destination span is smaller than dFF.", nameof(destination));
            }

            _intermediate.AsSpan().CopyTo(destination);
        }

        private void ValidateDecodeArguments(
            ReadOnlySpan<float> hidden,
            ReadOnlySpan<float> w1,
            ReadOnlySpan<float> b1,
            ReadOnlySpan<float> w2,
            ReadOnlySpan<float> b2,
            Span<float> output)
        {
            if (hidden.Length < DModel)
            {
                throw new ArgumentException("Hidden span is smaller than dModel.", nameof(hidden));
            }

            if (w1.Length < DModel * DFF)
            {
                throw new ArgumentException("W1 span is smaller than dModel * dFF.", nameof(w1));
            }

            if (!b1.IsEmpty && b1.Length < DFF)
            {
                throw new ArgumentException("B1 span is smaller than dFF.", nameof(b1));
            }

            if (w2.Length < DFF * DModel)
            {
                throw new ArgumentException("W2 span is smaller than dFF * dModel.", nameof(w2));
            }

            if (!b2.IsEmpty && b2.Length < DModel)
            {
                throw new ArgumentException("B2 span is smaller than dModel.", nameof(b2));
            }

            if (output.Length < DModel)
            {
                throw new ArgumentException("Output span is smaller than dModel.", nameof(output));
            }
        }

        private static void ApplyActivation(
            Span<float> values,
            FeedForwardActivation activation)
        {
            switch (activation)
            {
                case FeedForwardActivation.None:
                    return;

                case FeedForwardActivation.ReLU:
                    ApplyReLU(values);
                    return;

                case FeedForwardActivation.GeLU:
                    ApplyGeLU(values);
                    return;

                default:
                    throw new ArgumentOutOfRangeException(nameof(activation));
            }
        }

        private static void ApplyReLU(Span<float> values)
        {
            for (var i = 0; i < values.Length; i++)
            {
                if (values[i] < 0f)
                {
                    values[i] = 0f;
                }
            }
        }

        private static void ApplyGeLU(Span<float> values)
        {
            // Approximation used by many transformer implementations:
            //
            // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
            //
            // This is intentionally scalar for now. The first goal is a correct,
            // allocation-free single-token FFN block. Vectorization can be done
            // later if this becomes a bottleneck.
            const float sqrtTwoOverPi = 0.7978845608028654f;
            const float coeff = 0.044715f;

            for (var i = 0; i < values.Length; i++)
            {
                var x = values[i];
                var x3 = x * x * x;
                var inner = sqrtTwoOverPi * (x + coeff * x3);

                values[i] = 0.5f * x * (1f + MathF.Tanh(inner));
            }
        }
    }
}
