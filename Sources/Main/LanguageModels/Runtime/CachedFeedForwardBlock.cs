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
        private readonly float[] _gate;  // SwiGLU gate buffer (empty for GeLU)

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
            _gate         = activation == FeedForwardActivation.SwiGLU
                              ? new float[dFF]
                              : Array.Empty<float>();
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
            Decode(hidden, w1, [], w2, [], output);
        }

        /// <summary>
        /// SwiGLU feed-forward decode:
        ///   FFN(x) = (SiLU(x @ Wgate) * (x @ Wup)) @ Wdown
        ///
        /// Three weight matrices, no biases (Llama / Mistral / Qwen convention).
        /// wGate and wUp have shape [dModel, dFF]. wDown has shape [dFF, dModel].
        /// </summary>
        public void DecodeSwiGlu(
            ReadOnlySpan<float> hidden,
            ReadOnlySpan<float> wGate,
            ReadOnlySpan<float> wUp,
            ReadOnlySpan<float> wDown,
            Span<float> output)
        {
            if (hidden.Length < DModel)
            {
                throw new ArgumentException("Hidden span smaller than dModel.", nameof(hidden));
            }
            if (wGate.Length < DModel * DFF)
            {
                throw new ArgumentException("wGate span smaller than dModel * dFF.", nameof(wGate));
            }
            if (wUp.Length < DModel * DFF)
            {
                throw new ArgumentException("wUp span smaller than dModel * dFF.", nameof(wUp));
            }
            if (wDown.Length < DFF * DModel)
            {
                throw new ArgumentException("wDown span smaller than dFF * dModel.", nameof(wDown));
            }
            if (output.Length < DModel)
            {
                throw new ArgumentException("Output span smaller than dModel.", nameof(output));
            }

            // gate = SiLU(hidden @ Wgate)
            SingleTokenProjectionKernel.Project(hidden, wGate, [], _gate, DModel, DFF);
            ApplySiLU(_gate);

            // up = hidden @ Wup
            SingleTokenProjectionKernel.Project(hidden, wUp, [], _intermediate, DModel, DFF);

            // intermediate = gate * up (element-wise)
            System.Numerics.Tensors.TensorPrimitives.Multiply(_gate, _intermediate, _intermediate);

            // output = intermediate @ Wdown
            SingleTokenProjectionKernel.Project(_intermediate, wDown, [], output, DFF, DModel);
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

                case FeedForwardActivation.SwiGLU:
                    // SwiGLU is handled separately via DecodeSwiGlu — not through this path.
                    throw new InvalidOperationException(
                        "SwiGLU must be invoked via DecodeSwiGlu, not through the standard Decode path.");

                default:
                    throw new ArgumentOutOfRangeException(nameof(activation));
            }
        }

        private static void ApplySiLU(Span<float> values)
        {
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            for (var i = 0; i < values.Length; i++)
            {
                var x = values[i];
                values[i] = x / (1f + MathF.Exp(-x));
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
