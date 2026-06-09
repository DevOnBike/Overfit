// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

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
        private readonly sbyte[] _q8InputQuants;   // Q8 activation scratch (SwiGLU Q8 path)
        private readonly float[] _q8InputScales;
        private readonly sbyte[] _q8kInputQuants;  // Q4_K activation scratch (SwiGLU Q4_K path — Q8_K)
        private readonly float[] _q8kInputScales;
        private readonly short[] _q8kInputBsums;

        public CachedFeedForwardBlock(
            int dModel,
            int dFF,
            FeedForwardActivation activation = FeedForwardActivation.GeLU)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dFF);

            DModel = dModel;
            DFF = dFF;
            Activation = activation;

            _intermediate = new float[dFF];
            _gate = activation is FeedForwardActivation.SwiGLU or FeedForwardActivation.GeGLU
                              ? new float[dFF]
                              : [];

            // Q8 / Q4_K activation scratch — sized for the largest projection input. Gate/up
            // quantize the dModel-long hidden; down quantizes the dFF-long intermediate. For a dense
            // FFN dFF ≥ dModel, but a MoE expert can have dFF < dModel, so size to the max of the two.
            var scratch = Math.Max(dModel, dFF);
            _q8InputQuants = new sbyte[scratch];
            _q8InputScales = new float[(scratch + Q8DotKernel.BlockSize - 1) / Q8DotKernel.BlockSize];
            _q8kInputQuants = new sbyte[scratch];
            _q8kInputScales = new float[(scratch + Q4KDotKernel.SuperBlockElements - 1) / Q4KDotKernel.SuperBlockElements];
            _q8kInputBsums = new short[(scratch + Q4KDotKernel.GroupSize - 1) / Q4KDotKernel.GroupSize];
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

            SingleTokenProjectionKernel.ProjectParallel(
                hidden,
                w1,
                b1,
                _intermediate,
                DModel,
                DFF);

            ApplyActivation(_intermediate, Activation);

            SingleTokenProjectionKernel.ProjectParallel(
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
        /// Batched (prefill) GeLU/ReLU feed-forward — the multi-row counterpart of
        /// <see cref="Decode"/>. <paramref name="hidden"/> is row-major
        /// <c>[rows × dModel]</c>, <paramref name="output"/> <c>[rows × dModel]</c>.
        /// Both projections run through <see cref="BatchedProjectionKernel"/> (each
        /// weight tile read once and reused across all rows), and the activation is
        /// the SAME element-wise <c>ApplyActivation</c> the single-token path uses —
        /// applied across the whole <c>[rows × dFF]</c> intermediate — so the result
        /// is <b>bit-identical</b> to running <see cref="Decode"/> on each row.
        /// (Verified by <c>CachedFeedForwardBlockBatchedTests</c>.) Intended for the
        /// one-time prefill pass, so it allocates its <c>[rows × dFF]</c> scratch.
        /// </summary>
        public void DecodeBatched(
            ReadOnlySpan<float> hidden,
            int rows,
            ReadOnlySpan<float> w1,
            ReadOnlySpan<float> b1,
            ReadOnlySpan<float> w2,
            ReadOnlySpan<float> b2,
            Span<float> output)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);

            if (Activation is FeedForwardActivation.SwiGLU or FeedForwardActivation.GeGLU)
            {
                throw new OverfitRuntimeException("Gated FFN (SwiGLU/GeGLU) must use the gated batched path, not DecodeBatched.");
            }
            if (hidden.Length < (long)rows * DModel)
            {
                throw new ArgumentException("Hidden span < rows*dModel.", nameof(hidden));
            }
            if (output.Length < (long)rows * DModel)
            {
                throw new ArgumentException("Output span < rows*dModel.", nameof(output));
            }

            var intermediate = new float[(long)rows * DFF <= int.MaxValue ? rows * DFF : throw new ArgumentException("rows*dFF overflow.", nameof(rows))];

            BatchedProjectionKernel.ProjectParallel(hidden, rows, w1, b1, intermediate, DModel, DFF);

            ApplyActivation(intermediate.AsSpan(0, rows * DFF), Activation);

            BatchedProjectionKernel.ProjectParallel(intermediate, rows, w2, b2, output, DFF, DModel);
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
            SingleTokenProjectionKernel.ProjectParallel(hidden, wGate, [], _gate, DModel, DFF);
            ApplyGate(_gate, Activation);

            // up = hidden @ Wup
            SingleTokenProjectionKernel.ProjectParallel(hidden, wUp, [], _intermediate, DModel, DFF);

            // intermediate = gate * up (element-wise)
            TensorPrimitives.Multiply(_gate, _intermediate, _intermediate);

            // output = intermediate @ Wdown
            SingleTokenProjectionKernel.ProjectParallel(_intermediate, wDown, [], output, DFF, DModel);
        }

        /// <summary>
        /// SwiGLU decode with Q8_0-resident weights — the quantized counterpart
        /// of <see cref="DecodeSwiGlu"/>. Same math; each projection runs through
        /// <see cref="Q8DotKernel.ProjectParallel"/>. Used by the GGUF/Qwen
        /// decode path once weights are quantized (step 2.3b).
        /// </summary>
        public void DecodeSwiGluQuantized(
            ReadOnlySpan<float> hidden,
            Q8Weight wGate,
            Q8Weight wUp,
            Q8Weight wDown,
            Span<float> output)
        {
            // gate = SiLU(hidden @ Wgate)
            Q8DotKernel.ProjectParallel(hidden, wGate, [], _gate, _q8InputQuants, _q8InputScales);
            ApplyGate(_gate, Activation);

            // up = hidden @ Wup
            Q8DotKernel.ProjectParallel(hidden, wUp, [], _intermediate, _q8InputQuants, _q8InputScales);

            // intermediate = gate * up (element-wise)
            TensorPrimitives.Multiply(_gate, _intermediate, _intermediate);

            // output = intermediate @ Wdown
            Q8DotKernel.ProjectParallel(_intermediate, wDown, [], output, _q8InputQuants, _q8InputScales);
        }

        /// <summary>
        /// SwiGLU feed-forward decode with **per-weight dispatch** — each of the
        /// three projections picks its kernel from the weight's resident format
        /// (F32 / Q8_0 / Q4_K). The runtime calls this; a heterogeneous K-quant
        /// file (e.g. Q4_K_M with `ffn_gate`/`ffn_up` as Q4_K and `ffn_down` as
        /// Q6_K-then-Q8) is dispatched correctly on a per-projection basis. The
        /// older all-same-type <see cref="DecodeSwiGlu"/> /
        /// <see cref="DecodeSwiGluQuantized"/> entry points are kept for tests.
        /// </summary>
        internal void DecodeSwiGluDispatched(
            ReadOnlySpan<float> hidden,
            in DecodeWeight wGate,
            in DecodeWeight wUp,
            in DecodeWeight wDown,
            Span<float> output)
        {
            // Repacked 8×8 GEMV path (OVERFIT_REPACK_GEMV): quantize hidden once, then run
            // gate + up through the repacked kernel (8 rows/lane, no per-row hsum — ~2× the
            // per-core throughput of the 1-row kernel). Q4_K gate/up only; down stays Q6_K.
            if (Q4KGemvKernel.Enabled && wGate.IsQ4K && wUp.IsQ4K
                && wGate.Quantized4K.CanRepack && wUp.Quantized4K.CanRepack)
            {
                Q4KDotKernel.QuantizeActivationQ8K(
                    hidden.Slice(0, DModel), _q8kInputQuants, _q8kInputScales, _q8kInputBsums);
                Q4KGemvKernel.GemvParallel(
                    wGate.Quantized4K.EnsureRepacked(), DFF, DModel,
                    _q8kInputQuants, _q8kInputScales, _q8kInputBsums, _gate);
                ApplyGate(_gate, Activation);
                Q4KGemvKernel.GemvParallel(
                    wUp.Quantized4K.EnsureRepacked(), DFF, DModel,
                    _q8kInputQuants, _q8kInputScales, _q8kInputBsums, _intermediate);
            }
            // gate and up project the SAME hidden. When both are Q4_K (the Q4_K_M FFN
            // case), fuse them: quantize hidden once + one dispatch for both halves
            // (decode FFN is dispatch-overhead bound). Otherwise keep the two-call path.
            else if (wGate.IsQ4K && wUp.IsQ4K)
            {
                Q4KDotKernel.ProjectGateUpParallel(
                    hidden, wGate.Quantized4K, wUp.Quantized4K, _gate, _intermediate,
                    _q8kInputQuants, _q8kInputScales, _q8kInputBsums);
                ApplyGate(_gate, Activation);
            }
            else
            {
                // gate = SiLU(hidden @ Wgate)
                ProjectParallelDispatched(hidden, in wGate, [], _gate, DModel, DFF);
                ApplyGate(_gate, Activation);

                // up = hidden @ Wup
                ProjectParallelDispatched(hidden, in wUp, [], _intermediate, DModel, DFF);
            }

            // intermediate = gate * up (element-wise)
            TensorPrimitives.Multiply(_gate, _intermediate, _intermediate);

            // output = intermediate @ Wdown
            ProjectParallelDispatched(_intermediate, in wDown, [], output, DFF, DModel);
        }

        /// <summary>
        /// Batched (prefill) SwiGLU FFN over <paramref name="rows"/> token rows — the multi-row
        /// counterpart of <see cref="DecodeSwiGluDispatched"/>. Each of gate/up/down is a batched
        /// projection (<c>ProjectBatched</c>: read each weight row from DRAM once, reuse across all
        /// rows), so the dominant FFN weight-byte traffic is amortised ~<paramref name="rows"/>× vs
        /// looping the single-token path. Bit-identical to N× <see cref="DecodeSwiGluDispatched"/>.
        /// Scratch is allocated per call (prefill is a one-time pass, not the 0-alloc decode hot path).
        /// </summary>
        internal void DecodeSwiGluBatchedDispatched(
            ReadOnlySpan<float> hidden,
            int rows,
            in DecodeWeight wGate,
            in DecodeWeight wUp,
            in DecodeWeight wDown,
            Span<float> output)
        {
            var gate = new float[rows * DFF];
            var up = new float[rows * DFF];

            BatchedQuantProjection.Dispatch(hidden, rows, in wGate, [], gate, DModel, DFF);
            ApplyGate(gate, Activation);
            BatchedQuantProjection.Dispatch(hidden, rows, in wUp, [], up, DModel, DFF);
            TensorPrimitives.Multiply(gate, up, up);
            BatchedQuantProjection.Dispatch(up, rows, in wDown, [], output.Slice(0, rows * DModel), DFF, DModel);
        }

        /// <summary>
        /// Per-weight projection dispatch — picks the parallel kernel matching
        /// the weight's resident format. Q-paths use the block's owned activation
        /// scratch; F32 uses the F32 input-major kernel.
        /// </summary>
        private void ProjectParallelDispatched(
            ReadOnlySpan<float> input,
            in DecodeWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (weight.IsQ6K)
            {
                Q6KDotKernel.ProjectParallel(
                    input, weight.Quantized6K, bias, output,
                    _q8kInputQuants, _q8kInputScales, _q8kInputBsums);
            }
            else if (weight.IsQ4K)
            {
                Q4KDotKernel.ProjectParallel(
                    input, weight.Quantized4K, bias, output,
                    _q8kInputQuants, _q8kInputScales, _q8kInputBsums);
            }
            else if (weight.IsQuantized)
            {
                Q8DotKernel.ProjectParallel(
                    input, weight.Quantized, bias, output,
                    _q8InputQuants, _q8InputScales);
            }
            else
            {
                SingleTokenProjectionKernel.ProjectParallel(
                    input, weight.F32, bias, output, inputSize, outputSize);
            }
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
                    throw new OverfitRuntimeException(
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

        // Gated FFN activation applied to the gate branch: SiLU for SwiGLU (Llama/Qwen/Phi), GELU(tanh) for GeGLU (Gemma).
        private static void ApplyGate(Span<float> values, FeedForwardActivation activation)
        {
            if (activation == FeedForwardActivation.GeGLU)
            {
                ApplyGeLU(values);
            }
            else
            {
                ApplySiLU(values);
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
