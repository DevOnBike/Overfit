// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        // Frozen quantized base weights referenced by FrozenQuantizedLinear ops; the tape op
        // stores the list index in I0 (the weight is not an AutogradNode). Cleared by Reset().
        private List<IDequantRowSource>? _frozenQuantWeights;

        /// <summary>
        /// Linear projection through a <b>frozen</b> quantized weight (Q4_K / Q6_K), dequantized on
        /// the fly: <c>output[b, o] = Σ_i dequant(W)[o, i] · input[b, i]</c>. The weight is
        /// output-major (row <c>o</c> = one output's <see cref="IDequantRowSource.InputSize"/>-long
        /// contraction vector) and stays quantized in RAM — it is <b>never</b> an
        /// <see cref="AutogradNode"/> and <b>never</b> receives a gradient. Only the INPUT gradient
        /// flows back. This is the QLoRA base path: a 4–6-bit frozen base + trainable adapters
        /// (the LoRA branch is recorded separately and summed in by the caller).
        /// </summary>
        public AutogradNode FrozenQuantizedLinear(AutogradNode input, IDequantRowSource weight)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(weight);

            int n = input.Shape.D0, k = input.Shape.D1, m = weight.OutputSize;
            if (k != weight.InputSize)
            {
                throw new ArgumentException(
                    $"Input feature dim ({k}) must equal weight.InputSize ({weight.InputSize}).", nameof(input));
            }

            var output = CreateTemporary(new TensorShape(n, m), input.RequiresGrad, clearMemory: false);
            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            // Dequantize each output row ONCE, reuse across the batch (rows are the heavy part).
            using var rowBuf = new PooledBuffer<float>(k, clearMemory: false);
            var wRow = rowBuf.Span;
            for (var o = 0; o < m; o++)
            {
                weight.DecodeRow(o, wRow);
                for (var b = 0; b < n; b++)
                {
                    outS[b * m + o] = TensorPrimitives.Dot(inS.Slice(b * k, k), wRow);
                }
            }

            if (input.RequiresGrad)
            {
                _frozenQuantWeights ??= [];
                var idx = _frozenQuantWeights.Count;
                _frozenQuantWeights.Add(weight);
                Record(OpCode.FrozenQuantizedLinear, output, input, i0: idx, i1: k, i2: m);
            }

            return output;
        }

        /// <summary>
        /// Backward for <see cref="FrozenQuantizedLinear"/> — <b>dInput only</b> (the base is frozen,
        /// so there is no weight gradient and no weight-grad buffer = the QLoRA memory win):
        /// <c>dInput[b] += Σ_o dOutput[b, o] · dequant(W)[o]</c>. Accumulates into the input's grad
        /// (zeroed by ZeroGrad), reusing each dequantized row across the batch.
        /// </summary>
        private void FrozenQuantizedLinearBackward(in TapeOp op)
        {
            var input = op.A;
            if (!input.RequiresGrad)
            {
                return;
            }

            var weight = _frozenQuantWeights![op.I0];
            int k = op.I1, m = op.I2, n = op.Output.Shape.D0;
            var dy = op.Output.GradView.AsReadOnlySpan();
            var dx = input.GradView.AsSpan();

            using var rowBuf = new PooledBuffer<float>(k, clearMemory: false);
            var wRow = rowBuf.Span;
            for (var o = 0; o < m; o++)
            {
                weight.DecodeRow(o, wRow);
                for (var b = 0; b < n; b++)
                {
                    var g = dy[b * m + o];
                    if (g != 0f)
                    {
                        Simd.MulAdd(wRow, g, dx.Slice(b * k, k));
                    }
                }
            }
        }
    }
}
