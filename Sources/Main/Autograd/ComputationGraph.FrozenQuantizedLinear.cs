// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Threading.Tasks;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        // Above this many MACs (n·k·m) the per-output-row dequant+dot/axpy is parallelised over m
        // (Parallel.For, like Conv2D — this is the training path, not the 0-alloc decode hot path).
        private const long FrozenQuantParallelThreshold = 200_000;

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

            ForwardCompute(input.DataView.AsReadOnlySpan(), output.DataView.AsSpan(), weight, n, k, m);

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
            BackwardCompute(op.Output.GradView.AsReadOnlySpan(), input.GradView.AsSpan(), weight, n, k, m);
        }

        // ── forward: out[b,o] = Σ_i dequant(W)[o,i]·in[b,i], dequant each row once, reuse over batch ──

        private static void ForwardCompute(ReadOnlySpan<float> inS, Span<float> outS, IDequantRowSource weight, int n, int k, int m)
        {
            if ((long)n * k * m < FrozenQuantParallelThreshold)
            {
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
                return;
            }

            ForwardParallel(inS, outS, weight, n, k, m);
        }

        private static unsafe void ForwardParallel(ReadOnlySpan<float> inS, Span<float> outS, IDequantRowSource weight, int n, int k, int m)
        {
            fixed (float* inB = inS)
            fixed (float* outB = outS)
            {
                nint inP = (nint)inB, outP = (nint)outB;
                // Each output row o is independent and writes a disjoint column out[*, o] → no race.
                Parallel.For(0, m, () => new float[k], (o, _, scratch) =>
                {
                    var wRow = scratch.AsSpan(0, k);
                    weight.DecodeRow(o, wRow);
                    var ip = (float*)inP;
                    var op = (float*)outP;
                    for (var b = 0; b < n; b++)
                    {
                        op[b * m + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(ip + b * k, k), wRow);
                    }
                    return scratch;
                }, _ => { });
            }
        }

        // ── backward (dInput only): dx[b] += Σ_o dy[b,o]·dequant(W)[o]. Parallelising over o races on
        //    dx[b], so each thread accumulates a private partial and reduces under a lock at the end. ──

        private static void BackwardCompute(ReadOnlySpan<float> dy, Span<float> dx, IDequantRowSource weight, int n, int k, int m)
        {
            if ((long)n * k * m < FrozenQuantParallelThreshold)
            {
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
                return;
            }

            BackwardParallel(dy, dx, weight, n, k, m);
        }

        private static unsafe void BackwardParallel(ReadOnlySpan<float> dy, Span<float> dx, IDequantRowSource weight, int n, int k, int m)
        {
            fixed (float* dyB = dy)
            fixed (float* dxB = dx)
            {
                nint dyP = (nint)dyB, dxP = (nint)dxB;
                var gate = new object();
                Parallel.For(
                    0, m,
                    () => (scratch: new float[k], partial: new float[n * k]),
                    (o, _, local) =>
                    {
                        var wRow = local.scratch.AsSpan(0, k);
                        weight.DecodeRow(o, wRow);
                        var dyp = (float*)dyP;
                        for (var b = 0; b < n; b++)
                        {
                            var g = dyp[b * m + o];
                            if (g != 0f)
                            {
                                Simd.MulAdd(wRow, g, local.partial.AsSpan(b * k, k));
                            }
                        }
                        return local;
                    },
                    local =>
                    {
                        lock (gate)
                        {
                            var dxp = new Span<float>((float*)dxP, n * k);
                            TensorPrimitives.Add(dxp, local.partial, dxp);
                        }
                    });
            }
        }
    }
}
