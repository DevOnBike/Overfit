// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;
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

        // Context for the OverfitParallel workers. The managed weight rides through the void*
        // context as a GCHandleScope token (it has a DecodeRow method, so it can't be a plain float* like Conv2D).
        private unsafe struct FqlContext
        {
            public float* A;      // forward: input; backward: dy
            public float* Out;    // forward: output; backward: per-partition partial-dx base
            public nint Weight;   // GCHandleScope.Token (recover with GCHandleScope.Recover)
            public int N;
            public int K;
            public int M;
            public int P;         // backward partition count
        }

        private static unsafe void ForwardParallel(ReadOnlySpan<float> inS, Span<float> outS, IDequantRowSource weight, int n, int k, int m)
        {
            using var scope = new GcHandleScope(weight);
            fixed (float* inB = inS)
            fixed (float* outB = outS)
            {
                var ctx = new FqlContext { A = inB, Out = outB, Weight = scope.Token, N = n, K = k, M = m };
                OverfitParallel.For(0, m, &ForwardChunk, &ctx);
            }
        }

        private static unsafe void ForwardChunk(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<FqlContext>(context);
            var weight = GcHandleScope.Recover<IDequantRowSource>(ctx.Weight);
            using var rowBuf = new PooledBuffer<float>(ctx.K, clearMemory: false);
            var wRow = rowBuf.Span;
            for (var o = start; o < end; o++)
            {
                weight.DecodeRow(o, wRow);
                for (var b = 0; b < ctx.N; b++)
                {
                    ctx.Out[b * ctx.M + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(ctx.A + b * ctx.K, ctx.K), wRow);
                }
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
            // P fixed partitions of the output rows, each with a PRIVATE partial-dx slot → no race on
            // dx[b]; reduce the P partials into dx afterwards. (Parallelising over o would race; over
            // batch would re-dequant every row P×.) Partials are pooled + zeroed.
            var p = Math.Min(OverfitParallel.WorkerCount, m);
            var partialsLength = (long)p * n * k;
            if (partialsLength > int.MaxValue)
            {
                throw new ArgumentException($"Backward partials buffer would be {partialsLength} elements (> int.MaxValue).");
            }

            using var scope = new GcHandleScope(weight);
            using var partials = new PooledBuffer<float>((int)partialsLength, clearMemory: true);
            fixed (float* dyB = dy)
            fixed (float* partB = partials.Span)
            {
                var ctx = new FqlContext { A = dyB, Out = partB, Weight = scope.Token, N = n, K = k, M = m, P = p };
                OverfitParallel.For(0, p, &BackwardChunk, &ctx);
            }

            // Reduce: dx += Σ_partition partial.
            var part = partials.Span;
            for (var pi = 0; pi < p; pi++)
            {
                TensorPrimitives.Add(dx, part.Slice(pi * n * k, n * k), dx);
            }
        }

        private static unsafe void BackwardChunk(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<FqlContext>(context);
            var weight = GcHandleScope.Recover<IDequantRowSource>(ctx.Weight);
            using var rowBuf = new PooledBuffer<float>(ctx.K, clearMemory: false);
            var wRow = rowBuf.Span;
            for (var partition = start; partition < end; partition++)
            {
                var oStart = (int)((long)partition * ctx.M / ctx.P);
                var oEnd = (int)((long)(partition + 1) * ctx.M / ctx.P);
                var partial = ctx.Out + (long)partition * ctx.N * ctx.K; // this partition's private dx slot
                for (var o = oStart; o < oEnd; o++)
                {
                    weight.DecodeRow(o, wRow);
                    for (var b = 0; b < ctx.N; b++)
                    {
                        var g = ctx.A[b * ctx.M + o];
                        if (g != 0f)
                        {
                            Simd.MulAdd(wRow, g, new Span<float>(partial + b * ctx.K, ctx.K));
                        }
                    }
                }
            }
        }
    }
}
