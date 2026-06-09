// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Multi-threaded dequantize-and-matvec for the trainable model's <i>cached generation</i> path
    /// (<c>GenerateCached</c>* and the LM head). The reference decode looped over output rows on one core; this fans
    /// the rows across all cores — output-parallel, so race-free. Mirrors the training matmul's pattern
    /// (<c>ComputationGraph.FrozenQuantizedLinear</c>): SIMD <see cref="TensorPrimitives"/>,
    /// <see cref="OverfitParallelFor"/>, per-worker <see cref="PooledBuffer{T}"/> scratch (no per-row allocation),
    /// the managed weight passed through the unsafe context via a <see cref="GcHandleScope"/>.
    /// </summary>
    internal static unsafe class DequantMatVec
    {
        // Below this much work, the thread hand-off costs more than it saves — stay single-threaded.
        private const long ParallelThreshold = 1L << 16;

        /// <summary>Computes <c>dst[o] = dot(W.row(o), x)</c> for every output row of <paramref name="w"/>.</summary>
        public static void Run(ReadOnlySpan<float> x, IDequantRowSource w, Span<float> dst)
        {
            var outDim = w.OutputSize;
            var inDim = w.InputSize;

            if ((long)outDim * inDim < ParallelThreshold)
            {
                using var rowBuf = new PooledBuffer<float>(inDim, clearMemory: false);
                var row = rowBuf.Span;

                for (var o = 0; o < outDim; o++)
                {
                    w.DecodeRow(o, row);
                    dst[o] = TensorPrimitives.Dot(row, x);
                }

                return;
            }

            using var scope = new GcHandleScope(w);

            fixed (float* xp = x, dp = dst)
            {
                var ctx = new MatVecContext { X = xp, Dst = dp, Weight = scope.Token, K = inDim };
                OverfitParallelFor.For(0, outDim, &Chunk, &ctx);
            }
        }

        private struct MatVecContext
        {
            public float* X;
            public float* Dst;
            public nint Weight;
            public int K;
        }

        private static void Chunk(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<MatVecContext>(context);
            var weight = GcHandleScope.Recover<IDequantRowSource>(ctx.Weight);
            using var rowBuf = new PooledBuffer<float>(ctx.K, clearMemory: false);
            var row = rowBuf.Span;
            var xS = new ReadOnlySpan<float>(ctx.X, ctx.K);

            for (var o = start; o < end; o++)
            {
                weight.DecodeRow(o, row);
                ctx.Dst[o] = TensorPrimitives.Dot(row, xS);
            }
        }
    }
}
