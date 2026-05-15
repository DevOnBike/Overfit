// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // ADD
        // ====================================================================

        public static AutogradNode Add(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var requiresGrad = left.RequiresGrad || right.RequiresGrad;
            var output = AllocateNode(graph, left.Shape, requiresGrad, clearMemory: false);

            TensorKernels.Add(left.DataView.AsReadOnlySpan(), right.DataView.AsReadOnlySpan(), output.DataView.AsSpan());

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Add, output, left, right);
            }

            return output;
        }

        public static void AddBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                TensorPrimitives.Add(a.GradView.AsSpan(), output.GradView.AsReadOnlySpan(), a.GradView.AsSpan());
            }
            if (b.RequiresGrad)
            {
                TensorPrimitives.Add(b.GradView.AsSpan(), output.GradView.AsReadOnlySpan(), b.GradView.AsSpan());
            }
        }

        // ====================================================================
        // SUBTRACT
        // ====================================================================

        public static AutogradNode Subtract(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var requiresGrad = left.RequiresGrad || right.RequiresGrad;
            var output = AllocateNode(graph, left.Shape, requiresGrad, clearMemory: false);

            TensorPrimitives.Subtract(left.DataView.AsReadOnlySpan(), right.DataView.AsReadOnlySpan(), output.DataView.AsSpan());

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Subtract, output, left, right);
            }
            return output;
        }

        public static void SubtractBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                TensorPrimitives.Add(a.GradView.AsSpan(), output.GradView.AsReadOnlySpan(), a.GradView.AsSpan());
            }
            if (b.RequiresGrad)
            {
                TensorPrimitives.Subtract(b.GradView.AsSpan(), output.GradView.AsReadOnlySpan(), b.GradView.AsSpan());
            }
        }

        // ====================================================================
        // ADD BIAS
        // ====================================================================

        public static AutogradNode AddBias(ComputationGraph graph, AutogradNode input, AutogradNode bias)
        {
            int N = input.Shape.D0, C = input.Shape.D1;
            var requiresGrad = input.RequiresGrad || bias.RequiresGrad;
            var output = AllocateNode(graph, input.Shape, requiresGrad, clearMemory: false);

            if (N < BatchSequentialThreshold)
            {
                var inS = input.DataView.AsReadOnlySpan();
                var bS = bias.DataView.AsReadOnlySpan();
                var outS = output.DataView.AsSpan();

                for (var i = 0; i < N; i++)
                {
                    Simd.Add(inS.Slice(i * C, C), bS, outS.Slice(i * C, C));
                }
            }
            else
            {
                var inSpan = input.DataView.AsReadOnlySpan();
                var bSpan = bias.DataView.AsReadOnlySpan();
                var outSpan = output.DataView.AsSpan();

                unsafe
                {
                    fixed (float* inPtr = inSpan, bPtr = bSpan, outPtr = outSpan)
                    {
                        var ctx = new AddBiasCtx { Input = inPtr, Bias = bPtr, Output = outPtr, C = C };
                        OverfitParallelFor.For(0, N, &AddBiasChunk, &ctx);
                    }
                }
            }

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.AddBias, output, input, bias);
            }
            return output;
        }

        private unsafe struct AddBiasCtx
        {
            public float* Input;
            public float* Bias;
            public float* Output;
            public int C;
        }

        private static unsafe void AddBiasChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<AddBiasCtx>(contextPtr);
            var biasSpan = new ReadOnlySpan<float>(ctx.Bias, ctx.C);
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                Simd.Add(
                    new ReadOnlySpan<float>(ctx.Input + i * ctx.C, ctx.C),
                    biasSpan,
                    new Span<float>(ctx.Output + i * ctx.C, ctx.C));
            }
        }

        public static void AddBiasBackward(AutogradNode input, AutogradNode bias, AutogradNode output)
        {
            int N = input.Shape.D0, C = input.Shape.D1;

            if (input.RequiresGrad)
            {
                TensorPrimitives.Add(input.GradView.AsSpan(), output.GradView.AsReadOnlySpan(), input.GradView.AsSpan());
            }

            if (!bias.RequiresGrad)
            {
                return;
            }

            // Bias gradient = column-wise sum of output gradient across the batch dimension:
            //   bias.Grad[c] += sum_i( output.Grad[i, c] )
            //
            // Kept as a single sequential pass. Previous iteration used Parallel.For with
            // per-worker TensorStorage partials + ConcurrentBag merge; for realistic shapes
            // (N ≤ 512, C ≤ 2048) the parallel spawn + merge overhead dominated the actual
            // SIMD work, and the per-call allocations (worker wrappers + collection) were
            // the single largest contributor to backward-pass managed garbage. Sequential
            // SIMD keeps everything in the caller thread's L1/L2 and allocates nothing.
            var biasGrad = bias.GradView.AsSpan();
            var outGrad = output.GradView.AsReadOnlySpan();

            for (var i = 0; i < N; i++)
            {
                Simd.Add(biasGrad, outGrad.Slice(i * C, C), biasGrad);
            }
        }

        // ====================================================================
        // MATMUL
        // ====================================================================

        public static AutogradNode MatMul(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var requiresGrad = left.RequiresGrad || right.RequiresGrad;
            var output = MatMulRaw(graph, left, right, requiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MatMul, output, left, right);
            }
            return output;
        }

        public static AutogradNode MatMulRaw(ComputationGraph? graph, AutogradNode A, AutogradNode B, bool requiresGrad = false)
        {
            int aR = A.Shape.D0, aC = A.Shape.D1, bC = B.Shape.D1;

            // Konieczne clearMemory: true, bo będziemy akumulować mnożenie (+=)
            var C = AllocateNode(graph, new TensorShape(aR, bC), requiresGrad, clearMemory: true);

            if ((long)aR * aC * bC < ParallelThreshold)
            {
                MatMulRawSeq(A.DataView.AsReadOnlySpan(), B.DataView.AsReadOnlySpan(), aR, aC, bC, C.DataView.AsSpan());
            }
            else
            {
                var aSpan = A.DataView.AsReadOnlySpan();
                var bSpan = B.DataView.AsReadOnlySpan();
                var cSpan = C.DataView.AsSpan();

                unsafe
                {
                    fixed (float* aPtr = aSpan, bPtr = bSpan, cPtr = cSpan)
                    {
                        var ctx = new MatMulRawCtx { A = aPtr, B = bPtr, C = cPtr, AC = aC, BC = bC };
                        OverfitParallelFor.For(0, aR, &MatMulRawChunk, &ctx);
                    }
                }
            }

            return C;
        }

        private unsafe struct MatMulRawCtx
        {
            public float* A;
            public float* B;
            public float* C;
            public int AC;
            public int BC;
        }

        private static unsafe void MatMulRawChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<MatMulRawCtx>(contextPtr);
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                var rC = new Span<float>(ctx.C + i * ctx.BC, ctx.BC);
                var rA = new ReadOnlySpan<float>(ctx.A + i * ctx.AC, ctx.AC);

                for (var k = 0; k < ctx.AC; k++)
                {
                    var aVal = rA[k];
                    if (aVal != 0f)
                    {
                        Simd.MulAdd(new ReadOnlySpan<float>(ctx.B + k * ctx.BC, ctx.BC), aVal, rC);
                    }
                }
            }
        }

        public static void MatMulRawSeq(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, int aR, int aC, int bC, Span<float> cS)
        {
            for (var i = 0; i < aR; i++)
            {
                var rC = cS.Slice(i * bC, bC);
                var rA = aS.Slice(i * aC, aC);

                for (var k = 0; k < aC; k++)
                {
                    var aVal = rA[k];
                    if (aVal != 0f)
                    {
                        Simd.MulAdd(bS.Slice(k * bC, bC), aVal, rC);
                    }
                }
            }
        }

        public static void MatMulBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                MatMulAdd_A_BT_Raw(output, true, b, false, a, true);
            }
            if (b.RequiresGrad)
            {
                MatMulAdd_AT_B_Raw(a, false, output, true, b, true);
            }
        }

        // ====================================================================
        // MATMUL VARIANTS (A*B^T, A^T*B)
        // ====================================================================

        public static void MatMulAdd_A_BT_Raw(AutogradNode A, bool aGrad, AutogradNode B, bool bGrad, AutogradNode C, bool cGrad)
        {
            if (!cGrad || !C.RequiresGrad)
            {
                return;
            }

            int N = A.Shape.D0, K = A.Shape.D1, M = B.Shape.D0;

            if ((long)N * K * M < ParallelThreshold)
            {
                MatMulAdd_A_BT_Seq(
                    aGrad ? A.GradView.AsReadOnlySpan() : A.DataView.AsReadOnlySpan(),
                    bGrad ? B.GradView.AsReadOnlySpan() : B.DataView.AsReadOnlySpan(),
                    C.GradView.AsSpan(), N, K, M);
            }
            else
            {
                var aSpan = aGrad ? A.GradView.AsReadOnlySpan() : A.DataView.AsReadOnlySpan();
                var bSpan = bGrad ? B.GradView.AsReadOnlySpan() : B.DataView.AsReadOnlySpan();
                var cSpan = C.GradView.AsSpan();

                unsafe
                {
                    fixed (float* aPtr = aSpan, bPtr = bSpan, cPtr = cSpan)
                    {
                        var ctx = new MatMulAdd_A_BT_Ctx { A = aPtr, B = bPtr, C = cPtr, K = K, M = M };
                        OverfitParallelFor.For(0, N, &MatMulAdd_A_BT_Chunk, &ctx);
                    }
                }
            }
        }

        private unsafe struct MatMulAdd_A_BT_Ctx
        {
            public float* A;
            public float* B;
            public float* C;
            public int K;
            public int M;
        }

        private static unsafe void MatMulAdd_A_BT_Chunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<MatMulAdd_A_BT_Ctx>(contextPtr);
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                var aRow = new ReadOnlySpan<float>(ctx.A + i * ctx.K, ctx.K);
                var cRow = new Span<float>(ctx.C + i * ctx.M, ctx.M);

                for (var j = 0; j < ctx.M; j++)
                {
                    cRow[j] += Simd.Dot(aRow, new ReadOnlySpan<float>(ctx.B + j * ctx.K, ctx.K));
                }
            }
        }

        public static void MatMulAdd_A_BT_Seq(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, Span<float> cS, int N, int K, int M)
        {
            for (var i = 0; i < N; i++)
            {
                var aRow = aS.Slice(i * K, K);
                var cRow = cS.Slice(i * M, M);

                for (var j = 0; j < M; j++)
                {
                    cRow[j] += Simd.Dot(aRow, bS.Slice(j * K, K));
                }
            }
        }

        public static void MatMulAdd_AT_B_Raw(AutogradNode A, bool aGrad, AutogradNode B, bool bGrad, AutogradNode C, bool cGrad)
        {
            if (!cGrad || !C.RequiresGrad)
            {
                return;
            }

            int K = A.Shape.D0, N = A.Shape.D1, M = B.Shape.D1;

            if ((long)K * N * M < ParallelThreshold)
            {
                MatMulAdd_AT_B_Seq(
                    aGrad ? A.GradView.AsReadOnlySpan() : A.DataView.AsReadOnlySpan(),
                    bGrad ? B.GradView.AsReadOnlySpan() : B.DataView.AsReadOnlySpan(),
                    C.GradView.AsSpan(), K, N, M);
            }
            else
            {
                var aSpan = aGrad ? A.GradView.AsReadOnlySpan() : A.DataView.AsReadOnlySpan();
                var bSpan = bGrad ? B.GradView.AsReadOnlySpan() : B.DataView.AsReadOnlySpan();
                var cSpan = C.GradView.AsSpan();

                unsafe
                {
                    fixed (float* aPtr = aSpan, bPtr = bSpan, cPtr = cSpan)
                    {
                        var ctx = new MatMulAdd_AT_B_Ctx { A = aPtr, B = bPtr, C = cPtr, K = K, N = N, M = M };
                        OverfitParallelFor.For(0, N, &MatMulAdd_AT_B_Chunk, &ctx);
                    }
                }
            }
        }

        private unsafe struct MatMulAdd_AT_B_Ctx
        {
            public float* A;
            public float* B;
            public float* C;
            public int K;
            public int N;
            public int M;
        }

        private static unsafe void MatMulAdd_AT_B_Chunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<MatMulAdd_AT_B_Ctx>(contextPtr);
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                var cRow = new Span<float>(ctx.C + i * ctx.M, ctx.M);
                for (var k = 0; k < ctx.K; k++)
                {
                    var aVal = ctx.A[k * ctx.N + i];
                    if (aVal != 0f)
                    {
                        Simd.MulAdd(new ReadOnlySpan<float>(ctx.B + k * ctx.M, ctx.M), aVal, cRow);
                    }
                }
            }
        }

        public static void MatMulAdd_AT_B_Seq(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, Span<float> cS, int K, int N, int M)
        {
            for (var i = 0; i < N; i++)
            {
                var cRow = cS.Slice(i * M, M);
                for (var k = 0; k < K; k++)
                {
                    var aVal = aS[k * N + i];
                    if (aVal != 0f)
                    {
                        Simd.MulAdd(bS.Slice(k * M, M), aVal, cRow);
                    }
                }
            }
        }

        // ====================================================================
        // MULTIPLY (ELEMENT-WISE)
        // ====================================================================

        public static AutogradNode Multiply(ComputationGraph graph, AutogradNode a, AutogradNode b)
        {
            var requiresGrad = a.RequiresGrad || b.RequiresGrad;
            var output = AllocateNode(graph, a.Shape, requiresGrad, clearMemory: false);

            TensorKernels.Multiply(a.DataView.AsReadOnlySpan(), b.DataView.AsReadOnlySpan(), output.DataView.AsSpan());

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Multiply, output, a, b);
            }
            return output;
        }

        public static void MultiplyBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                TensorPrimitives.MultiplyAdd(output.GradView.AsReadOnlySpan(), b.DataView.AsReadOnlySpan(), a.GradView.AsSpan(), a.GradView.AsSpan());
            }
            if (b.RequiresGrad)
            {
                TensorPrimitives.MultiplyAdd(output.GradView.AsReadOnlySpan(), a.DataView.AsReadOnlySpan(), b.GradView.AsSpan(), b.GradView.AsSpan());
            }
        }

        // ====================================================================
        // LINEAR
        // ====================================================================

        /// <summary>
        /// Compatibility shim — delegates to <see cref="ComputationGraph.Linear"/> (PR5-7a).
        /// Implementation lives in ComputationGraph.Linear.cs.
        /// Null-graph inference path handled in <see cref="ComputationGraph.LinearOp"/>.
        /// </summary>
        public static AutogradNode Linear(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode weights,
            AutogradNode bias)
            => graph != null
                ? graph.Linear(input, weights, bias)
                : ComputationGraph.LinearOp(null, input, weights, bias);

        // Linear backward parallel threshold.
        // Below: use new sequential span-only kernels (zero TPL overhead).
        // Above: use existing parallel MatMulBackward (TPL justified by work size).
        // 200_000 ops chosen so that:
        //   - Linear(8,10):    64 * 8 * 10 =  5120 ops → sequential kernels ✓
        //   - Linear(64,10):   64 * 64 * 10 = 40960 ops → sequential kernels ✓
        //   - Linear(1352,64): 64 * 1352 * 64 = 5.5M ops → parallel MatMul ✓
        private const long LinearBackwardSequentialThreshold = 200_000;

        public static void LinearBackward(AutogradNode input, AutogradNode weights, AutogradNode bias, AutogradNode output)
        {
            var batchSize = input.Shape.D0;
            var inputSize = weights.Shape.D0;
            var outputSize = weights.Shape.D1;
            var ops = (long)batchSize * inputSize * outputSize;

            if (ops < LinearBackwardSequentialThreshold)
            {
                // Small matrix: sequential span-only kernels.
                // Eliminates Parallel.For overhead (was ~1-3 KB TPL alloc per call).
                if (input.RequiresGrad)
                {
                    LinearKernels.BackwardInput(
                        output.GradView.AsReadOnlySpan(),
                        weights.DataView.AsReadOnlySpan(),
                        input.GradView.AsSpan(),
                        batchSize, inputSize, outputSize);
                }

                if (weights.RequiresGrad)
                {
                    LinearKernels.AccumulateWeightGrad(
                        input.DataView.AsReadOnlySpan(),
                        output.GradView.AsReadOnlySpan(),
                        weights.GradView.AsSpan(),
                        batchSize, inputSize, outputSize);
                }
            }
            else
            {
                // Large matrix: existing parallel MatMul.
                MatMulBackward(input, weights, output);
            }

            // Bias grad always sequential (outputSize is small).
            if (bias.RequiresGrad)
            {
                LinearKernels.AccumulateBiasGrad(
                    output.GradView.AsReadOnlySpan(),
                    bias.GradView.AsSpan(),
                    batchSize, outputSize);
            }
        }
    }
}