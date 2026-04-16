// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // ADD
        // ====================================================================

        public static AutogradNode Add(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var resD = AllocateLike(left, false);
            TensorPrimitives.Add(left.DataView.AsReadOnlySpan(), right.DataView.AsReadOnlySpan(), resD.GetView().AsSpan());

            var output = new AutogradNode(resD, left.RequiresGrad || right.RequiresGrad);
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
            var resD = AllocateLike(left, false);
            TensorPrimitives.Subtract(left.DataView.AsReadOnlySpan(), right.DataView.AsReadOnlySpan(), resD.GetView().AsSpan());

            var output = new AutogradNode(resD, left.RequiresGrad || right.RequiresGrad);

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
            int N = input.DataView.GetDim(0), C = input.DataView.GetDim(1);
            var resD = new FastTensor<float>(N, C, false);

            if (N < BatchSequentialThreshold)
            {
                var inS = input.DataView.AsReadOnlySpan();
                var bS = bias.DataView.AsReadOnlySpan();
                var outS = resD.GetView().AsSpan();

                for (var i = 0; i < N; i++)
                {
                    Simd.Add(inS.Slice(i * C, C), bS, outS.Slice(i * C, C));
                }
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    Simd.Add(
                        input.DataView.AsReadOnlySpan().Slice(i * C, C),
                        bias.DataView.AsReadOnlySpan(),
                        resD.GetView().AsSpan().Slice(i * C, C));
                });
            }

            var output = new AutogradNode(resD, input.RequiresGrad || bias.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.AddBias, output, input, bias);
            }

            return output;
        }

        public static void AddBiasBackward(AutogradNode input, AutogradNode bias, AutogradNode output)
        {
            int N = input.DataView.GetDim(0), C = input.DataView.GetDim(1);

            if (input.RequiresGrad)
            {
                TensorPrimitives.Add(input.GradView.AsSpan(), output.GradView.AsReadOnlySpan(), input.GradView.AsSpan());
            }

            if (!bias.RequiresGrad)
            {
                return;
            }

            if (N < BatchSequentialThreshold)
            {
                var bG = bias.GradView.AsSpan();
                var oG = output.GradView.AsReadOnlySpan();
                for (var i = 0; i < N; i++)
                {
                    Simd.Add(bG, oG.Slice(i * C, C), bG);
                }

                return;
            }

            var partials = new ConcurrentBag<FastTensor<float>>();

            Parallel.For(0, N,
                () => new FastTensor<float>(C),
                (i, state, localGrad) =>
                {
                    Simd.Add(
                        localGrad.GetView().AsReadOnlySpan(),
                        output.GradView.AsReadOnlySpan().Slice(i * C, C),
                        localGrad.GetView().AsSpan());
                    return localGrad;
                },
                localGrad =>
                {
                    partials.Add(localGrad);
                });

            var biasGrad = bias.GradView.AsSpan();
            foreach (var partial in partials)
            {
                try
                {
                    Simd.Add(biasGrad, partial.GetView().AsReadOnlySpan(), biasGrad);
                }
                finally
                {
                    partial.Dispose();
                }
            }
        }

        // ====================================================================
        // MATMUL
        // ====================================================================

        public static AutogradNode MatMul(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var resD = MatMulRaw(left, right);
            var output = new AutogradNode(resD, left.RequiresGrad || right.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MatMul, output, left, right);
            }

            return output;
        }

        public static FastTensor<float> MatMulRaw(AutogradNode A, AutogradNode B)
        {
            int aR = A.DataView.GetDim(0), aC = A.DataView.GetDim(1), bC = B.DataView.GetDim(1);
            var C = new FastTensor<float>(aR, bC);

            if ((long)aR * aC * bC < ParallelThreshold)
            {
                MatMulRawSeq(A.DataView.AsReadOnlySpan(), B.DataView.AsReadOnlySpan(), aR, aC, bC, C.GetView().AsSpan());
            }
            else
            {
                Parallel.For(0, aR, i =>
                {
                    var rC = C.GetView().AsSpan().Slice(i * bC, bC);
                    var rA = A.DataView.AsReadOnlySpan().Slice(i * aC, aC);
                    var bS = B.DataView.AsReadOnlySpan();

                    for (var k = 0; k < aC; k++)
                    {
                        var aVal = rA[k];
                        if (aVal != 0f)
                        {
                            Simd.MulAdd(bS.Slice(k * bC, bC), aVal, rC);
                        }
                    }
                });
            }

            return C;
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
            MatMulAdd_A_BT_Raw(output, false, b, false, a, true);
            MatMulAdd_AT_B_Raw(a, false, output, false, b, true);
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

            int N = A.DataView.GetDim(0), K = A.DataView.GetDim(1), M = B.DataView.GetDim(0);

            if ((long)N * K * M < ParallelThreshold)
            {
                MatMulAdd_A_BT_Seq(
                    aGrad ? A.GradView.AsReadOnlySpan() : A.DataView.AsReadOnlySpan(),
                    bGrad ? B.GradView.AsReadOnlySpan() : B.DataView.AsReadOnlySpan(),
                    C.GradView.AsSpan(), N, K, M);
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    var aRow = (aGrad ? A.GradView.AsReadOnlySpan() : A.DataView.AsReadOnlySpan()).Slice(i * K, K);
                    var cRow = C.GradView.AsSpan().Slice(i * M, M);
                    var bData = bGrad ? B.GradView.AsReadOnlySpan() : B.DataView.AsReadOnlySpan();

                    for (var j = 0; j < M; j++)
                    {
                        cRow[j] += Simd.Dot(aRow, bData.Slice(j * K, K));
                    }
                });
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

            int K = A.DataView.GetDim(0), N = A.DataView.GetDim(1), M = B.DataView.GetDim(1);

            if ((long)K * N * M < ParallelThreshold)
            {
                MatMulAdd_AT_B_Seq(
                    aGrad ? A.GradView.AsReadOnlySpan() : A.DataView.AsReadOnlySpan(),
                    bGrad ? B.GradView.AsReadOnlySpan() : B.DataView.AsReadOnlySpan(),
                    C.GradView.AsSpan(), K, N, M);
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    var cRow = C.GradView.AsSpan().Slice(i * M, M);
                    var aData = aGrad ? A.GradView.AsReadOnlySpan() : A.DataView.AsReadOnlySpan();
                    var bData = bGrad ? B.GradView.AsReadOnlySpan() : B.DataView.AsReadOnlySpan();

                    for (var k = 0; k < K; k++)
                    {
                        var aVal = aData[k * N + i];
                        if (aVal != 0f)
                        {
                            Simd.MulAdd(bData.Slice(k * M, M), aVal, cRow);
                        }
                    }
                });
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
            var resD = AllocateLike(a, false);
            TensorPrimitives.Multiply(a.DataView.AsReadOnlySpan(), b.DataView.AsReadOnlySpan(), resD.GetView().AsSpan());

            var output = new AutogradNode(resD, a.RequiresGrad || b.RequiresGrad);
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

        public static AutogradNode Linear(ComputationGraph graph, AutogradNode input, AutogradNode weights, AutogradNode bias)
        {
            return AddBias(graph, MatMul(graph, input, weights), bias);
        }
    }
}
