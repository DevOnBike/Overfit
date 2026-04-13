// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Core
{
    public static class TensorMath
    {
        private const long ParallelThreshold = 4096;
        private const int StackAllocThreshold = 1024;

        // ====================================================================
        // HELPER ALLOCATOR
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static FastTensor<float> AllocateLike(AutogradNode node, bool clearMemory = true)
        {
            var v = node.DataView;
            return v.Rank switch
            {
                1 => new FastTensor<float>(v.GetDim(0), clearMemory),
                2 => new FastTensor<float>(v.GetDim(0), v.GetDim(1), clearMemory),
                3 => new FastTensor<float>(v.GetDim(0), v.GetDim(1), v.GetDim(2), clearMemory),
                4 => new FastTensor<float>(v.GetDim(0), v.GetDim(1), v.GetDim(2), v.GetDim(3), clearMemory),
                _ => throw new InvalidOperationException("Nieobsługiwany wymiar")
            };
        }

        // ====================================================================
        // 1. CORE ALGEBRA (Parallel Safe - No Span Capturing)
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

        public static AutogradNode AddBias(ComputationGraph graph, AutogradNode input, AutogradNode bias)
        {
            int N = input.DataView.GetDim(0), C = input.DataView.GetDim(1);
            var resD = new FastTensor<float>(N, C, clearMemory: false);

            Parallel.For(0, N, i =>
            {
                // Widoki są generowane bezpiecznie WNĘTRZU lambdy
                TensorPrimitives.Add(input.DataView.AsReadOnlySpan().Slice(i * C, C), bias.DataView.AsReadOnlySpan(), resD.GetView().AsSpan().Slice(i * C, C));
            });

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
            if (bias.RequiresGrad)
            {
                var bG = bias.GradView.AsSpan();
                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Add(bG, output.GradView.AsReadOnlySpan().Slice(i * C, C), bG);
                }
            }
        }

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
            var C = new FastTensor<float>(aR, bC, clearMemory: true);

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
                        if (rA[k] != 0f)
                        {
                            TensorPrimitives.MultiplyAdd(bS.Slice(k * bC, bC), rA[k], rC, rC);
                        }
                    }
                });
            }
            return C;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MatMulRawSeq(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, int aR, int aC, int bC, Span<float> cS)
        {
            cS.Clear();

            for (var i = 0; i < aR; i++)
            {
                var rC = cS.Slice(i * bC, bC);
                var rA = aS.Slice(i * aC, aC);
                for (var k = 0; k < aC; k++)
                {
                    if (rA[k] != 0f)
                    {
                        TensorPrimitives.MultiplyAdd(bS.Slice(k * bC, bC), rA[k], rC, rC);
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

        public static void MatMulAdd_A_BT_Raw(AutogradNode A, bool aGrad, AutogradNode B, bool bGrad, AutogradNode C, bool cGrad)
        {
            var N = (aGrad ? A.GradView : A.DataView).GetDim(0);
            var K = (aGrad ? A.GradView : A.DataView).GetDim(1);
            var M = (bGrad ? B.GradView : B.DataView).GetDim(0);

            if ((long)N * K * M < ParallelThreshold)
            {
                var aS = (aGrad ? A.GradView : A.DataView).AsReadOnlySpan();
                var bS = (bGrad ? B.GradView : B.DataView).AsReadOnlySpan();
                var cS = (cGrad ? C.GradView : C.DataView).AsSpan();
                MatMulAdd_A_BT_Seq(aS, bS, cS, N, K, M);
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    var aS = (aGrad ? A.GradView : A.DataView).AsReadOnlySpan();
                    var bS = (bGrad ? B.GradView : B.DataView).AsReadOnlySpan();
                    var cS = (cGrad ? C.GradView : C.DataView).AsSpan();

                    var rA = aS.Slice(i * K, K);
                    var rC = cS.Slice(i * M, M);
                    for (var j = 0; j < M; j++)
                    {
                        rC[j] += TensorPrimitives.Dot(rA, bS.Slice(j * K, K));
                    }
                });
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MatMulAdd_A_BT_Seq(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, Span<float> cS, int N, int K, int M)
        {
            for (var i = 0; i < N; i++)
            {
                var rA = aS.Slice(i * K, K); var rC = cS.Slice(i * M, M);
                for (var j = 0; j < M; j++)
                {
                    rC[j] += TensorPrimitives.Dot(rA, bS.Slice(j * K, K));
                }
            }
        }

        public static void MatMulAdd_AT_B_Raw(AutogradNode A, bool aGrad, AutogradNode B, bool bGrad, AutogradNode C, bool cGrad)
        {
            var K = (aGrad ? A.GradView : A.DataView).GetDim(0);
            var N = (aGrad ? A.GradView : A.DataView).GetDim(1);
            var M = (bGrad ? B.GradView : B.DataView).GetDim(1);

            if ((long)N * K * M < ParallelThreshold)
            {
                var aS = (aGrad ? A.GradView : A.DataView).AsReadOnlySpan();
                var bS = (bGrad ? B.GradView : B.DataView).AsReadOnlySpan();
                var cS = (cGrad ? C.GradView : C.DataView).AsSpan();
                MatMulAdd_AT_B_Seq(aS, bS, cS, K, N, M);
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    var aS = (aGrad ? A.GradView : A.DataView).AsReadOnlySpan();
                    var bS = (bGrad ? B.GradView : B.DataView).AsReadOnlySpan();
                    var cS = (cGrad ? C.GradView : C.DataView).AsSpan();

                    var rC = cS.Slice(i * M, M);
                    for (var k = 0; k < K; k++)
                    {
                        var vA = aS[k * N + i];
                        if (vA != 0f)
                        {
                            TensorPrimitives.MultiplyAdd(bS.Slice(k * M, M), vA, rC, rC);
                        }
                    }
                });
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MatMulAdd_AT_B_Seq(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, Span<float> cS, int K, int N, int M)
        {
            for (var i = 0; i < N; i++)
            {
                var rC = cS.Slice(i * M, M);
                for (var k = 0; k < K; k++)
                {
                    var vA = aS[k * N + i];
                    if (vA != 0f)
                    {
                        TensorPrimitives.MultiplyAdd(bS.Slice(k * M, M), vA, rC, rC);
                    }
                }
            }
        }

        // ====================================================================
        // 2. CNN OPERATIONS (NCHW)
        // ====================================================================

        public static AutogradNode Conv2D(ComputationGraph graph, AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            int outH = h - k + 1, outW = w - k + 1, batchSize = input.DataView.GetDim(0), kSqInC = k * k * inC, colS = kSqInC * outH * outW;
            var workspaceArr = ArrayPool<float>.Shared.Rent(batchSize * colS);
            var resultData = new FastTensor<float>(batchSize, outC, outH, outW, clearMemory: false);
            try
            {
                Parallel.For(0, batchSize, n =>
                {
                    var w2D = weights.DataView.Reshape(outC, kSqInC);
                    var colS_n = workspaceArr.AsSpan(n * colS, colS);
                    Im2Col(input.DataView.AsReadOnlySpan().Slice(n * inC * h * w, inC * h * w), inC, h, w, k, 1, 0, colS_n);
                    MatMulRawSeq(w2D.AsReadOnlySpan(), colS_n, outC, kSqInC, outH * outW, resultData.GetView().AsSpan().Slice(n * outC * outH * outW, outC * outH * outW));
                });
            }
            finally { ArrayPool<float>.Shared.Return(workspaceArr); }

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || weights.RequiresGrad);
            if (outputNode.RequiresGrad)
            {
                graph?.Record(OpCode.Conv2D, outputNode, input, weights, inC, outC, h, w, k);
            }
            return outputNode;
        }

        public static void Conv2DBackward(AutogradNode input, AutogradNode weights, AutogradNode output, int inC, int outC, int h, int w, int k)
        {
            if (!input.RequiresGrad && !weights.RequiresGrad)
            {
                return;
            }

            int outH = h - k + 1, outW = w - k + 1, batchSize = input.DataView.GetDim(0), kSqInC = k * k * inC, K = outH * outW;
            FastTensor<float> w2DTContig = null;

            if (input.RequiresGrad)
            {
                var w2D = weights.DataView.Reshape(outC, kSqInC);
                w2DTContig = FastTensor<float>.FromView(w2D.Transpose2D());
            }

            var weightLock = new object();
            Parallel.For(0, batchSize,
                localInit: () => (ArrayPool<float>.Shared.Rent(kSqInC * K), ArrayPool<float>.Shared.Rent(outC * K), input.RequiresGrad ? ArrayPool<float>.Shared.Rent(kSqInC * K) : null, weights.RequiresGrad ? new FastTensor<float>(outC, kSqInC, clearMemory: true) : null),
                body: (n, loopState, ws) =>
                {
                    var colS = ws.Item1.AsSpan(0, kSqInC * K); var outGS = ws.Item2.AsSpan(0, outC * K);
                    Im2Col(input.DataView.AsReadOnlySpan().Slice(n * inC * h * w, inC * h * w), inC, h, w, k, 1, 0, colS);
                    output.GradView.AsReadOnlySpan().Slice(n * outC * K, outC * K).CopyTo(outGS);

                    if (weights.RequiresGrad)
                    {
                        var dwS = ws.Item4.GetView().AsSpan();
                        for (var i = 0; i < outC; i++)
                        {
                            var rGS = outGS.Slice(i * K, K);
                            for (var j = 0; j < kSqInC; j++)
                            {
                                dwS[i * kSqInC + j] += TensorPrimitives.Dot(rGS, colS.Slice(j * K, K));
                            }
                        }
                    }
                    if (input.RequiresGrad)
                    {
                        var dColS = ws.Item3.AsSpan(0, kSqInC * K);
                        MatMulRawSeq(w2DTContig.GetView().AsReadOnlySpan(), outGS, kSqInC, outC, K, dColS);
                        Col2Im(dColS, inC, h, w, k, 1, 0, input.GradView.AsSpan().Slice(n * inC * h * w, inC * h * w));
                    }
                    return ws;
                },
                localFinally: ws =>
                {
                    ArrayPool<float>.Shared.Return(ws.Item1); ArrayPool<float>.Shared.Return(ws.Item2);
                    if (ws.Item3 != null)
                    {
                        ArrayPool<float>.Shared.Return(ws.Item3);
                    }
                    if (ws.Item4 != null)
                    {
                        lock (weightLock)
                        {
                            TensorPrimitives.Add(weights.GradView.AsSpan(), ws.Item4.GetView().AsReadOnlySpan(), weights.GradView.AsSpan());
                        }
                        ws.Item4.Dispose();
                    }
                });
            w2DTContig?.Dispose();
        }

        // ====================================================================
        // 3. ACTIVATIONS, POOLING & DROPOUT
        // ====================================================================

        public static AutogradNode ReLU(ComputationGraph graph, AutogradNode input)
        {
            var res = AllocateLike(input, false);

            var inSpan = input.DataView.AsReadOnlySpan();
            var outSpan = res.GetView().AsSpan();

            // TensorPrimitives w .NET 8 to najszybsza opcja dla Forward. Zostawiamy.
            TensorPrimitives.Max(inSpan, 0f, outSpan);

            var output = new AutogradNode(res, input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.ReLU, output, input);
            }

            return output;
        }

        public static void ReluBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var inS = input.DataView.AsReadOnlySpan();
            var goS = output.GradView.AsReadOnlySpan();
            var giS = input.GradView.AsSpan();

            var i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                var vCount = Vector<float>.Count;
                var vZero = Vector<float>.Zero;

                for (; i <= inS.Length - vCount; i += vCount)
                {
                    var vIn = new Vector<float>(inS.Slice(i));
                    var vGo = new Vector<float>(goS.Slice(i));
                    var vGi = new Vector<float>(giS.Slice(i));

                    // Krok 1: Maska bitowa (zwraca Vector<int>)
                    var vMask = Vector.GreaterThan(vIn, vZero);

                    // Krok 2: ConditionalSelect natywnie radzi sobie z maską int dla wektorów float.
                    // Tam gdzie maska ma jedynki, bierze vGo. Tam gdzie zera, bierze vZero.
                    var vGradToPass = Vector.ConditionalSelect(vMask, vGo, vZero);

                    // Krok 3: Akumulacja i zapis
                    (vGi + vGradToPass).CopyTo(giS.Slice(i));
                }
            }

            // Resztówka (dla elementów, które nie zmieściły się w pełnym wektorze)
            for (; i < inS.Length; i++)
            {
                if (inS[i] > 0f)
                {
                    giS[i] += goS[i];
                }
            }
        }

        public static AutogradNode Sigmoid(ComputationGraph graph, AutogradNode input)
        {
            var res = AllocateLike(input, false);
            TensorPrimitives.Sigmoid(input.DataView.AsReadOnlySpan(), res.GetView().AsSpan());
            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Sigmoid, output, input);
            }
            return output;
        }

        public static void SigmoidBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }
            var outS = output.DataView.AsReadOnlySpan(); var ogS = output.GradView.AsReadOnlySpan(); var igS = input.GradView.AsSpan();

            for (var i = 0; i < igS.Length; i += StackAllocThreshold)
            {
                var c = Math.Min(StackAllocThreshold, igS.Length - i);
                using var buf = new NativeBuffer<float>(c);
                var b = buf.Span; var o = outS.Slice(i, c);

                TensorPrimitives.Subtract(1f, o, b); TensorPrimitives.Multiply(o, b, b);
                TensorPrimitives.MultiplyAdd(ogS.Slice(i, c), b, igS.Slice(i, c), igS.Slice(i, c));
            }
        }

        public static AutogradNode Tanh(ComputationGraph graph, AutogradNode input)
        {
            var res = AllocateLike(input, false);
            TensorPrimitives.Tanh(input.DataView.AsReadOnlySpan(), res.GetView().AsSpan());
            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Tanh, output, input);
            }
            return output;
        }

        public static void TanhBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }
            var outS = output.DataView.AsReadOnlySpan(); var ogS = output.GradView.AsReadOnlySpan(); var igS = input.GradView.AsSpan();

            for (var i = 0; i < igS.Length; i += StackAllocThreshold)
            {
                var c = Math.Min(StackAllocThreshold, igS.Length - i);
                using var buf = new NativeBuffer<float>(c);
                var b = buf.Span; var o = outS.Slice(i, c);

                TensorPrimitives.Multiply(o, o, b); TensorPrimitives.Subtract(1f, b, b);
                TensorPrimitives.MultiplyAdd(ogS.Slice(i, c), b, igS.Slice(i, c), igS.Slice(i, c));
            }
        }

        public static AutogradNode MaxPool2D(ComputationGraph graph, AutogradNode input, int channels, int h, int w, int pool)
        {
            int oH = h / pool, oW = w / pool, batch = input.DataView.GetDim(0);
            var res = new FastTensor<float>(batch, channels, oH, oW, clearMemory: false);
            var idx = new AutogradNode(new FastTensor<float>(batch, channels, oH, oW, clearMemory: false), false);

            Parallel.For(0, batch, n =>
            {
                ref var iR = ref MemoryMarshal.GetReference(input.DataView.AsReadOnlySpan());
                ref var oR = ref MemoryMarshal.GetReference(res.GetView().AsSpan());
                ref var xR = ref MemoryMarshal.GetReference(idx.DataView.AsSpan());

                for (var c = 0; c < channels; c++)
                {
                    for (var oh = 0; oh < oH; oh++)
                    {
                        for (var ow = 0; ow < oW; ow++)
                        {
                            var max = float.MinValue; var mI = -1;
                            for (var ph = 0; ph < pool; ph++)
                            {
                                for (var pw = 0; pw < pool; pw++)
                                {
                                    var aI = n * channels * h * w + c * h * w + (oh * pool + ph) * w + (ow * pool + pw);
                                    var v = Unsafe.Add(ref iR, aI); if (v > max) { max = v; mI = aI; }
                                }
                            }
                            var oI = n * channels * oH * oW + c * oH * oW + oh * oW + ow;
                            Unsafe.Add(ref oR, oI) = max; Unsafe.Add(ref xR, oI) = mI;
                        }
                    }
                }
            });
            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MaxPool2D, output, input, idx);
            }
            return output;
        }

        public static void MaxPool2DBackward(AutogradNode input, AutogradNode maxIndices, AutogradNode output)
        {
            ref var iG = ref MemoryMarshal.GetReference(input.GradView.AsSpan());
            ref var xG = ref MemoryMarshal.GetReference(maxIndices.DataView.AsReadOnlySpan());
            ref var oG = ref MemoryMarshal.GetReference(output.GradView.AsReadOnlySpan());

            for (var i = 0; i < maxIndices.DataView.Size; i++)
            {
                Unsafe.Add(ref iG, (int)Unsafe.Add(ref xG, i)) += Unsafe.Add(ref oG, i);
            }
        }

        public static AutogradNode GlobalAveragePool2D(ComputationGraph graph, AutogradNode input, int channels, int h, int w)
        {
            var res = new FastTensor<float>(input.DataView.GetDim(0), channels, clearMemory: false);
            float sz = h * w;

            Parallel.For(0, input.DataView.GetDim(0), n =>
            {
                for (var c = 0; c < channels; c++)
                {
                    res.GetView()[n, c] = TensorPrimitives.Sum(input.DataView.AsReadOnlySpan().Slice(n * channels * h * w + c * h * w, h * w)) / sz;
                }
            });

            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.GlobalAveragePool2D, output, input, null, h, w, channels);
            }
            return output;
        }

        public static void GlobalAvgPool2DBackward(AutogradNode input, AutogradNode output, int h, int w, int channels)
        {
            float sz = h * w;
            Parallel.For(0, input.DataView.GetDim(0), n =>
            {
                for (var c = 0; c < channels; c++)
                {
                    var s = input.GradView.AsSpan().Slice(n * channels * h * w + c * h * w, h * w);
                    TensorPrimitives.Add(s, output.GradView.AsReadOnlySpan()[n * channels + c] / sz, s);
                }
            });
        }

        public static AutogradNode Dropout(ComputationGraph graph, AutogradNode input, float probability, bool isTraining)
        {
            var resD = AllocateLike(input, false);
            var maskNode = new AutogradNode(AllocateLike(input, false), false);

            if (isTraining)
            {
                var sc = 1f / (1f - probability); var sz = input.DataView.Size; var thr = (byte)(probability * 255f);

                using var rndBuf = new NativeBuffer<byte>(sz);
                Random.Shared.NextBytes(rndBuf.Span);

                var mS = maskNode.DataView.AsSpan();
                for (var i = 0; i < sz; i++)
                {
                    mS[i] = rndBuf.Span[i] > thr ? sc : 0f;
                }

                TensorPrimitives.Multiply(input.DataView.AsReadOnlySpan(), mS, resD.GetView().AsSpan());
            }
            else
            {
                input.DataView.AsReadOnlySpan().CopyTo(resD.GetView().AsSpan());
            }
            var output = new AutogradNode(resD, input.RequiresGrad);
            if (output.RequiresGrad && isTraining)
            {
                graph?.Record(OpCode.Dropout, output, input, maskNode);
            }
            else
            {
                maskNode.Dispose();
            }

            return output;
        }

        public static void DropoutBackward(AutogradNode input, AutogradNode mask, AutogradNode output) =>
            TensorPrimitives.MultiplyAdd(output.GradView.AsReadOnlySpan(), mask.DataView.AsReadOnlySpan(), input.GradView.AsSpan(), input.GradView.AsSpan());

        // ====================================================================
        // 4. LOSS FUNCTIONS & BATCHNORM
        // ====================================================================

        public static AutogradNode SoftmaxCrossEntropy(ComputationGraph graph, AutogradNode logits, AutogradNode target)
        {
            int rows = logits.DataView.GetDim(0), cols = logits.DataView.GetDim(1); var total = 0f;
            var probs = new FastTensor<float>(rows, cols, clearMemory: false);

            for (var r = 0; r < rows; r++)
            {
                var pR = probs.GetView().AsSpan().Slice(r * cols, cols);
                TensorPrimitives.SoftMax(logits.DataView.AsReadOnlySpan().Slice(r * cols, cols), pR);
                for (var c = 0; c < cols; c++)
                {
                    if (target.DataView[r, c] > 0.5f)
                    {
                        total -= MathF.Log(pR[c] + 1e-15f);
                    }
                }
            }

            var resTensor = new FastTensor<float>(1, clearMemory: false);
            resTensor.GetView().AsSpan()[0] = total / rows;
            var output = new AutogradNode(resTensor, logits.RequiresGrad);

            if (logits.RequiresGrad)
            {
                graph?.Record(OpCode.SoftmaxCrossEntropy, output, logits, target, nodeContext: [new AutogradNode(probs, false)]);
            }
            else
            {
                probs.Dispose();
            }

            return output;
        }

        public static void SoftmaxCrossEntropyBackward(AutogradNode logits, AutogradNode target, AutogradNode output, AutogradNode probsNode)
        {
            int R = logits.DataView.GetDim(0), C = logits.DataView.GetDim(1); var s = output.GradView.AsReadOnlySpan()[0] / R;
            for (var r = 0; r < R; r++)
            {
                var pS = probsNode.DataView.AsReadOnlySpan().Slice(r * C, C); var tS = target.DataView.AsReadOnlySpan().Slice(r * C, C); var gS = logits.GradView.AsSpan().Slice(r * C, C);
                TensorPrimitives.MultiplyAdd(pS, s, gS, gS); TensorPrimitives.MultiplyAdd(tS, -s, gS, gS);
            }
        }

        public static AutogradNode MSELoss(ComputationGraph graph, AutogradNode prediction, AutogradNode target)
        {
            var sz = prediction.DataView.Size; float mse;
            using (var diffBuf = new NativeBuffer<float>(sz))
            {
                var dS = diffBuf.Span;
                TensorPrimitives.Subtract(prediction.DataView.AsReadOnlySpan(), target.DataView.AsReadOnlySpan(), dS);
                mse = TensorPrimitives.Dot(dS, dS) / sz;
            }
            var resTensor = new FastTensor<float>(1, clearMemory: false);
            resTensor.GetView().AsSpan()[0] = mse;

            var output = new AutogradNode(resTensor, prediction.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MseLoss, output, prediction, target);
            }
            return output;
        }

        public static void MSELossBackward(AutogradNode p, AutogradNode t, AutogradNode o)
        {
            var f = o.GradView.AsReadOnlySpan()[0] * (2f / p.DataView.Size);
            TensorPrimitives.MultiplyAdd(p.DataView.AsReadOnlySpan(), f, p.GradView.AsSpan(), p.GradView.AsSpan());
            TensorPrimitives.MultiplyAdd(t.DataView.AsReadOnlySpan(), -f, p.GradView.AsSpan(), p.GradView.AsSpan());
        }

        public static AutogradNode DirectionalLoss(ComputationGraph graph, AutogradNode prediction, AutogradNode target, float gamma = 10f)
        {
            var sz = prediction.DataView.Size; float loss;
            using (var tempBuf = new NativeBuffer<float>(sz))
            {
                var s = tempBuf.Span;
                TensorPrimitives.Subtract(prediction.DataView.AsReadOnlySpan(), target.DataView.AsReadOnlySpan(), s);
                var mse = TensorPrimitives.SumOfSquares(s);
                TensorPrimitives.Multiply(prediction.DataView.AsReadOnlySpan(), target.DataView.AsReadOnlySpan(), s); TensorPrimitives.Min(s, 0f, s);
                loss = (mse + TensorPrimitives.Sum(s) * -gamma) / sz;
            }
            var resTensor = new FastTensor<float>(1, clearMemory: false);
            resTensor.GetView().AsSpan()[0] = loss;

            var output = new AutogradNode(resTensor, prediction.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.DirectionalLoss, output, prediction, target, BitConverter.SingleToInt32Bits(gamma));
            }
            return output;
        }

        public static void DirectionalLossBackward(AutogradNode p, AutogradNode t, AutogradNode o, float gamma)
        {
            var s = o.GradView.AsReadOnlySpan()[0] / p.DataView.Size; var pG = p.GradView.AsSpan(); var pD = p.DataView.AsReadOnlySpan(); var tD = t.DataView.AsReadOnlySpan(); var i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                var vS = Vector<float>.Count;
                for (; i <= pD.Length - vS; i += vS)
                {
                    var vP = new Vector<float>(pD.Slice(i)); var vT = new Vector<float>(tD.Slice(i));
                    (new Vector<float>(pG.Slice(i)) + (vP - vT) * (2f * s) + Vector.ConditionalSelect(Vector.LessThan(vP * vT, Vector<float>.Zero), -vT * (gamma * s), Vector<float>.Zero)).CopyTo(pG.Slice(i));
                }
            }
            for (; i < pD.Length; i++)
            {
                pG[i] += 2f * (pD[i] - tD[i]) * s + (pD[i] * tD[i] < 0f ? -tD[i] * (gamma * s) : 0f);
            }
        }

        public static AutogradNode BatchNorm1D(ComputationGraph graph, AutogradNode input, AutogradNode gamma, AutogradNode beta, FastTensor<float> runningMean, FastTensor<float> runningVar, float momentum, float eps, bool isTraining)
        {
            int N = input.DataView.GetDim(0), C = input.DataView.GetDim(1); var outD = new FastTensor<float>(N, C, clearMemory: false);
            var mean = new AutogradNode(new FastTensor<float>(C, clearMemory: true), false);
            var invStd = new AutogradNode(new FastTensor<float>(C, clearMemory: true), false);

            if (isTraining)
            {
                var mS = mean.DataView.AsSpan();
                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Add(mS, input.DataView.AsReadOnlySpan().Slice(i * C, C), mS);
                }
                TensorPrimitives.Multiply(mS, 1f / N, mS);

                using var vB = new NativeBuffer<float>(C, clearMemory: true);
                using var tB = new NativeBuffer<float>(C);

                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Subtract(input.DataView.AsReadOnlySpan().Slice(i * C, C), mS, tB.Span);
                    TensorPrimitives.MultiplyAdd(tB.Span, tB.Span, vB.Span, vB.Span);
                }
                TensorPrimitives.Multiply(vB.Span, 1f / N, vB.Span);

                var rmS = runningMean.GetView().AsSpan(); var rvS = runningVar.GetView().AsSpan(); var ivS = invStd.DataView.AsSpan();
                TensorPrimitives.Multiply(rmS, 1f - momentum, rmS); TensorPrimitives.MultiplyAdd(mS, momentum, rmS, rmS);
                TensorPrimitives.Multiply(rvS, 1f - momentum, rvS); TensorPrimitives.MultiplyAdd(vB.Span, momentum, rvS, rvS);
                TensorPrimitives.Add(vB.Span, eps, ivS); TensorPrimitives.ReciprocalSqrt(ivS, ivS);
            }
            else
            {
                runningMean.GetView().AsReadOnlySpan().CopyTo(mean.DataView.AsSpan()); var ivS = invStd.DataView.AsSpan();
                TensorPrimitives.Add(runningVar.GetView().AsReadOnlySpan(), eps, ivS); TensorPrimitives.ReciprocalSqrt(ivS, ivS);
            }
            for (var i = 0; i < N; i++)
            {
                var oR = outD.GetView().AsSpan().Slice(i * C, C); TensorPrimitives.Subtract(input.DataView.AsReadOnlySpan().Slice(i * C, C), mean.DataView.AsReadOnlySpan(), oR);
                TensorPrimitives.Multiply(oR, invStd.DataView.AsReadOnlySpan(), oR); TensorPrimitives.MultiplyAdd(oR, gamma.DataView.AsReadOnlySpan(), beta.DataView.AsReadOnlySpan(), oR);
            }
            var output = new AutogradNode(outD, input.RequiresGrad);
            if (output.RequiresGrad && isTraining)
            {
                graph?.Record(OpCode.BatchNorm1D, output, input, null, 0, 0, 0, 0, 0, [gamma, beta, mean, invStd]);
            }
            return output;
        }

        public static void BatchNorm1DBackward(AutogradNode input, AutogradNode output, AutogradNode gamma, AutogradNode beta, AutogradNode mean, AutogradNode invStd)
        {
            if (!input.RequiresGrad && !gamma.RequiresGrad && !beta.RequiresGrad)
            {
                return;
            }

            int N = input.DataView.GetDim(0), C = input.DataView.GetDim(1);

            using var coeffBuf = new NativeBuffer<float>(C); var coeff = coeffBuf.Span;
            using var termBuf = new NativeBuffer<float>(C); var term = termBuf.Span;
            using var sDyBuf = new NativeBuffer<float>(C, true); var sDy = sDyBuf.Span;
            using var sDyXBuf = new NativeBuffer<float>(C, true); var sDyX = sDyXBuf.Span;
            using var xHRBuf = new NativeBuffer<float>(C); var xHR = xHRBuf.Span;

            TensorPrimitives.Multiply(gamma.DataView.AsReadOnlySpan(), invStd.DataView.AsReadOnlySpan(), coeff); TensorPrimitives.Multiply(coeff, 1f / N, coeff);
            for (var i = 0; i < N; i++)
            {
                var gR = output.GradView.AsReadOnlySpan().Slice(i * C, C); var iR = input.DataView.AsReadOnlySpan().Slice(i * C, C);
                TensorPrimitives.Subtract(iR, mean.DataView.AsReadOnlySpan(), xHR); TensorPrimitives.Multiply(xHR, invStd.DataView.AsReadOnlySpan(), xHR);
                TensorPrimitives.Add(sDy, gR, sDy); TensorPrimitives.MultiplyAdd(gR, xHR, sDyX, sDyX);
                if (beta.RequiresGrad)
                {
                    TensorPrimitives.Add(beta.GradView.AsSpan(), gR, beta.GradView.AsSpan());
                }
                if (gamma.RequiresGrad)
                {
                    TensorPrimitives.MultiplyAdd(gR, xHR, gamma.GradView.AsSpan(), gamma.GradView.AsSpan());
                }
            }

            if (input.RequiresGrad)
            {
                var iGS = input.GradView.AsSpan();
                for (var i = 0; i < N; i++)
                {
                    var gR = output.GradView.AsReadOnlySpan().Slice(i * C, C); var iGR = iGS.Slice(i * C, C); var iR = input.DataView.AsReadOnlySpan().Slice(i * C, C);
                    TensorPrimitives.Subtract(iR, mean.DataView.AsReadOnlySpan(), xHR); TensorPrimitives.Multiply(xHR, invStd.DataView.AsReadOnlySpan(), xHR);
                    TensorPrimitives.Multiply(gR, N, term); TensorPrimitives.Subtract(term, sDy, term);

                    using var tempXHatBuf = new NativeBuffer<float>(C); var tempXHat = tempXHatBuf.Span;
                    xHR.CopyTo(tempXHat);

                    TensorPrimitives.Multiply(tempXHat, sDyX, tempXHat); TensorPrimitives.Subtract(term, tempXHat, term);
                    TensorPrimitives.MultiplyAdd(coeff, term, iGR, iGR);
                }
            }
        }

        // ====================================================================
        // 5. BPTT & SEQUENCE (Fused LSTM, Repeat, Multiply, Slices)
        // ====================================================================

        public static (AutogradNode hNew, AutogradNode cNew) FusedLSTMStep(ComputationGraph graph, AutogradNode x, AutogradNode hPrev, AutogradNode cPrev, AutogradNode W, AutogradNode U, AutogradNode B)
        {
            int batchSize = x.DataView.GetDim(0), hS = hPrev.DataView.GetDim(1);
            var gD = MatMulRaw(x, W);
            using var uh = MatMulRaw(hPrev, U);

            var cnD = new FastTensor<float>(batchSize, hS, clearMemory: false);
            var hnD = new FastTensor<float>(batchSize, hS, clearMemory: false);

            Parallel.For(0, batchSize, b =>
            {
                var bg = gD.GetView().AsSpan().Slice(b * 4 * hS, 4 * hS);
                TensorPrimitives.Add(bg, uh.GetView().AsReadOnlySpan().Slice(b * 4 * hS, 4 * hS), bg); TensorPrimitives.Add(bg, B.DataView.AsReadOnlySpan(), bg);

                var f = bg.Slice(0, hS); var i = bg.Slice(hS, hS); var g = bg.Slice(2 * hS, hS); var o = bg.Slice(3 * hS, hS);
                TensorPrimitives.Sigmoid(f, f); TensorPrimitives.Sigmoid(i, i); TensorPrimitives.Tanh(g, g); TensorPrimitives.Sigmoid(o, o);

                var bcn = cnD.GetView().AsSpan().Slice(b * hS, hS);
                TensorPrimitives.Multiply(f, cPrev.DataView.AsReadOnlySpan().Slice(b * hS, hS), bcn); TensorPrimitives.MultiplyAdd(i, g, bcn, bcn);

                var bhn = hnD.GetView().AsSpan().Slice(b * hS, hS);
                TensorPrimitives.Tanh(bcn, bhn); TensorPrimitives.Multiply(o, bhn, bhn);
            });
            var req = x.RequiresGrad || hPrev.RequiresGrad || cPrev.RequiresGrad || W.RequiresGrad || U.RequiresGrad || B.RequiresGrad;
            var hNode = new AutogradNode(hnD, req); var cNode = new AutogradNode(cnD, req);

            if (graph != null && graph.IsRecording && req)
            {
                graph.Record(OpCode.FusedLSTMStep, hNode, x, hPrev, nodeContext: [cPrev, W, U, B, cNode, new AutogradNode(gD, false)]);
            }
            else
            {
                gD.Dispose();
            }

            return (hNode, cNode);
        }

        public static void FusedLSTMStepBackward(AutogradNode x, AutogradNode hPrev, AutogradNode hNew, AutogradNode[] ctx)
        {
            var cPrev = ctx[0]; var W = ctx[1]; var U = ctx[2]; var B = ctx[3]; var cNew = ctx[4]; var gates = ctx[5];
            int batchSize = x.DataView.GetDim(0), hS = hPrev.DataView.GetDim(1);
            using var dGNode = new AutogradNode(new FastTensor<float>(batchSize, 4 * hS, clearMemory: false));

            Parallel.For(0, batchSize, localInit: () => ArrayPool<float>.Shared.Rent(hS * 4),
                body: (b, state, arr) =>
                {
                    var dGS = dGNode.DataView.AsSpan().Slice(b * 4 * hS, 4 * hS);
                    var gs = gates.DataView.AsReadOnlySpan().Slice(b * 4 * hS, 4 * hS); var f = gs.Slice(0, hS); var i = gs.Slice(hS, hS); var g = gs.Slice(2 * hS, hS); var o = gs.Slice(3 * hS, hS);
                    var dh = hNew.GradView.AsReadOnlySpan().Slice(b * hS, hS); var dc = cNew.GradView.AsSpan().Slice(b * hS, hS);
                    var tS = arr.AsSpan(); var tanhC = tS.Slice(0, hS); var t1 = tS.Slice(hS, hS); var t2 = tS.Slice(2 * hS, hS);

                    TensorPrimitives.Tanh(cNew.DataView.AsReadOnlySpan().Slice(b * hS, hS), tanhC);
                    TensorPrimitives.Subtract(1f, o, t1); TensorPrimitives.Multiply(o, t1, t1); TensorPrimitives.Multiply(dh, tanhC, t2); TensorPrimitives.Multiply(t2, t1, dGS.Slice(3 * hS, hS));
                    TensorPrimitives.Multiply(tanhC, tanhC, t1); TensorPrimitives.Subtract(1f, t1, t1); TensorPrimitives.Multiply(dh, o, t2); TensorPrimitives.MultiplyAdd(t2, t1, dc, dc);
                    TensorPrimitives.Multiply(g, g, t1); TensorPrimitives.Subtract(1f, t1, t1); TensorPrimitives.Multiply(dc, i, t2); TensorPrimitives.Multiply(t2, t1, dGS.Slice(2 * hS, hS));
                    TensorPrimitives.Subtract(1f, i, t1); TensorPrimitives.Multiply(i, t1, t1); TensorPrimitives.Multiply(dc, g, t2); TensorPrimitives.Multiply(t2, t1, dGS.Slice(hS, hS));
                    TensorPrimitives.Subtract(1f, f, t1); TensorPrimitives.Multiply(f, t1, t1); TensorPrimitives.Multiply(dc, cPrev.DataView.AsReadOnlySpan().Slice(b * hS, hS), t2); TensorPrimitives.Multiply(t2, t1, dGS.Slice(0, hS));

                    if (cPrev.RequiresGrad) { var dcp = cPrev.GradView.AsSpan().Slice(b * hS, hS); TensorPrimitives.MultiplyAdd(dc, f, dcp, dcp); }
                    return arr;
                }, localFinally: arr => ArrayPool<float>.Shared.Return(arr));

            if (x.RequiresGrad)
            {
                MatMulAdd_A_BT_Raw(dGNode, false, W, false, x, true);
            }
            if (W.RequiresGrad)
            {
                MatMulAdd_AT_B_Raw(x, false, dGNode, false, W, true);
            }
            if (hPrev.RequiresGrad)
            {
                MatMulAdd_A_BT_Raw(dGNode, false, U, false, hPrev, true);
            }
            if (U.RequiresGrad)
            {
                MatMulAdd_AT_B_Raw(hPrev, false, dGNode, false, U, true);
            }

            if (B.RequiresGrad)
            {
                var dbS = B.GradView.AsSpan(); var dGS_full = dGNode.DataView.AsReadOnlySpan();
                for (var b = 0; b < batchSize; b++)
                {
                    TensorPrimitives.Add(dbS, dGS_full.Slice(b * 4 * hS, 4 * hS), dbS);
                }
            }
        }

        public static AutogradNode RepeatVector(ComputationGraph graph, AutogradNode input, int seqLen)
        {
            int batch = input.DataView.GetDim(0), hS = input.DataView.GetDim(1);
            var res = new FastTensor<float>(batch, seqLen, hS, clearMemory: false);
            for (var b = 0; b < batch; b++)
            {
                var src = input.DataView.AsReadOnlySpan().Slice(b * hS, hS);
                for (var t = 0; t < seqLen; t++)
                {
                    src.CopyTo(res.GetView().AsSpan().Slice(b * seqLen * hS + t * hS, hS));
                }
            }
            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.RepeatVector, output, input, null, seqLen, hS);
            }
            return output;
        }

        public static void RepeatVectorBackward(AutogradNode input, AutogradNode output, int seqLen, int hS)
        {
            if (!input.RequiresGrad)
            {
                return;
            }
            var iGS = input.GradView.AsSpan(); var ogS = output.GradView.AsReadOnlySpan();
            for (var b = 0; b < input.DataView.GetDim(0); b++)
            {
                var dst = iGS.Slice(b * hS, hS);
                for (var t = 0; t < seqLen; t++)
                {
                    TensorPrimitives.Add(dst, ogS.Slice(b * seqLen * hS + t * hS, hS), dst);
                }
            }
        }

        public static AutogradNode Multiply(ComputationGraph graph, AutogradNode a, AutogradNode b)
        {
            var res = AllocateLike(a, false);
            TensorPrimitives.Multiply(a.DataView.AsReadOnlySpan(), b.DataView.AsReadOnlySpan(), res.GetView().AsSpan());
            var output = new AutogradNode(res, a.RequiresGrad || b.RequiresGrad);
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

        public static AutogradNode GateSlice(ComputationGraph graph, AutogradNode gates, int hiddenSize, int gateIndex)
        {
            var res = new FastTensor<float>(gates.DataView.GetDim(0), hiddenSize, clearMemory: false);
            int batch = gates.DataView.GetDim(0), stride = 4 * hiddenSize, offset = gateIndex * hiddenSize;
            for (var b = 0; b < batch; b++)
            {
                gates.DataView.AsReadOnlySpan().Slice(b * stride + offset, hiddenSize).CopyTo(res.GetView().AsSpan().Slice(b * hiddenSize, hiddenSize));
            }
            var output = new AutogradNode(res, gates.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.GateSlice, output, gates, null, gateIndex, hiddenSize);
            }
            return output;
        }

        public static void GateSliceBackward(AutogradNode gates, AutogradNode output, int hiddenSize, int gateIndex)
        {
            if (!gates.RequiresGrad)
            {
                return;
            }
            int batch = gates.DataView.GetDim(0), offset = gateIndex * hiddenSize, stride = 4 * hiddenSize;
            for (var b = 0; b < batch; b++)
            {
                var dst = gates.GradView.AsSpan().Slice(b * stride + offset, hiddenSize);
                TensorPrimitives.Add(dst, output.GradView.AsReadOnlySpan().Slice(b * hiddenSize, hiddenSize), dst);
            }
        }

        public static void TimestepSliceBackward(AutogradNode input, AutogradNode output, int t, int seqLen, int inputSize)
        {
            if (!input.RequiresGrad)
            {
                return;
            }
            for (var b = 0; b < input.DataView.GetDim(0); b++)
            {
                var dst = input.GradView.AsSpan().Slice(b * seqLen * inputSize + t * inputSize, inputSize);
                TensorPrimitives.Add(dst, output.GradView.AsReadOnlySpan().Slice(b * inputSize, inputSize), dst);
            }
        }

        public static void StackTimestepsBackward(AutogradNode[] allH, AutogradNode output, int batch, int seqLen, int hiddenSize)
        {
            for (var t = 0; t < seqLen; t++)
            {
                if (!allH[t].RequiresGrad)
                {
                    continue;
                }
                for (var b = 0; b < batch; b++)
                {
                    var dst = allH[t].GradView.AsSpan().Slice(b * hiddenSize, hiddenSize);
                    TensorPrimitives.Add(dst, output.GradView.AsReadOnlySpan().Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize), dst);
                }
            }
        }

        // ====================================================================
        // 6. GRAPH HELPERS
        // ====================================================================

        public static AutogradNode Reshape(ComputationGraph graph, AutogradNode input, params int[] newShape)
        {
            // Obliczamy całkowitą liczbę elementów w nowym kształcie
            var totalNewElements = 1;
            foreach (var dim in newShape)
            {
                totalNewElements *= dim;
            }

            // Tworzymy widok 2D (Batch, Reszta), bo Twój TensorView póki co pewnie tylko to wspiera stabilnie
            var newView = input.DataView.Reshape(newShape[0], totalNewElements / newShape[0]);
            var resD = FastTensor<float>.FromView(newView);
            var output = new AutogradNode(resD, input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Reshape, output, input);
            }
            return output;
        }

        public static void ReshapeBackward(AutogradNode input, AutogradNode output) =>
            TensorPrimitives.Add(input.GradView.AsSpan(), output.GradView.AsReadOnlySpan(), input.GradView.AsSpan());

        public static AutogradNode Linear(ComputationGraph graph, AutogradNode input, AutogradNode weights, AutogradNode bias) =>
            AddBias(graph, MatMul(graph, input, weights), bias);

        public static void Im2Col(ReadOnlySpan<float> input, int channels, int h, int w, int k, int s, int p, Span<float> output)
        {
            int oH = (h + 2 * p - k) / s + 1, oW = (w + 2 * p - k) / s + 1;
            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < k; kh++)
                {
                    for (var kw = 0; kw < k; kw++)
                    {
                        var rO = (c * k * k + kh * k + kw) * oH * oW;
                        for (var y = 0; y < oH; y++)
                        {
                            var i = y * s - p + kh;
                            if (i >= 0 && i < h)
                            {
                                var iO = c * h * w + i * w;
                                for (var x = 0; x < oW; x++) { var j = x * s - p + kw; output[rO + y * oW + x] = (j >= 0 && j < w) ? input[iO + j] : 0f; }
                            }
                            else
                            {
                                output.Slice(rO + y * oW, oW).Clear();
                            }
                        }
                    }
                }
            }
        }

        public static void Col2Im(ReadOnlySpan<float> col, int channels, int h, int w, int k, int s, int p, Span<float> gI)
        {
            int oH = (h + 2 * p - k) / s + 1, oW = (w + 2 * p - k) / s + 1;
            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < k; kh++)
                {
                    for (var kw = 0; kw < k; kw++)
                    {
                        var rO = (c * k * k + kh * k + kw) * oH * oW;
                        for (var y = 0; y < oH; y++)
                        {
                            var i = y * s - p + kh;
                            if (i >= 0 && i < h)
                            {
                                var iO = c * h * w + i * w;
                                for (var x = 0; x < oW; x++)
                                {
                                    var j = x * s - p + kw;
                                    if (j >= 0 && j < w)
                                    {
                                        gI[iO + j] += col[rO + y * oW + x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}