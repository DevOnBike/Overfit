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
        // 1. CORE ALGEBRA (Parallel Safe - No Span Capturing)
        // ====================================================================

        public static AutogradNode Add(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var resD = FastTensor<float>.SameShape(left.Data);
            TensorPrimitives.Add(left.Data.AsReadOnlySpan(), right.Data.AsReadOnlySpan(), resD.AsSpan());
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
                TensorPrimitives.Add(a.Grad.AsSpan(), output.Grad.AsReadOnlySpan(), a.Grad.AsSpan());
            }
            if (b.RequiresGrad)
            {
                TensorPrimitives.Add(b.Grad.AsSpan(), output.Grad.AsReadOnlySpan(), b.Grad.AsSpan());
            }
        }

        public static AutogradNode Subtract(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var resD = FastTensor<float>.SameShape(left.Data);
            TensorPrimitives.Subtract(left.Data.AsReadOnlySpan(), right.Data.AsReadOnlySpan(), resD.AsSpan());
            var output = new AutogradNode(resD, left.RequiresGrad || right.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Subtract, output, left, right);
            }
            return output;
        }

        public static AutogradNode AddBias(ComputationGraph graph, AutogradNode input, AutogradNode bias)
        {
            int N = input.Data.GetDim(0), C = input.Data.GetDim(1);
            var resD = new FastTensor<float>(N, C);
            Parallel.For(0, N, i =>
            {
                // LATE SPANNING: Tworzymy spany wewnątrz lambdy (naprawia CS9108)
                TensorPrimitives.Add(input.Data.AsReadOnlySpan().Slice(i * C, C), bias.Data.AsReadOnlySpan(), resD.AsSpan().Slice(i * C, C));
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
            int N = input.Data.GetDim(0), C = input.Data.GetDim(1);
            if (input.RequiresGrad)
            {
                TensorPrimitives.Add(input.Grad.AsSpan(), output.Grad.AsReadOnlySpan(), input.Grad.AsSpan());
            }
            if (bias.RequiresGrad)
            {
                var bG = bias.Grad.AsSpan();
                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Add(bG, output.Grad.AsReadOnlySpan().Slice(i * C, C), bG);
                }
            }
        }

        public static AutogradNode MatMul(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var resD = MatMulRaw(left.Data, right.Data);
            var output = new AutogradNode(resD, left.RequiresGrad || right.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MatMul, output, left, right);
            }
            return output;
        }

        public static FastTensor<float> MatMulRaw(FastTensor<float> A, FastTensor<float> B)
        {
            int aR = A.GetDim(0), aC = A.GetDim(1), bC = B.GetDim(1);
            var C = new FastTensor<float>(aR, bC);
            if ((long)aR * aC * bC < ParallelThreshold)
            {
                MatMulRawSeq(A.AsReadOnlySpan(), B.AsReadOnlySpan(), aR, aC, bC, C.AsSpan());
            }
            else
            {
                Parallel.For(0, aR, i =>
                {
                    var rC = C.AsSpan().Slice(i * bC, bC);
                    var rA = A.AsReadOnlySpan().Slice(i * aC, aC);
                    var bS = B.AsReadOnlySpan();
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
        private static void MatMulRawSeq(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, int aR, int aC, int bC, Span<float> cS)
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
                MatMulAdd_A_BT_Raw(output.Grad, b.Data, a.Grad);
            }
            if (b.RequiresGrad)
            {
                MatMulAdd_AT_B_Raw(a.Data, output.Grad, b.Grad);
            }
        }

        public static void MatMulAdd_A_BT_Raw(FastTensor<float> A, FastTensor<float> B, FastTensor<float> C)
        {
            int N = A.GetDim(0), K = A.GetDim(1), M = B.GetDim(0);
            if ((long)N * K * M < ParallelThreshold)
            {
                MatMulAdd_A_BT_Seq(A.AsReadOnlySpan(), B.AsReadOnlySpan(), C.AsSpan(), N, K, M);
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    var rA = A.AsReadOnlySpan().Slice(i * K, K);
                    var rC = C.AsSpan().Slice(i * M, M);
                    var bS = B.AsReadOnlySpan();
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

        public static void MatMulAdd_AT_B_Raw(FastTensor<float> A, FastTensor<float> B, FastTensor<float> C)
        {
            int K = A.GetDim(0), N = A.GetDim(1), M = B.GetDim(1);
            if ((long)N * K * M < ParallelThreshold)
            {
                MatMulAdd_AT_B_Seq(A.AsReadOnlySpan(), B.AsReadOnlySpan(), C.AsSpan(), K, N, M);
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    var rC = C.AsSpan().Slice(i * M, M);
                    var aS = A.AsReadOnlySpan();
                    var bS = B.AsReadOnlySpan();
                    for (var k = 0; k < K; k++)
                    {
                        var vA = aS[k * N + i]; if (vA != 0f)
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
                    var vA = aS[k * N + i]; if (vA != 0f)
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
            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.GetDim(0), kSqInC = k * k * inC, colS = kSqInC * outH * outW;
            var workspaceArr = ArrayPool<float>.Shared.Rent(batchSize * colS); var resultData = new FastTensor<float>(batchSize, outC, outH, outW);
            try
            {
                using var w2D = weights.Data.Reshape(outC, kSqInC);
                Parallel.For(0, batchSize, n =>
                {
                    var colS_n = workspaceArr.AsSpan(n * colS, colS);
                    Im2Col(input.Data.AsReadOnlySpan().Slice(n * inC * h * w, inC * h * w), inC, h, w, k, 1, 0, colS_n);
                    MatMulRawSeq(w2D.AsReadOnlySpan(), colS_n, outC, kSqInC, outH * outW, resultData.AsSpan().Slice(n * outC * outH * outW, outC * outH * outW));
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
            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.GetDim(0), kSqInC = k * k * inC, K = outH * outW;
            FastTensor<float> w2DTContig = null;
            if (input.RequiresGrad) { using var w2D = weights.Data.Reshape(outC, kSqInC); using var w2DT = w2D.Transpose(0, 1); w2DTContig = w2DT.ToContiguous(); }
            var weightLock = new object();
            Parallel.For(0, batchSize,
                localInit: () => (ArrayPool<float>.Shared.Rent(kSqInC * K), ArrayPool<float>.Shared.Rent(outC * K), input.RequiresGrad ? ArrayPool<float>.Shared.Rent(kSqInC * K) : null, weights.RequiresGrad ? new FastTensor<float>(true, outC, kSqInC) : null),
                body: (n, loopState, ws) =>
                {
                    var colS = ws.Item1.AsSpan(0, kSqInC * K); var outGS = ws.Item2.AsSpan(0, outC * K);
                    Im2Col(input.Data.AsReadOnlySpan().Slice(n * inC * h * w, inC * h * w), inC, h, w, k, 1, 0, colS);
                    output.Grad.AsReadOnlySpan().Slice(n * outC * K, outC * K).CopyTo(outGS);
                    if (weights.RequiresGrad)
                    {
                        var dwS = ws.Item4.AsSpan();
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
                        var dColS = ws.Item3.AsSpan(0, kSqInC * K); MatMulRawSeq(w2DTContig.AsReadOnlySpan(), outGS, kSqInC, outC, K, dColS);
                        Col2Im(dColS, inC, h, w, k, 1, 0, input.Grad.AsSpan().Slice(n * inC * h * w, inC * h * w));
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
                            TensorPrimitives.Add(weights.Grad.AsSpan(), ws.Item4.AsReadOnlySpan(), weights.Grad.AsSpan());
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
            var res = FastTensor<float>.SameShape(input.Data, false);
            TensorPrimitives.Max(input.Data.AsReadOnlySpan(), 0f, res.AsSpan());
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
            var inS = input.Data.AsReadOnlySpan(); var goS = output.Grad.AsReadOnlySpan(); var giS = input.Grad.AsSpan();
            var i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                var vS = Vector<float>.Count;
                for (; i <= inS.Length - vS; i += vS)
                {
                    var vIn = new Vector<float>(inS.Slice(i)); var vMask = Vector.GreaterThan(vIn, Vector<float>.Zero);
                    (new Vector<float>(giS.Slice(i)) + Vector.ConditionalSelect(vMask, new Vector<float>(goS.Slice(i)), Vector<float>.Zero)).CopyTo(giS.Slice(i));
                }
            }
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
            var res = FastTensor<float>.SameShape(input.Data, false);
            TensorPrimitives.Sigmoid(input.Data.AsReadOnlySpan(), res.AsSpan());
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
            var outS = output.Data.AsReadOnlySpan(); var ogS = output.Grad.AsReadOnlySpan(); var igS = input.Grad.AsSpan();
            Span<float> buf = stackalloc float[StackAllocThreshold];
            for (var i = 0; i < igS.Length; i += StackAllocThreshold)
            {
                var c = Math.Min(StackAllocThreshold, igS.Length - i); var b = buf.Slice(0, c); var o = outS.Slice(i, c);
                TensorPrimitives.Subtract(1f, o, b); TensorPrimitives.Multiply(o, b, b);
                TensorPrimitives.MultiplyAdd(ogS.Slice(i, c), b, igS.Slice(i, c), igS.Slice(i, c));
            }
        }

        public static AutogradNode Tanh(ComputationGraph graph, AutogradNode input)
        {
            var res = FastTensor<float>.SameShape(input.Data, false);
            TensorPrimitives.Tanh(input.Data.AsReadOnlySpan(), res.AsSpan());
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
            var outS = output.Data.AsReadOnlySpan(); var ogS = output.Grad.AsReadOnlySpan(); var igS = input.Grad.AsSpan();
            Span<float> buf = stackalloc float[StackAllocThreshold];
            for (var i = 0; i < igS.Length; i += StackAllocThreshold)
            {
                var c = Math.Min(StackAllocThreshold, igS.Length - i); var b = buf.Slice(0, c); var o = outS.Slice(i, c);
                TensorPrimitives.Multiply(o, o, b); TensorPrimitives.Subtract(1f, b, b);
                TensorPrimitives.MultiplyAdd(ogS.Slice(i, c), b, igS.Slice(i, c), igS.Slice(i, c));
            }
        }

        public static AutogradNode MaxPool2D(ComputationGraph graph, AutogradNode input, int channels, int h, int w, int pool)
        {
            int oH = h / pool, oW = w / pool, batch = input.Data.GetDim(0);
            var res = new FastTensor<float>(batch, channels, oH, oW); var idx = new AutogradNode(new FastTensor<float>(batch, channels, oH, oW), false);
            Parallel.For(0, batch, n =>
            {
                ref var iR = ref MemoryMarshal.GetReference(input.Data.AsReadOnlySpan()); ref var oR = ref MemoryMarshal.GetReference(res.AsSpan()); ref var xR = ref MemoryMarshal.GetReference(idx.Data.AsSpan());
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

                            var oI = n * channels * oH * oW + c * oH * oW + oh * oW + ow; Unsafe.Add(ref oR, oI) = max; Unsafe.Add(ref xR, oI) = mI;
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
            ref var iG = ref MemoryMarshal.GetReference(input.Grad.AsSpan()); ref var xG = ref MemoryMarshal.GetReference(maxIndices.Data.AsReadOnlySpan());
            ref var oG = ref MemoryMarshal.GetReference(output.Grad.AsReadOnlySpan());
            for (var i = 0; i < maxIndices.Data.Size; i++)
            {
                Unsafe.Add(ref iG, (int)Unsafe.Add(ref xG, i)) += Unsafe.Add(ref oG, i);
            }
        }

        public static AutogradNode GlobalAveragePool2D(ComputationGraph graph, AutogradNode input, int channels, int h, int w)
        {
            var res = new FastTensor<float>(input.Data.GetDim(0), channels); float sz = h * w;
            Parallel.For(0, input.Data.GetDim(0), n =>
            {
                for (var c = 0; c < channels; c++)
                {
                    res[n, c] = TensorPrimitives.Sum(input.Data.AsReadOnlySpan().Slice(n * channels * h * w + c * h * w, h * w)) / sz;
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
            Parallel.For(0, input.Data.GetDim(0), n =>
            {
                for (var c = 0; c < channels; c++)
                {
                    var s = input.Grad.AsSpan().Slice(n * channels * h * w + c * h * w, h * w);
                    TensorPrimitives.Add(s, output.Grad.AsReadOnlySpan()[n * channels + c] / sz, s);
                }
            });
        }

        public static AutogradNode Dropout(ComputationGraph graph, AutogradNode input, float probability, bool isTraining)
        {
            var resD = FastTensor<float>.SameShape(input.Data); var mask = new AutogradNode(FastTensor<float>.SameShape(input.Data), false);
            if (isTraining)
            {
                var sc = 1f / (1f - probability); var sz = input.Data.Size; var thr = (byte)(probability * 255f); byte[] rA = null;
                try
                {
                    var rndB = sz <= 2048 ? stackalloc byte[sz] : (rA = ArrayPool<byte>.Shared.Rent(sz)).AsSpan(0, sz);
                    Random.Shared.NextBytes(rndB); var mS = mask.Data.AsSpan();
                    for (var i = 0; i < sz; i++)
                    {
                        mS[i] = rndB[i] > thr ? sc : 0f;
                    }
                    TensorPrimitives.Multiply(input.Data.AsReadOnlySpan(), mS, resD.AsSpan());
                }
                finally
                {
                    if (rA != null)
                    {
                        ArrayPool<byte>.Shared.Return(rA);
                    }
                }
            }
            else
            {
                input.Data.AsReadOnlySpan().CopyTo(resD.AsSpan());
            }
            var output = new AutogradNode(resD, input.RequiresGrad);
            if (output.RequiresGrad && isTraining)
            {
                graph?.Record(OpCode.Dropout, output, input, mask);
            }
            else
            {
                mask.Dispose();
            }
            return output;
        }

        public static void DropoutBackward(AutogradNode input, AutogradNode mask, AutogradNode output) => TensorPrimitives.MultiplyAdd(output.Grad.AsReadOnlySpan(), mask.Data.AsReadOnlySpan(), input.Grad.AsSpan(), input.Grad.AsSpan());

        // ====================================================================
        // 4. LOSS FUNCTIONS & BATCHNORM
        // ====================================================================

        public static AutogradNode SoftmaxCrossEntropy(ComputationGraph graph, AutogradNode logits, AutogradNode target)
        {
            int rows = logits.Data.GetDim(0), cols = logits.Data.GetDim(1); var total = 0f; var probs = new FastTensor<float>(rows, cols);
            for (var r = 0; r < rows; r++)
            {
                var pR = probs.AsSpan().Slice(r * cols, cols); TensorPrimitives.SoftMax(logits.Data.AsReadOnlySpan().Slice(r * cols, cols), pR);
                for (var c = 0; c < cols; c++)
                {
                    if (target.Data[r, c] > 0.5f)
                    {
                        total -= MathF.Log(pR[c] + 1e-15f);
                    }
                }
            }
            var output = new AutogradNode(new FastTensor<float>(1, 1) { [0, 0] = total / rows }, logits.RequiresGrad);
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
            int R = logits.Data.GetDim(0), C = logits.Data.GetDim(1); var s = output.Grad.AsReadOnlySpan()[0] / R;
            for (var r = 0; r < R; r++)
            {
                var pS = probsNode.Data.AsReadOnlySpan().Slice(r * C, C); var tS = target.Data.AsReadOnlySpan().Slice(r * C, C); var gS = logits.Grad.AsSpan().Slice(r * C, C);
                TensorPrimitives.MultiplyAdd(pS, s, gS, gS); TensorPrimitives.MultiplyAdd(tS, -s, gS, gS);
            }
        }

        public static AutogradNode MSELoss(ComputationGraph graph, AutogradNode prediction, AutogradNode target)
        {
            var sz = prediction.Data.Size; var diffA = ArrayPool<float>.Shared.Rent(sz); float mse;
            try { var dS = diffA.AsSpan(0, sz); TensorPrimitives.Subtract(prediction.Data.AsReadOnlySpan(), target.Data.AsReadOnlySpan(), dS); mse = TensorPrimitives.Dot(dS, dS) / sz; }
            finally { ArrayPool<float>.Shared.Return(diffA); }
            var output = new AutogradNode(new FastTensor<float>(1, 1) { [0, 0] = mse }, prediction.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MseLoss, output, prediction, target);
            }
            return output;
        }

        public static void MSELossBackward(AutogradNode p, AutogradNode t, AutogradNode o)
        {
            var f = o.Grad.AsReadOnlySpan()[0] * (2f / p.Data.Size);
            TensorPrimitives.MultiplyAdd(p.Data.AsReadOnlySpan(), f, p.Grad.AsSpan(), p.Grad.AsSpan());
            TensorPrimitives.MultiplyAdd(t.Data.AsReadOnlySpan(), -f, p.Grad.AsSpan(), p.Grad.AsSpan());
        }

        public static AutogradNode DirectionalLoss(ComputationGraph graph, AutogradNode prediction, AutogradNode target, float gamma = 10f)
        {
            var sz = prediction.Data.Size; float loss; var tempA = ArrayPool<float>.Shared.Rent(sz);
            try
            {
                var s = tempA.AsSpan(0, sz); TensorPrimitives.Subtract(prediction.Data.AsReadOnlySpan(), target.Data.AsReadOnlySpan(), s); var mse = TensorPrimitives.SumOfSquares(s);
                TensorPrimitives.Multiply(prediction.Data.AsReadOnlySpan(), target.Data.AsReadOnlySpan(), s); TensorPrimitives.Min(s, 0f, s);
                loss = (mse + TensorPrimitives.Sum(s) * -gamma) / sz;
            }
            finally { ArrayPool<float>.Shared.Return(tempA); }
            var output = new AutogradNode(new FastTensor<float>(1, 1) { [0, 0] = loss }, prediction.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.DirectionalLoss, output, prediction, target, BitConverter.SingleToInt32Bits(gamma));
            }
            return output;
        }

        public static void DirectionalLossBackward(AutogradNode p, AutogradNode t, AutogradNode o, float gamma)
        {
            var s = o.Grad.AsReadOnlySpan()[0] / p.Data.Size; var pG = p.Grad.AsSpan(); var pD = p.Data.AsReadOnlySpan(); var tD = t.Data.AsReadOnlySpan(); var i = 0;
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
            int N = input.Data.GetDim(0), C = input.Data.GetDim(1); var outD = new FastTensor<float>(N, C);
            var mean = new AutogradNode(new FastTensor<float>(C), false); var invStd = new AutogradNode(new FastTensor<float>(C), false);
            if (isTraining)
            {
                var mS = mean.Data.AsSpan(); for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Add(mS, input.Data.AsReadOnlySpan().Slice(i * C, C), mS);
                }
                TensorPrimitives.Multiply(mS, 1f / N, mS);
                using var vB = new FastTensor<float>(true, C); using var tB = new FastTensor<float>(false, C);
                for (var i = 0; i < N; i++) { TensorPrimitives.Subtract(input.Data.AsReadOnlySpan().Slice(i * C, C), mS, tB.AsSpan()); TensorPrimitives.MultiplyAdd(tB.AsReadOnlySpan(), tB.AsReadOnlySpan(), vB.AsSpan(), vB.AsSpan()); }
                TensorPrimitives.Multiply(vB.AsReadOnlySpan(), 1f / N, vB.AsSpan());
                var rmS = runningMean.AsSpan(); var rvS = runningVar.AsSpan(); var ivS = invStd.Data.AsSpan();
                TensorPrimitives.Multiply(rmS, 1f - momentum, rmS); TensorPrimitives.MultiplyAdd(mS, momentum, rmS, rmS);
                TensorPrimitives.Multiply(rvS, 1f - momentum, rvS); TensorPrimitives.MultiplyAdd(vB.AsReadOnlySpan(), momentum, rvS, rvS);
                TensorPrimitives.Add(vB.AsReadOnlySpan(), eps, ivS); TensorPrimitives.ReciprocalSqrt(ivS, ivS);
            }
            else
            {
                runningMean.AsReadOnlySpan().CopyTo(mean.Data.AsSpan()); var ivS = invStd.Data.AsSpan();
                TensorPrimitives.Add(runningVar.AsReadOnlySpan(), eps, ivS); TensorPrimitives.ReciprocalSqrt(ivS, ivS);
            }
            for (var i = 0; i < N; i++)
            {
                var oR = outD.AsSpan().Slice(i * C, C); TensorPrimitives.Subtract(input.Data.AsReadOnlySpan().Slice(i * C, C), mean.Data.AsReadOnlySpan(), oR);
                TensorPrimitives.Multiply(oR, invStd.Data.AsReadOnlySpan(), oR); TensorPrimitives.MultiplyAdd(oR, gamma.Data.AsReadOnlySpan(), beta.Data.AsReadOnlySpan(), oR);
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
            int N = input.Data.GetDim(0), C = input.Data.GetDim(1);
            float[] cA = null, tA = null, sDA = null, sDXA = null, xHA = null;
            try
            {
                var coeff = C <= StackAllocThreshold ? stackalloc float[C] : (cA = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                var term = C <= StackAllocThreshold ? stackalloc float[C] : (tA = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                var sDy = C <= StackAllocThreshold ? stackalloc float[C] : (sDA = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                var sDyX = C <= StackAllocThreshold ? stackalloc float[C] : (sDXA = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                var xHR = C <= StackAllocThreshold ? stackalloc float[C] : (xHA = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                sDy.Clear(); sDyX.Clear(); TensorPrimitives.Multiply(gamma.Data.AsReadOnlySpan(), invStd.Data.AsReadOnlySpan(), coeff); TensorPrimitives.Multiply(coeff, 1f / N, coeff);
                for (var i = 0; i < N; i++)
                {
                    var gR = output.Grad.AsReadOnlySpan().Slice(i * C, C); var iR = input.Data.AsReadOnlySpan().Slice(i * C, C);
                    TensorPrimitives.Subtract(iR, mean.Data.AsReadOnlySpan(), xHR); TensorPrimitives.Multiply(xHR, invStd.Data.AsReadOnlySpan(), xHR);
                    TensorPrimitives.Add(sDy, gR, sDy); TensorPrimitives.MultiplyAdd(gR, xHR, sDyX, sDyX);
                    if (beta.RequiresGrad)
                    {
                        TensorPrimitives.Add(beta.Grad.AsSpan(), gR, beta.Grad.AsSpan());
                    }
                    if (gamma.RequiresGrad)
                    {
                        TensorPrimitives.MultiplyAdd(gR, xHR, gamma.Grad.AsSpan(), gamma.Grad.AsSpan());
                    }
                }
                if (input.RequiresGrad)
                {
                    var iGS = input.Grad.AsSpan(); for (var i = 0; i < N; i++)
                    {
                        var gR = output.Grad.AsReadOnlySpan().Slice(i * C, C); var iGR = iGS.Slice(i * C, C); var iR = input.Data.AsReadOnlySpan().Slice(i * C, C);
                        TensorPrimitives.Subtract(iR, mean.Data.AsReadOnlySpan(), xHR); TensorPrimitives.Multiply(xHR, invStd.Data.AsReadOnlySpan(), xHR);
                        TensorPrimitives.Multiply(gR, N, term); TensorPrimitives.Subtract(term, sDy, term);
                        Span<float> tempXHat = stackalloc float[C]; xHR.CopyTo(tempXHat);
                        TensorPrimitives.Multiply(tempXHat, sDyX, tempXHat); TensorPrimitives.Subtract(term, tempXHat, term);
                        TensorPrimitives.MultiplyAdd(coeff, term, iGR, iGR);
                    }
                }
            }
            finally
            {
                if (cA != null)
                {
                    ArrayPool<float>.Shared.Return(cA);
                }
                if (tA != null)
                {
                    ArrayPool<float>.Shared.Return(tA);
                }
                if (sDA != null)
                {
                    ArrayPool<float>.Shared.Return(sDA);
                }
                if (sDXA != null)
                {
                    ArrayPool<float>.Shared.Return(sDXA);
                }
                if (xHA != null)
                {
                    ArrayPool<float>.Shared.Return(xHA);
                }
            }
        }

        // ====================================================================
        // 5. BPTT & SEQUENCE (Fused LSTM, Repeat, Multiply, Slices)
        // ====================================================================

        public static (AutogradNode hNew, AutogradNode cNew) FusedLSTMStep(ComputationGraph graph, AutogradNode x, AutogradNode hPrev, AutogradNode cPrev, AutogradNode W, AutogradNode U, AutogradNode B)
        {
            int batchSize = x.Data.GetDim(0), hS = hPrev.Data.GetDim(1); var gD = MatMulRaw(x.Data, W.Data); using var uh = MatMulRaw(hPrev.Data, U.Data);
            var cnD = new FastTensor<float>(batchSize, hS); var hnD = new FastTensor<float>(batchSize, hS);
            Parallel.For(0, batchSize, b =>
            {
                var bg = gD.AsSpan().Slice(b * 4 * hS, 4 * hS); TensorPrimitives.Add(bg, uh.AsReadOnlySpan().Slice(b * 4 * hS, 4 * hS), bg); TensorPrimitives.Add(bg, B.Data.AsReadOnlySpan(), bg);
                var f = bg.Slice(0, hS); var i = bg.Slice(hS, hS); var g = bg.Slice(2 * hS, hS); var o = bg.Slice(3 * hS, hS);
                TensorPrimitives.Sigmoid(f, f); TensorPrimitives.Sigmoid(i, i); TensorPrimitives.Tanh(g, g); TensorPrimitives.Sigmoid(o, o);
                var bcn = cnD.AsSpan().Slice(b * hS, hS); TensorPrimitives.Multiply(f, cPrev.Data.AsReadOnlySpan().Slice(b * hS, hS), bcn); TensorPrimitives.MultiplyAdd(i, g, bcn, bcn);
                var bhn = hnD.AsSpan().Slice(b * hS, hS); TensorPrimitives.Tanh(bcn, bhn); TensorPrimitives.Multiply(o, bhn, bhn);
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
            int batchSize = x.Data.GetDim(0), hS = hPrev.Data.GetDim(1); using var dG = new FastTensor<float>(batchSize, 4 * hS);
            Parallel.For(0, batchSize, localInit: () => ArrayPool<float>.Shared.Rent(hS * 4),
                body: (b, state, arr) =>
                {
                    var dGS = dG.AsSpan().Slice(b * 4 * hS, 4 * hS);
                    var gs = gates.Data.AsReadOnlySpan().Slice(b * 4 * hS, 4 * hS); var f = gs.Slice(0, hS); var i = gs.Slice(hS, hS); var g = gs.Slice(2 * hS, hS); var o = gs.Slice(3 * hS, hS);
                    var dh = hNew.Grad.AsReadOnlySpan().Slice(b * hS, hS); var dc = cNew.Grad.AsSpan().Slice(b * hS, hS); var tS = arr.AsSpan(); var tanhC = tS.Slice(0, hS); var t1 = tS.Slice(hS, hS); var t2 = tS.Slice(2 * hS, hS);
                    TensorPrimitives.Tanh(cNew.Data.AsReadOnlySpan().Slice(b * hS, hS), tanhC);
                    TensorPrimitives.Subtract(1f, o, t1); TensorPrimitives.Multiply(o, t1, t1); TensorPrimitives.Multiply(dh, tanhC, t2); TensorPrimitives.Multiply(t2, t1, dGS.Slice(3 * hS, hS));
                    TensorPrimitives.Multiply(tanhC, tanhC, t1); TensorPrimitives.Subtract(1f, t1, t1); TensorPrimitives.Multiply(dh, o, t2); TensorPrimitives.MultiplyAdd(t2, t1, dc, dc);
                    TensorPrimitives.Multiply(g, g, t1); TensorPrimitives.Subtract(1f, t1, t1); TensorPrimitives.Multiply(dc, i, t2); TensorPrimitives.Multiply(t2, t1, dGS.Slice(2 * hS, hS));
                    TensorPrimitives.Subtract(1f, i, t1); TensorPrimitives.Multiply(i, t1, t1); TensorPrimitives.Multiply(dc, g, t2); TensorPrimitives.Multiply(t2, t1, dGS.Slice(hS, hS));
                    TensorPrimitives.Subtract(1f, f, t1); TensorPrimitives.Multiply(f, t1, t1); TensorPrimitives.Multiply(dc, cPrev.Data.AsReadOnlySpan().Slice(b * hS, hS), t2); TensorPrimitives.Multiply(t2, t1, dGS.Slice(0, hS));
                    if (cPrev.RequiresGrad) { var dcp = cPrev.Grad.AsSpan().Slice(b * hS, hS); TensorPrimitives.MultiplyAdd(dc, f, dcp, dcp); }
                    return arr;
                }, localFinally: arr => ArrayPool<float>.Shared.Return(arr));
            if (x.RequiresGrad)
            {
                MatMulAdd_A_BT_Raw(dG, W.Data, x.Grad);
            }
            if (W.RequiresGrad)
            {
                MatMulAdd_AT_B_Raw(x.Data, dG, W.Grad);
            }
            if (hPrev.RequiresGrad)
            {
                MatMulAdd_A_BT_Raw(dG, U.Data, hPrev.Grad);
            }
            if (U.RequiresGrad)
            {
                MatMulAdd_AT_B_Raw(hPrev.Data, dG, U.Grad);
            }
            if (B.RequiresGrad)
            {
                var dbS = B.Grad.AsSpan(); var dGS_full = dG.AsReadOnlySpan(); for (var b = 0; b < batchSize; b++)
                {
                    TensorPrimitives.Add(dbS, dGS_full.Slice(b * 4 * hS, 4 * hS), dbS);
                }
            }
        }

        public static AutogradNode RepeatVector(ComputationGraph graph, AutogradNode input, int seqLen)
        {
            int batch = input.Data.GetDim(0), hS = input.Data.GetDim(1); var res = new FastTensor<float>(false, batch, seqLen, hS);
            for (var b = 0; b < batch; b++)
            {
                var src = input.Data.AsReadOnlySpan().Slice(b * hS, hS); for (var t = 0; t < seqLen; t++)
                {
                    src.CopyTo(res.AsSpan().Slice(b * seqLen * hS + t * hS, hS));
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
            var iGS = input.Grad.AsSpan(); var ogS = output.Grad.AsReadOnlySpan();
            for (var b = 0; b < input.Data.GetDim(0); b++)
            {
                var dst = iGS.Slice(b * hS, hS); for (var t = 0; t < seqLen; t++)
                {
                    TensorPrimitives.Add(dst, ogS.Slice(b * seqLen * hS + t * hS, hS), dst);
                }
            }
        }

        public static AutogradNode Multiply(ComputationGraph graph, AutogradNode a, AutogradNode b)
        {
            var res = FastTensor<float>.SameShape(a.Data, false); TensorPrimitives.Multiply(a.Data.AsReadOnlySpan(), b.Data.AsReadOnlySpan(), res.AsSpan());
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
                TensorPrimitives.MultiplyAdd(output.Grad.AsReadOnlySpan(), b.Data.AsReadOnlySpan(), a.Grad.AsSpan(), a.Grad.AsSpan());
            }
            if (b.RequiresGrad)
            {
                TensorPrimitives.MultiplyAdd(output.Grad.AsReadOnlySpan(), a.Data.AsReadOnlySpan(), b.Grad.AsSpan(), b.Grad.AsSpan());
            }
        }

        public static AutogradNode GateSlice(ComputationGraph graph, AutogradNode gates, int hiddenSize, int gateIndex)
        {
            var res = new FastTensor<float>(false, gates.Data.GetDim(0), hiddenSize);
            int batch = gates.Data.GetDim(0), stride = 4 * hiddenSize, offset = gateIndex * hiddenSize;
            for (var b = 0; b < batch; b++)
            {
                gates.Data.AsReadOnlySpan().Slice(b * stride + offset, hiddenSize).CopyTo(res.AsSpan().Slice(b * hiddenSize, hiddenSize));
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
            int batch = gates.Data.GetDim(0), offset = gateIndex * hiddenSize, stride = 4 * hiddenSize;
            for (var b = 0; b < batch; b++) { var dst = gates.Grad.AsSpan().Slice(b * stride + offset, hiddenSize); TensorPrimitives.Add(dst, output.Grad.AsReadOnlySpan().Slice(b * hiddenSize, hiddenSize), dst); }
        }

        public static void TimestepSliceBackward(AutogradNode input, AutogradNode output, int t, int seqLen, int inputSize)
        {
            if (!input.RequiresGrad)
            {
                return;
            }
            for (var b = 0; b < input.Data.GetDim(0); b++) { var dst = input.Grad.AsSpan().Slice(b * seqLen * inputSize + t * inputSize, inputSize); TensorPrimitives.Add(dst, output.Grad.AsReadOnlySpan().Slice(b * inputSize, inputSize), dst); }
        }

        public static void StackTimestepsBackward(AutogradNode[] allH, AutogradNode output, int batch, int seqLen, int hiddenSize)
        {
            for (var t = 0; t < seqLen; t++)
            {
                if (!allH[t].RequiresGrad)
                {
                    continue;
                }
                for (var b = 0; b < batch; b++) { var dst = allH[t].Grad.AsSpan().Slice(b * hiddenSize, hiddenSize); TensorPrimitives.Add(dst, output.Grad.AsReadOnlySpan().Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize), dst); }
            }
        }

        // ====================================================================
        // 6. GRAPH HELPERS
        // ====================================================================

        public static AutogradNode Reshape(ComputationGraph graph, AutogradNode input, params int[] newShape)
        {
            var output = new AutogradNode(input.Data.Reshape(newShape), input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Reshape, output, input);
            }
            return output;
        }

        public static void ReshapeBackward(AutogradNode input, AutogradNode output) => TensorPrimitives.Add(input.Grad.AsSpan(), output.Grad.AsReadOnlySpan(), input.Grad.AsSpan());

        public static AutogradNode Linear(ComputationGraph graph, AutogradNode input, AutogradNode weights, AutogradNode bias) => AddBias(graph, MatMul(graph, input, weights), bias);

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
                            var i = y * s - p + kh; if (i >= 0 && i < h)
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
                            var i = y * s - p + kh; if (i >= 0 && i < h)
                            {
                                var iO = c * h * w + i * w;
                                for (var x = 0; x < oW; x++)
                                {
                                    var j = x * s - p + kw; if (j >= 0 && j < w)
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