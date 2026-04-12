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
    /// <summary>
    ///     Static utility class providing high-performance tensor operations with Autograd support.
    ///     Utilizes SIMD (TensorPrimitives), Task Parallel Library (TPL), and L1/L2 Cache Tiling.
    /// </summary>
    public static class TensorMath
    {
        // Workload threshold (in FMA operations) below which Parallel.For is inefficient.
        private const long ParallelThreshold = 4096;

        // Rozmiar bufora alokowanego na stosie (4 KB - idealnie mieści się w L1).
        private const int StackAllocThreshold = 1024;

        // ====================================================================
        // CACHE TILING PARAMETERS (Złoty podział dla L1/L2 Cache)
        // ====================================================================
        // Pętle wewnątrz MatMul dzielimy na mniejsze klocki, aby nie przepełnić cache'u 
        // procesora przed zakończeniem operacji na wektorach SIMD.
        private const int BlockK = 128;
        private const int BlockJ = 512; // 512 floatów = 2KB na wiersz, idealne dla wektoryzacji.

        // ====================================================================
        // 1. BASIC LINEAR ALGEBRA (With Blocked Matrix Multiplication)
        // ====================================================================

        public static AutogradNode Add(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var resultData = FastTensor<float>.SameShape(left.Data);
            TensorPrimitives.Add(left.Data.AsSpan(), right.Data.AsSpan(), resultData.AsSpan());
            var outputNode = new AutogradNode(resultData, left.RequiresGrad || right.RequiresGrad);

            if (outputNode.RequiresGrad)
            {
                graph?.Record(OpCode.Add, outputNode, left, right);
            }

            return outputNode;
        }

        public static void AddBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                TensorPrimitives.Add(a.Grad.AsSpan(), output.Grad.AsSpan(), a.Grad.AsSpan());
            }

            if (b.RequiresGrad)
            {
                TensorPrimitives.Add(b.Grad.AsSpan(), output.Grad.AsSpan(), b.Grad.AsSpan());
            }
        }

        public static AutogradNode AddBias(ComputationGraph graph, AutogradNode input, AutogradNode bias)
        {
            var N = input.Data.GetDim(0);
            var C = input.Data.GetDim(1);
            var resultData = new FastTensor<float>(N, C);
            var inS = input.Data.AsSpan();
            var bS = bias.Data.AsSpan();
            var resS = resultData.AsSpan();

            for (var i = 0; i < N; i++)
            {
                TensorPrimitives.Add(inS.Slice(i * C, C), bS, resS.Slice(i * C, C));
            }

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || bias.RequiresGrad);

            if (outputNode.RequiresGrad)
            {
                graph?.Record(OpCode.AddBias, outputNode, input, bias);
            }

            return outputNode;
        }

        public static void AddBiasBackward(AutogradNode input, AutogradNode bias, AutogradNode output)
        {
            var N = input.Data.GetDim(0);
            var C = input.Data.GetDim(1);

            if (input.RequiresGrad)
            {
                TensorPrimitives.Add(input.Grad.AsSpan(), output.Grad.AsSpan(), input.Grad.AsSpan());
            }

            if (bias.RequiresGrad)
            {
                var bGS = bias.Grad.AsSpan();
                var oGS = output.Grad.AsSpan();

                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Add(bGS, oGS.Slice(i * C, C), bGS);
                }
            }
        }

        public static AutogradNode MatMul(ComputationGraph graph, AutogradNode left, AutogradNode right)
        {
            var resultData = MatMulRaw(left.Data, right.Data);
            var outputNode = new AutogradNode(resultData, left.RequiresGrad || right.RequiresGrad);

            if (outputNode.RequiresGrad)
            {
                graph?.Record(OpCode.MatMul, outputNode, left, right);
            }

            return outputNode;
        }

        public static FastTensor<float> MatMulRaw(FastTensor<float> A, FastTensor<float> B)
        {
            int aRows = A.GetDim(0), aCols = A.GetDim(1), bCols = B.GetDim(1);
            var C = new FastTensor<float>(aRows, bCols);

            var totalWork = (long)aRows * aCols * bCols;

            if (totalWork < ParallelThreshold)
            {
                MatMulRawSequential(A.AsSpan(), B.AsSpan(), aRows, aCols, bCols, C.AsSpan());
            }
            else
            {
                Parallel.For(0, aRows, i =>
                {
                    var aS = A.AsSpan();
                    var bS = B.AsSpan();
                    var rowA = aS.Slice(i * aCols, aCols);
                    var rowCFull = C.AsSpan().Slice(i * bCols, bCols);

                    // Cache Tiling: Blokowanie pętli K i J dla spójności danych w L2 Cache
                    for (var j0 = 0; j0 < bCols; j0 += BlockJ)
                    {
                        var jLen = Math.Min(bCols - j0, BlockJ);
                        var rowC = rowCFull.Slice(j0, jLen);

                        for (var k0 = 0; k0 < aCols; k0 += BlockK)
                        {
                            var kMax = Math.Min(k0 + BlockK, aCols);
                            for (var k = k0; k < kMax; k++)
                            {
                                var valA = rowA[k];
                                
                                if (valA != 0f)
                                {
                                    var rowB = bS.Slice(k * bCols + j0, jLen);
                                    TensorPrimitives.MultiplyAdd(rowB, valA, rowC, rowC);
                                }
                            }
                        }
                    }
                });
            }

            return C;
        }

        private static void MatMulRawSequential(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, int aR, int aC, int bC, Span<float> cS)
        {
            cS.Clear();

            // Cache Tiling for Single-Thread
            for (var j0 = 0; j0 < bC; j0 += BlockJ)
            {
                var jLen = Math.Min(bC - j0, BlockJ);

                for (var k0 = 0; k0 < aC; k0 += BlockK)
                {
                    var kMax = Math.Min(k0 + BlockK, aC);

                    for (var i = 0; i < aR; i++)
                    {
                        var rowA = aS.Slice(i * aC, aC);
                        var rowC = cS.Slice(i * bC + j0, jLen);

                        for (var k = k0; k < kMax; k++)
                        {
                            var valA = rowA[k];
                            
                            if (valA != 0f)
                            {
                                var rowB = bS.Slice(k * bC + j0, jLen);
                                TensorPrimitives.MultiplyAdd(rowB, valA, rowC, rowC);
                            }
                        }
                    }
                }
            }
        }

        public static void MatMulBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                MatMulAdd_A_BT(output.Grad, b.Data, a.Grad);
            }

            if (b.RequiresGrad)
            {
                MatMulAdd_AT_B(a.Data, output.Grad, b.Grad);
            }
        }

        // C += A * B^T
        private static void MatMulAdd_A_BT(FastTensor<float> A, FastTensor<float> B, FastTensor<float> CGrad)
        {
            int N = A.GetDim(0), K = A.GetDim(1), M = B.GetDim(0);
            var totalWork = (long)N * K * M;

            if (totalWork < ParallelThreshold)
            {
                var aS = A.AsSpan();
                var bS = B.AsSpan();
                var cS = CGrad.AsSpan();

                for (var i = 0; i < N; i++)
                {
                    var aRowFull = aS.Slice(i * K, K);
                    var cRow = cS.Slice(i * M, M);

                    // Cache Tiling: Partial Dot Products
                    for (var j0 = 0; j0 < M; j0 += BlockJ)
                    {
                        var jMax = Math.Min(j0 + BlockJ, M);
                        for (var k0 = 0; k0 < K; k0 += BlockK)
                        {
                            var kLen = Math.Min(K - k0, BlockK);
                            var aRowBlock = aRowFull.Slice(k0, kLen);

                            for (var j = j0; j < jMax; j++)
                            {
                                cRow[j] += TensorPrimitives.Dot(aRowBlock, bS.Slice(j * K + k0, kLen));
                            }
                        }
                    }
                }
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    var aRowFull = A.AsSpan().Slice(i * K, K);
                    var cRow = CGrad.AsSpan().Slice(i * M, M);
                    var bS = B.AsSpan();

                    // Cache Tiling for Backward Pass (Dot Product Accumulation)
                    for (var j0 = 0; j0 < M; j0 += BlockJ)
                    {
                        var jMax = Math.Min(j0 + BlockJ, M);
                        for (var k0 = 0; k0 < K; k0 += BlockK)
                        {
                            var kLen = Math.Min(K - k0, BlockK);
                            var aRowBlock = aRowFull.Slice(k0, kLen);

                            for (var j = j0; j < jMax; j++)
                            {
                                cRow[j] += TensorPrimitives.Dot(aRowBlock, bS.Slice(j * K + k0, kLen));
                            }
                        }
                    }
                });
            }
        }

        // C += A^T * B
        private static void MatMulAdd_AT_B(FastTensor<float> A, FastTensor<float> B, FastTensor<float> CGrad)
        {
            var K = A.GetDim(0);
            var N = A.GetDim(1);
            var M = B.GetDim(1);
            var totalWork = (long)N * K * M;

            if (totalWork < ParallelThreshold)
            {
                var cS = CGrad.AsSpan();
                var aS = A.AsSpan();
                var bS = B.AsSpan();

                for (var i = 0; i < N; i++)
                {
                    var cRowFull = cS.Slice(i * M, M);

                    for (var j0 = 0; j0 < M; j0 += BlockJ)
                    {
                        var jLen = Math.Min(M - j0, BlockJ);
                        var cRowBlock = cRowFull.Slice(j0, jLen);

                        for (var k0 = 0; k0 < K; k0 += BlockK)
                        {
                            var kMax = Math.Min(k0 + BlockK, K);
                            for (var k = k0; k < kMax; k++)
                            {
                                var aVal = aS[k * N + i];
                                
                                if (aVal != 0f)
                                {
                                    var bRowBlock = bS.Slice(k * M + j0, jLen);
                                    TensorPrimitives.MultiplyAdd(bRowBlock, aVal, cRowBlock, cRowBlock);
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                Parallel.For(0, N, i =>
                {
                    var cRowFull = CGrad.AsSpan().Slice(i * M, M);
                    var aS = A.AsSpan();
                    var bS = B.AsSpan();

                    // Cache Tiling
                    for (var j0 = 0; j0 < M; j0 += BlockJ)
                    {
                        var jLen = Math.Min(M - j0, BlockJ);
                        var cRowBlock = cRowFull.Slice(j0, jLen);

                        for (var k0 = 0; k0 < K; k0 += BlockK)
                        {
                            var kMax = Math.Min(k0 + BlockK, K);
                            for (var k = k0; k < kMax; k++)
                            {
                                var aVal = aS[k * N + i];
                                
                                if (aVal != 0f)
                                {
                                    var bRowBlock = bS.Slice(k * M + j0, jLen);
                                    TensorPrimitives.MultiplyAdd(bRowBlock, aVal, cRowBlock, cRowBlock);
                                }
                            }
                        }
                    }
                });
            }
        }

        // ====================================================================
        // 2. CNN OPERATIONS (NCHW)
        // ====================================================================

        public static AutogradNode Conv2D(ComputationGraph graph, AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.GetDim(0), kSqInC = k * k * inC;
            int colSizePerImg = kSqInC * outH * outW, inSize = inC * h * w, outSize = outC * outH * outW;

            var workspaceArr = ArrayPool<float>.Shared.Rent(batchSize * colSizePerImg);
            var resultData = new FastTensor<float>(batchSize, outC, outH, outW);

            try
            {
                using var weights2D = weights.Data.Reshape(outC, kSqInC);

                if (batchSize == 1)
                {
                    var colSpan = workspaceArr.AsSpan(0, colSizePerImg);
                    Im2Col(input.Data.AsSpan().Slice(0, inSize), inC, h, w, k, 1, 0, colSpan);
                    MatMulRawSequential(weights2D.AsSpan(), colSpan, outC, kSqInC, outH * outW, resultData.AsSpan().Slice(0, outSize));
                }
                else
                {
                    Parallel.For(0, batchSize, n =>
                    {
                        var colSpan = workspaceArr.AsSpan(n * colSizePerImg, colSizePerImg);
                        Im2Col(input.Data.AsSpan().Slice(n * inSize, inSize), inC, h, w, k, 1, 0, colSpan);
                        MatMulRawSequential(weights2D.AsSpan(), colSpan, outC, kSqInC, outH * outW, resultData.AsSpan().Slice(n * outSize, outSize));
                    });
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(workspaceArr);
            }

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

            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.GetDim(0), kSqInC = k * k * inC;
            int inSize = inC * h * w, outSize = outC * outH * outW;
            var K = outH * outW;

            FastTensor<float> weights2DTContig = null;

            if (input.RequiresGrad)
            {
                using var weights2D = weights.Data.Reshape(outC, kSqInC);
                using var weights2DT = weights2D.Transpose(0, 1);

                weights2DTContig = weights2DT.ToContiguous();
            }

            var weightLock = new object();

            Parallel.For(0, batchSize,
            localInit: () =>
            {
                var colArr = ArrayPool<float>.Shared.Rent(kSqInC * K);
                var gradMatArr = ArrayPool<float>.Shared.Rent(outC * K);
                var dColArr = input.RequiresGrad ? ArrayPool<float>.Shared.Rent(kSqInC * K) : null;
                var localDw = weights.RequiresGrad ? new FastTensor<float>(true, outC, kSqInC) : null;
                return (colArr, gradMatArr, dColArr, localDw);
            },
            body: (n, loopState, ws) =>
            {
                var (colArr, gradMatArr, dColArr, localDw) = ws;
                var colSpan = colArr.AsSpan(0, kSqInC * K);
                var outGradSpan = gradMatArr.AsSpan(0, outC * K);

                Im2Col(input.Data.AsSpan().Slice(n * inSize, inSize), inC, h, w, k, 1, 0, colSpan);
                output.Grad.AsSpan().Slice(n * outSize, outSize).CopyTo(outGradSpan);

                if (weights.RequiresGrad)
                {
                    var dwSpan = localDw.AsSpan();
                    for (var r = 0; r < outC; r++)
                    {
                        var outGradRow = outGradSpan.Slice(r * K, K);
                        for (var c = 0; c < kSqInC; c++)
                        {
                            dwSpan[r * kSqInC + c] += TensorPrimitives.Dot(outGradRow, colSpan.Slice(c * K, K));
                        }
                    }
                }

                if (input.RequiresGrad)
                {
                    var dColSpan = dColArr.AsSpan(0, kSqInC * K);
                    MatMulRawSequential(weights2DTContig.AsSpan(), outGradSpan, kSqInC, outC, K, dColSpan);
                    Col2Im(dColSpan, inC, h, w, k, 1, 0, input.Grad.AsSpan().Slice(n * inSize, inSize));
                }

                return ws;
            },
            localFinally: ws =>
            {
                var (colArr, gradMatArr, dColArr, localDw) = ws;

                ArrayPool<float>.Shared.Return(colArr);
                ArrayPool<float>.Shared.Return(gradMatArr);

                if (dColArr != null) { ArrayPool<float>.Shared.Return(dColArr); }

                if (localDw != null)
                {
                    lock (weightLock)
                    {
                        TensorPrimitives.Add(weights.Grad.AsSpan(), localDw.AsSpan(), weights.Grad.AsSpan());
                    }
                    localDw.Dispose();
                }
            }
            );

            weights2DTContig?.Dispose();
        }

        public static AutogradNode MaxPool2D(ComputationGraph graph, AutogradNode input, int channels, int inputH, int inputW, int poolSize)
        {
            int outputH = inputH / poolSize, outputW = inputW / poolSize, batchSize = input.Data.GetDim(0);
            var resultData = new FastTensor<float>(batchSize, channels, outputH, outputW);
            var maxIndices = new AutogradNode(new FastTensor<float>(batchSize, channels, outputH, outputW), false);

            Parallel.For(0, batchSize, n =>
            {
                ref var inRef = ref MemoryMarshal.GetReference(input.Data.AsSpan());
                ref var outRef = ref MemoryMarshal.GetReference(resultData.AsSpan());
                ref var idxRef = ref MemoryMarshal.GetReference(maxIndices.Data.AsSpan());

                var bInOffset = n * channels * inputH * inputW;
                var bOutOffset = n * channels * outputH * outputW;

                for (var c = 0; c < channels; c++)
                {
                    var cInOffset = bInOffset + c * inputH * inputW;
                    var cOutOffset = bOutOffset + c * outputH * outputW;

                    for (var oh = 0; oh < outputH; oh++)
                    {
                        var ohInOffset = cInOffset + oh * poolSize * inputW;
                        var ohOutOffset = cOutOffset + oh * outputW;

                        for (var ow = 0; ow < outputW; ow++)
                        {
                            var maxVal = float.MinValue;
                            var maxIdx = -1;
                            var owInOffset = ohInOffset + ow * poolSize;

                            for (var ph = 0; ph < poolSize; ph++)
                            {
                                var phInOffset = owInOffset + ph * inputW;

                                for (var pw = 0; pw < poolSize; pw++)
                                {
                                    var absIdx = phInOffset + pw;
                                    var val = Unsafe.Add(ref inRef, absIdx);

                                    if (val > maxVal)
                                    {
                                        maxVal = val;
                                        maxIdx = absIdx;
                                    }
                                }
                            }

                            var outAbsIdx = ohOutOffset + ow;
                            Unsafe.Add(ref outRef, outAbsIdx) = maxVal;
                            Unsafe.Add(ref idxRef, outAbsIdx) = maxIdx;
                        }
                    }
                }
            });

            var output = new AutogradNode(resultData, input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MaxPool2D, output, input, maxIndices);
            }

            return output;
        }

        public static void MaxPool2DBackward(AutogradNode input, AutogradNode maxIndices, AutogradNode output)
        {
            ref var iGRef = ref MemoryMarshal.GetReference(input.Grad.AsSpan());
            ref var oGRef = ref MemoryMarshal.GetReference(output.Grad.AsSpan());
            ref var idxRef = ref MemoryMarshal.GetReference(maxIndices.Data.AsSpan());

            var len = maxIndices.Data.Size;

            for (var i = 0; i < len; i++)
            {
                var maxIdx = (int)Unsafe.Add(ref idxRef, i);
                Unsafe.Add(ref iGRef, maxIdx) += Unsafe.Add(ref oGRef, i);
            }
        }

        public static AutogradNode GlobalAveragePool2D(ComputationGraph graph, AutogradNode input, int channels, int h, int w)
        {
            var batchSize = input.Data.GetDim(0);
            var resData = new FastTensor<float>(batchSize, channels);
            float spatialSize = h * w;

            Parallel.For(0, batchSize, n =>
            {
                for (var c = 0; c < channels; c++)
                {
                    resData[n, c] = TensorPrimitives.Sum(input.Data.AsSpan().Slice(n * channels * h * w + c * h * w, h * w)) / spatialSize;
                }
            });

            var output = new AutogradNode(resData, input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.GlobalAveragePool2D, output, input, null, h, w, channels);
            }

            return output;
        }

        public static void GlobalAvgPool2DBackward(AutogradNode input, AutogradNode output, int h, int w, int channels)
        {
            var batchSize = input.Data.GetDim(0);
            float spatialSize = h * w;

            Parallel.For(0, batchSize, n =>
            {
                for (var c = 0; c < channels; c++)
                {
                    var gradSlice = input.Grad.AsSpan().Slice(n * channels * h * w + c * h * w, h * w);
                    var val = output.Grad[n, c] / spatialSize;
                    TensorPrimitives.Add(gradSlice, val, gradSlice);
                }
            });
        }

        // ====================================================================
        // 3. ACTIVATION AND REGULARIZATION
        // ====================================================================

        public static AutogradNode ReLU(ComputationGraph graph, AutogradNode input)
        {
            var res = FastTensor<float>.SameShape(input.Data, false);
            TensorPrimitives.Max(input.Data.AsSpan(), 0f, res.AsSpan());
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

            var inS = (ReadOnlySpan<float>)input.Data.AsSpan();
            var goS = (ReadOnlySpan<float>)output.Grad.AsSpan();
            var giS = input.Grad.AsSpan();
            var i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                var vZero = Vector<float>.Zero;
                var vecSize = Vector<float>.Count;

                for (; i <= inS.Length - vecSize; i += vecSize)
                {
                    var vIn = new Vector<float>(inS.Slice(i));
                    var vGo = new Vector<float>(goS.Slice(i));
                    var vGi = new Vector<float>(giS.Slice(i));
                    var vMask = Vector.GreaterThan(vIn, vZero);
                    var vFiltered = Vector.ConditionalSelect(vMask, vGo, vZero);
                    (vGi + vFiltered).CopyTo(giS.Slice(i));
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

        public static AutogradNode Dropout(ComputationGraph graph, AutogradNode input, float probability, bool isTraining)
        {
            var resData = FastTensor<float>.SameShape(input.Data);
            var mask = new AutogradNode(FastTensor<float>.SameShape(input.Data), false);

            if (isTraining)
            {
                var scale = 1f / (1f - probability);
                var size = input.Data.Size;
                var inS = input.Data.AsSpan();
                var resS = resData.AsSpan();
                var maskS = mask.Data.AsSpan();

                byte[] rentedArray = null;

                try
                {
                    var randomBytes = size <= 2048 ? stackalloc byte[size] : (rentedArray = ArrayPool<byte>.Shared.Rent(size)).AsSpan(0, size);

                    Random.Shared.NextBytes(randomBytes);

                    var thresholdByte = (byte)(probability * 255f);

                    for (var i = 0; i < size; i++)
                    {
                        maskS[i] = randomBytes[i] > thresholdByte ? scale : 0f;
                    }

                    TensorPrimitives.Multiply(inS, maskS, resS);
                }
                finally
                {
                    if (rentedArray != null)
                    {
                        ArrayPool<byte>.Shared.Return(rentedArray);
                    }
                }
            }
            else
            {
                input.Data.AsSpan().CopyTo(resData.AsSpan());
            }

            var output = new AutogradNode(resData, input.RequiresGrad);

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

        public static void DropoutBackward(AutogradNode input, AutogradNode mask, AutogradNode output)
        {
            var giS = input.Grad.AsSpan();
            var goS = output.Grad.AsSpan();
            var mS = mask.Data.AsSpan();
            TensorPrimitives.MultiplyAdd(goS, mS, giS, giS);
        }

        // ====================================================================
        // 4. LOSS FUNCTIONS
        // ====================================================================

        public static AutogradNode SoftmaxCrossEntropy(ComputationGraph graph, AutogradNode logits, AutogradNode target)
        {
            int rows = logits.Data.GetDim(0), cols = logits.Data.GetDim(1);
            var totalLoss = 0f;
            var probsTensor = new FastTensor<float>(rows, cols);

            for (var r = 0; r < rows; r++)
            {
                var pRow = probsTensor.AsSpan().Slice(r * cols, cols);
                TensorPrimitives.SoftMax(logits.Data.AsSpan().Slice(r * cols, cols), pRow);

                for (var c = 0; c < cols; c++)
                {
                    if (target.Data[r, c] > 0.5f)
                    {
                        totalLoss -= MathF.Log(pRow[c] + 1e-15f);
                    }
                }
            }

            var res = new FastTensor<float>(1, 1)
            {
                [0, 0] = totalLoss / rows
            };
            var output = new AutogradNode(res, logits.RequiresGrad);

            if (logits.RequiresGrad)
            {
                var probsNode = new AutogradNode(probsTensor, false);
                graph?.Record(OpCode.SoftmaxCrossEntropy, output, logits, target, nodeContext: [probsNode]);
            }
            else
            {
                probsTensor.Dispose();
            }

            return output;
        }

        public static void SoftmaxCrossEntropyBackward(AutogradNode logits, AutogradNode target, AutogradNode output, AutogradNode probsNode)
        {
            var rows = logits.Data.GetDim(0);
            var cols = logits.Data.GetDim(1);
            var scale = output.Grad.AsSpan()[0] / rows;
            var probsS = (ReadOnlySpan<float>)probsNode.Data.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var pS = probsS.Slice(r * cols, cols);
                var tS = target.Data.AsSpan().Slice(r * cols, cols);
                var gS = logits.Grad.AsSpan().Slice(r * cols, cols);

                TensorPrimitives.MultiplyAdd(pS, scale, gS, gS);
                TensorPrimitives.MultiplyAdd(tS, -scale, gS, gS);
            }
        }

        public static AutogradNode MSELoss(ComputationGraph graph, AutogradNode prediction, AutogradNode target)
        {
            var size = prediction.Data.Size;
            var diffArr = ArrayPool<float>.Shared.Rent(size);
            float mse;

            try
            {
                var diffSpan = diffArr.AsSpan(0, size);
                TensorPrimitives.Subtract(prediction.Data.AsSpan(), target.Data.AsSpan(), diffSpan);
                mse = TensorPrimitives.Dot(diffSpan, diffSpan) / size;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(diffArr);
            }

            var res = new FastTensor<float>(1, 1)
            {
                [0, 0] = mse
            };

            var output = new AutogradNode(res, prediction.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MseLoss, output, prediction, target);
            }

            return output;
        }

        public static void MSELossBackward(AutogradNode p, AutogradNode t, AutogradNode o)
        {
            var factor = o.Grad.AsSpan()[0] * (2f / p.Data.Size);
            var pGrad = p.Grad.AsSpan();
            var pData = p.Data.AsSpan();
            var tData = t.Data.AsSpan();

            TensorPrimitives.MultiplyAdd(pData, factor, pGrad, pGrad);
            TensorPrimitives.MultiplyAdd(tData, -factor, pGrad, pGrad);
        }

        // ====================================================================
        // 5. NORMALIZATION AND UTILITIES
        // ====================================================================

        public static AutogradNode BatchNorm1D(ComputationGraph graph, AutogradNode input, AutogradNode gamma, AutogradNode beta, FastTensor<float> runningMean, FastTensor<float> runningVar, float momentum, float eps, bool isTraining)
        {
            int N = input.Data.GetDim(0), C = input.Data.GetDim(1);
            var outputData = new FastTensor<float>(N, C);
            var mean = new AutogradNode(new FastTensor<float>(C), false);
            var invStd = new AutogradNode(new FastTensor<float>(C), false);

            if (isTraining)
            {
                var meanS = mean.Data.AsSpan();

                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Add(meanS, input.Data.AsSpan().Slice(i * C, C), meanS);
                }

                TensorPrimitives.Multiply(meanS, 1f / N, meanS);

                using var varBuf = new FastTensor<float>(true, C);
                using var tempBuf = new FastTensor<float>(false, C);
                var varS = varBuf.AsSpan();
                var tempS = tempBuf.AsSpan();
                var meanR = (ReadOnlySpan<float>)meanS;

                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Subtract(input.Data.AsSpan().Slice(i * C, C), meanR, tempS);
                    TensorPrimitives.MultiplyAdd(tempS, tempS, varS, varS);
                }

                TensorPrimitives.Multiply(varS, 1f / N, varS);

                var rmS = runningMean.AsSpan();
                var rvS = runningVar.AsSpan();
                var ivS = invStd.Data.AsSpan();

                TensorPrimitives.Multiply(rmS, 1f - momentum, rmS);
                TensorPrimitives.MultiplyAdd(meanR, momentum, rmS, rmS);

                TensorPrimitives.Multiply(rvS, 1f - momentum, rvS);
                TensorPrimitives.MultiplyAdd(varS, momentum, rvS, rvS);

                TensorPrimitives.Add(varS, eps, ivS);
                TensorPrimitives.ReciprocalSqrt(ivS, ivS);
            }
            else
            {
                runningMean.AsSpan().CopyTo(mean.Data.AsSpan());
                var ivS = invStd.Data.AsSpan();
                TensorPrimitives.Add(runningVar.AsSpan(), eps, ivS);
                TensorPrimitives.ReciprocalSqrt(ivS, ivS);
            }

            var invStdR = (ReadOnlySpan<float>)invStd.Data.AsSpan();
            var meanRo = (ReadOnlySpan<float>)mean.Data.AsSpan();
            var gammaR = (ReadOnlySpan<float>)gamma.Data.AsSpan();
            var betaR = (ReadOnlySpan<float>)beta.Data.AsSpan();

            for (var i = 0; i < N; i++)
            {
                var inRow = (ReadOnlySpan<float>)input.Data.AsSpan().Slice(i * C, C);
                var outRow = outputData.AsSpan().Slice(i * C, C);

                TensorPrimitives.Subtract(inRow, meanRo, outRow);
                TensorPrimitives.Multiply(outRow, invStdR, outRow);
                TensorPrimitives.MultiplyAdd(outRow, gammaR, betaR, outRow);
            }

            var output = new AutogradNode(outputData, input.RequiresGrad);
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

            var inS = input.Data.AsSpan();
            var outGradS = output.Grad.AsSpan();
            var meanS = (ReadOnlySpan<float>)mean.Data.AsSpan();
            var invStdS = (ReadOnlySpan<float>)invStd.Data.AsSpan();
            var gammaS = (ReadOnlySpan<float>)gamma.Data.AsSpan();

            float[] coeffArr = null;
            float[] termArr = null;
            float[] sumDyArr = null;
            float[] sumDyXHatArr = null;
            float[] xHatRowArr = null;

            try
            {
                var coeff = C <= StackAllocThreshold ? stackalloc float[C] : (coeffArr = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                var term = C <= StackAllocThreshold ? stackalloc float[C] : (termArr = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                var sumDy = C <= StackAllocThreshold ? stackalloc float[C] : (sumDyArr = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                var sumDyXHat = C <= StackAllocThreshold ? stackalloc float[C] : (sumDyXHatArr = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);
                var xHatRow = C <= StackAllocThreshold ? stackalloc float[C] : (xHatRowArr = ArrayPool<float>.Shared.Rent(C)).AsSpan(0, C);

                sumDy.Clear();
                sumDyXHat.Clear();

                TensorPrimitives.Multiply(gammaS, invStdS, coeff);
                TensorPrimitives.Multiply(coeff, 1f / N, coeff);

                for (var i = 0; i < N; i++)
                {
                    var gradRow = (ReadOnlySpan<float>)outGradS.Slice(i * C, C);
                    var inRow = (ReadOnlySpan<float>)inS.Slice(i * C, C);

                    TensorPrimitives.Subtract(inRow, meanS, xHatRow);
                    TensorPrimitives.Multiply(xHatRow, invStdS, xHatRow);

                    TensorPrimitives.Add(sumDy, gradRow, sumDy);
                    TensorPrimitives.MultiplyAdd(gradRow, xHatRow, sumDyXHat, sumDyXHat);

                    if (beta.RequiresGrad)
                    {
                        TensorPrimitives.Add(beta.Grad.AsSpan(), gradRow, beta.Grad.AsSpan());
                    }

                    if (gamma.RequiresGrad)
                    {
                        TensorPrimitives.MultiplyAdd(gradRow, xHatRow, gamma.Grad.AsSpan(), gamma.Grad.AsSpan());
                    }
                }

                if (input.RequiresGrad)
                {
                    var inGradS = input.Grad.AsSpan();
                    for (var i = 0; i < N; i++)
                    {
                        var gradRow = (ReadOnlySpan<float>)outGradS.Slice(i * C, C);
                        var inGradRow = inGradS.Slice(i * C, C);
                        var inRow = (ReadOnlySpan<float>)inS.Slice(i * C, C);

                        TensorPrimitives.Subtract(inRow, meanS, xHatRow);
                        TensorPrimitives.Multiply(xHatRow, invStdS, xHatRow);

                        TensorPrimitives.Multiply(gradRow, N, term);
                        TensorPrimitives.Subtract(term, sumDy, term);

                        TensorPrimitives.Multiply(xHatRow, sumDyXHat, xHatRow);
                        TensorPrimitives.Subtract(term, xHatRow, term);

                        TensorPrimitives.MultiplyAdd(coeff, term, inGradRow, inGradRow);
                    }
                }
            }
            finally
            {
                if (coeffArr != null)
                {
                    ArrayPool<float>.Shared.Return(coeffArr);
                }
                
                if (termArr != null)
                {
                    ArrayPool<float>.Shared.Return(termArr);
                }
                
                if (sumDyArr != null)
                {
                    ArrayPool<float>.Shared.Return(sumDyArr);
                }
                
                if (sumDyXHatArr != null)
                {
                    ArrayPool<float>.Shared.Return(sumDyXHatArr);
                }
                
                if (xHatRowArr != null)
                {
                    ArrayPool<float>.Shared.Return(xHatRowArr);
                }
            }
        }

        public static AutogradNode Reshape(ComputationGraph graph, AutogradNode input, params int[] newShape)
        {
            var output = new AutogradNode(input.Data.Reshape(newShape), input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Reshape, output, input);
            }

            return output;
        }

        public static void ReshapeBackward(AutogradNode input, AutogradNode output)
        {
            TensorPrimitives.Add(input.Grad.AsSpan(), output.Grad.AsSpan(), input.Grad.AsSpan());
        }

        public static void Im2Col(ReadOnlySpan<float> input, int channels, int height, int width, int kSize, int stride, int padding, Span<float> output)
        {
            var outH = (height + 2 * padding - kSize) / stride + 1;
            var outW = (width + 2 * padding - kSize) / stride + 1;
            var channelSize = height * width;

            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < kSize; kh++)
                {
                    for (var kw = 0; kw < kSize; kw++)
                    {
                        var rowOffset = (c * kSize * kSize + kh * kSize + kw) * outH * outW;
                        for (var y = 0; y < outH; y++)
                        {
                            var i = y * stride - padding + kh;
                            var outIdxY = rowOffset + y * outW;

                            if (i >= 0 && i < height)
                            {
                                var inputRowOffset = c * channelSize + i * width;

                                if (stride == 1)
                                {
                                    var startX = Math.Max(0, padding - kw);
                                    var endX = Math.Min(outW, width + padding - kw);

                                    if (startX > 0)
                                    {
                                        output.Slice(outIdxY, startX).Clear();
                                    }

                                    if (endX > startX)
                                    {
                                        var startJ = startX - padding + kw;
                                        var len = endX - startX;
                                        input.Slice(inputRowOffset + startJ, len).CopyTo(output.Slice(outIdxY + startX, len));
                                    }

                                    if (endX < outW)
                                    {
                                        output.Slice(outIdxY + endX, outW - endX).Clear();
                                    }
                                }
                                else
                                {
                                    for (var x = 0; x < outW; x++)
                                    {
                                        var j = x * stride - padding + kw;
                                        output[outIdxY + x] = j >= 0 && j < width ? input[inputRowOffset + j] : 0f;
                                    }
                                }
                            }
                            else
                            {
                                output.Slice(outIdxY, outW).Clear();
                            }
                        }
                    }
                }
            }
        }

        public static void Col2Im(ReadOnlySpan<float> colData, int channels, int height, int width, int kSize, int stride, int padding, Span<float> gradInput)
        {
            var outH = (height + 2 * padding - kSize) / stride + 1;
            var outW = (width + 2 * padding - kSize) / stride + 1;
            var channelSize = height * width;

            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < kSize; kh++)
                {
                    for (var kw = 0; kw < kSize; kw++)
                    {
                        var rowOffset = (c * kSize * kSize + kh * kSize + kw) * outH * outW;
                        for (var y = 0; y < outH; y++)
                        {
                            var i = y * stride - padding + kh;
                            var outIdxY = rowOffset + y * outW;

                            if (i >= 0 && i < height)
                            {
                                var inputRowOffset = c * channelSize + i * width;

                                if (stride == 1)
                                {
                                    var startX = Math.Max(0, padding - kw);
                                    var endX = Math.Min(outW, width + padding - kw);

                                    if (endX > startX)
                                    {
                                        var startJ = startX - padding + kw;
                                        var len = endX - startX;

                                        var inSlice = gradInput.Slice(inputRowOffset + startJ, len);
                                        var outSlice = colData.Slice(outIdxY + startX, len);

                                        TensorPrimitives.Add(inSlice, outSlice, inSlice);
                                    }
                                }
                                else
                                {
                                    for (var x = 0; x < outW; x++)
                                    {
                                        var j = x * stride - padding + kw;
                                        if (j >= 0 && j < width)
                                        {
                                            gradInput[inputRowOffset + j] += colData[outIdxY + x];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        public static AutogradNode Linear(ComputationGraph graph, AutogradNode input, AutogradNode weights, AutogradNode bias)
        {
            var product = MatMul(graph, input, weights);
            return AddBias(graph, product, bias);
        }

        public static AutogradNode DirectionalLoss(ComputationGraph graph, AutogradNode prediction, AutogradNode target, float gamma = 10f)
        {
            var pS = prediction.Data.AsSpan();
            var tS = target.Data.AsSpan();
            var size = pS.Length;
            float totalLoss;

            var tempArr = ArrayPool<float>.Shared.Rent(size);
            try
            {
                var tempS = tempArr.AsSpan(0, size);

                TensorPrimitives.Subtract(pS, tS, tempS);
                var sumMse = TensorPrimitives.SumOfSquares(tempS);

                TensorPrimitives.Multiply(pS, tS, tempS);
                TensorPrimitives.Min(tempS, 0f, tempS);

                var sumPenalty = TensorPrimitives.Sum(tempS) * -gamma;
                totalLoss = sumMse + sumPenalty;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(tempArr);
            }

            var res = new FastTensor<float>(1, 1)
            {
                [0, 0] = totalLoss / size
            };

            var output = new AutogradNode(res, prediction.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.DirectionalLoss, output, prediction, target, BitConverter.SingleToInt32Bits(gamma));
            }

            return output;
        }

        public static void DirectionalLossBackward(AutogradNode p, AutogradNode t, AutogradNode o, float gamma)
        {
            var scale = o.Grad.AsSpan()[0] / p.Data.Size;
            var mseScale = 2f * scale;
            var penaltyScale = gamma * scale;

            var pGrad = p.Grad.AsSpan();
            var pData = p.Data.AsSpan();
            var tData = t.Data.AsSpan();
            var i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                var vecSize = Vector<float>.Count;
                var vZero = Vector<float>.Zero;
                var vMseScale = new Vector<float>(mseScale);
                var vPenaltyScale = new Vector<float>(penaltyScale);

                for (; i <= pData.Length - vecSize; i += vecSize)
                {
                    var vP = new Vector<float>(pData.Slice(i));
                    var vT = new Vector<float>(tData.Slice(i));
                    var vG = new Vector<float>(pGrad.Slice(i));

                    var vDiff = vP - vT;
                    var vBaseGrad = vDiff * vMseScale;

                    var vProd = vP * vT;
                    var vWrongDirectionMask = Vector.LessThan(vProd, vZero);

                    var vPenaltyGrad = Vector.ConditionalSelect(
                    vWrongDirectionMask,
                    -vT * vPenaltyScale,
                    vZero
                    );

                    (vG + vBaseGrad + vPenaltyGrad).CopyTo(pGrad.Slice(i));
                }
            }

            for (; i < pData.Length; i++)
            {
                var baseGrad = 2f * (pData[i] - tData[i]) * scale;
                var penaltyGrad = pData[i] * tData[i] < 0f ? -tData[i] * penaltyScale : 0f;
                pGrad[i] += baseGrad + penaltyGrad;
            }
        }

        // ====================================================================
        // ZOPTYMALIZOWANE AKTYWACJE (CHUNKOWANIE NA STOSIE)
        // ====================================================================

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

            var outS = output.Data.AsReadOnlySpan();
            var outGS = output.Grad.AsReadOnlySpan();
            var inGS = input.Grad.AsSpan();
            var len = inGS.Length;

            Span<float> buf = stackalloc float[StackAllocThreshold];

            for (var i = 0; i < len; i += StackAllocThreshold)
            {
                var chunk = Math.Min(StackAllocThreshold, len - i);
                var oChunk = outS.Slice(i, chunk);
                var ogChunk = outGS.Slice(i, chunk);
                var igChunk = inGS.Slice(i, chunk);
                var b = buf.Slice(0, chunk);

                TensorPrimitives.Subtract(1f, oChunk, b);
                TensorPrimitives.Multiply(oChunk, b, b);
                TensorPrimitives.MultiplyAdd(ogChunk, b, igChunk, igChunk);
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

            var outS = output.Data.AsReadOnlySpan();
            var outGS = output.Grad.AsReadOnlySpan();
            var inGS = input.Grad.AsSpan();
            var len = inGS.Length;

            Span<float> buf = stackalloc float[StackAllocThreshold];

            for (var i = 0; i < len; i += StackAllocThreshold)
            {
                var chunk = Math.Min(StackAllocThreshold, len - i);
                var oChunk = outS.Slice(i, chunk);
                var ogChunk = outGS.Slice(i, chunk);
                var igChunk = inGS.Slice(i, chunk);
                var b = buf.Slice(0, chunk);

                TensorPrimitives.Multiply(oChunk, oChunk, b);
                TensorPrimitives.Subtract(1f, b, b);
                TensorPrimitives.MultiplyAdd(ogChunk, b, igChunk, igChunk);
            }
        }

        // ====================================================================
        // MULTIPLY (element-wise)
        // ====================================================================

        public static AutogradNode Multiply(ComputationGraph graph, AutogradNode a, AutogradNode b)
        {
            var res = FastTensor<float>.SameShape(a.Data, false);

            TensorPrimitives.Multiply(a.Data.AsReadOnlySpan(), b.Data.AsReadOnlySpan(), res.AsSpan());

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

        // ====================================================================
        // GATE SLICE
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static AutogradNode GateSlice(
            ComputationGraph graph,
            AutogradNode gates,
            int hiddenSize,
            int gateIndex)
        {
            var batch = gates.Data.GetDim(0);
            var offset = gateIndex * hiddenSize;
            var stride = 4 * hiddenSize;

            var res = new FastTensor<float>(false, batch, hiddenSize);
            var srcS = gates.Data.AsReadOnlySpan();
            var dstS = res.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                srcS.Slice(b * stride + offset, hiddenSize).CopyTo(dstS.Slice(b * hiddenSize, hiddenSize));
            }

            var output = new AutogradNode(res, gates.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.GateSlice, output, gates, null, gateIndex, hiddenSize);
            }

            return output;
        }

        public static void GateSliceBackward(
            AutogradNode gates,
            AutogradNode output,
            int hiddenSize,
            int gateIndex)
        {
            if (!gates.RequiresGrad)
            {
                return;
            }

            var batch = gates.Data.GetDim(0);
            var offset = gateIndex * hiddenSize;
            var stride = 4 * hiddenSize;

            var srcS = output.Grad.AsReadOnlySpan();
            var dstS = gates.Grad.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                var dst = dstS.Slice(b * stride + offset, hiddenSize);

                TensorPrimitives.Add(dst, srcS.Slice(b * hiddenSize, hiddenSize), dst);
            }
        }

        // ====================================================================
        // TIMESTEP SLICE BACKWARD
        // ====================================================================

        public static void TimestepSliceBackward(
            AutogradNode input,
            AutogradNode output,
            int t,
            int seqLen,
            int inputSize)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var batch = input.Data.GetDim(0);
            var srcS = output.Grad.AsReadOnlySpan();
            var dstS = input.Grad.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                var dst = dstS.Slice(b * seqLen * inputSize + t * inputSize, inputSize);
                TensorPrimitives.Add(dst, srcS.Slice(b * inputSize, inputSize), dst);
            }
        }

        // ====================================================================
        // STACK TIMESTEPS BACKWARD
        // ====================================================================

        public static void StackTimestepsBackward(
            AutogradNode[] allH,
            AutogradNode output,
            int batch,
            int seqLen,
            int hiddenSize)
        {
            var srcS = output.Grad.AsReadOnlySpan();

            for (var t = 0; t < seqLen; t++)
            {
                var h = allH[t];

                if (!h.RequiresGrad)
                {
                    continue;
                }

                var dstS = h.Grad.AsSpan();

                for (var b = 0; b < batch; b++)
                {
                    var dst = dstS.Slice(b * hiddenSize, hiddenSize);
                    TensorPrimitives.Add(
                    dst,
                    srcS.Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize),
                    dst);
                }
            }
        }

        // ====================================================================
        // REPEAT VECTOR
        // ====================================================================

        public static AutogradNode RepeatVector(ComputationGraph graph, AutogradNode input, int seqLen)
        {
            var batch = input.Data.GetDim(0);
            var hiddenSize = input.Data.GetDim(1);

            var res = new FastTensor<float>(false, batch, seqLen, hiddenSize);
            var srcS = input.Data.AsReadOnlySpan();
            var dstS = res.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                var src = srcS.Slice(b * hiddenSize, hiddenSize);

                for (var t = 0; t < seqLen; t++)
                {
                    src.CopyTo(dstS.Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize));
                }
            }

            var output = new AutogradNode(res, input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.RepeatVector, output, input, null, seqLen, hiddenSize);
            }

            return output;
        }

        // ====================================================================
        // REPEAT VECTOR BACKWARD
        // ====================================================================

        public static void RepeatVectorBackward(
            AutogradNode input,
            AutogradNode output,
            int seqLen,
            int hiddenSize)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var batch = input.Data.GetDim(0);
            var srcS = output.Grad.AsReadOnlySpan();
            var dstS = input.Grad.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                var dst = dstS.Slice(b * hiddenSize, hiddenSize);

                for (var t = 0; t < seqLen; t++)
                {
                    TensorPrimitives.Add(dst, srcS.Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize), dst);
                }
            }
        }
    }
}