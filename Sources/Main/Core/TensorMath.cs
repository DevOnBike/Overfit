using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Core
{
    public static class TensorMath
    {
        // ====================================================================
        // 1. PODSTAWOWA ALGEBRA I MATERIE (ZOPTYMALIZOWANE)
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
            var N = input.Data.GetDim(0); var C = input.Data.GetDim(1);
            var resultData = new FastTensor<float>(N, C);
            var inS = input.Data.AsSpan(); var bS = bias.Data.AsSpan(); var resS = resultData.AsSpan();

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

            Parallel.For(0, aRows, i =>
            {
                var aS = A.AsSpan(); var bS = B.AsSpan(); var cS = C.AsSpan();
                var rowC = cS.Slice(i * bCols, bCols);

                for (var k = 0; k < aCols; k++)
                {
                    var valA = aS[i * aCols + k];

                    if (valA == 0)
                    {
                        continue;
                    }

                    TensorPrimitives.MultiplyAdd(bS.Slice(k * bCols, bCols), valA, rowC, rowC);
                }
            });

            return C;
        }

        private static void MatMulRawSequential(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, int aR, int aC, int bC, Span<float> cS)
        {
            cS.Clear();

            for (var i = 0; i < aR; i++)
            {
                var rowC = cS.Slice(i * bC, bC);
                var rowA = aS.Slice(i * aC, aC);

                for (var k = 0; k < aC; k++)
                {
                    var valA = rowA[k];

                    if (valA != 0f)
                    {
                        TensorPrimitives.MultiplyAdd(bS.Slice(k * bC, bC), valA, rowC, rowC);
                    }
                }
            }
        }

        public static void MatMulBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                using var gradA = MatMul_A_BT(output.Grad, b.Data);
                TensorPrimitives.Add(a.Grad.AsSpan(), gradA.AsSpan(), a.Grad.AsSpan());
            }

            if (b.RequiresGrad)
            {
                using var gradB = MatMul_AT_B(a.Data, output.Grad);
                TensorPrimitives.Add(b.Grad.AsSpan(), gradB.AsSpan(), b.Grad.AsSpan());
            }
        }

        private static FastTensor<float> MatMul_A_BT(FastTensor<float> A, FastTensor<float> B)
        {
            int N = A.GetDim(0), K = A.GetDim(1), M = B.GetDim(0);
            var C = new FastTensor<float>(false, N, M);

            Parallel.For(0, N, i =>
            {
                var aRow = A.AsSpan().Slice(i * K, K);
                var cRow = C.AsSpan().Slice(i * M, M);
                var bS = B.AsSpan();

                for (var j = 0; j < M; j++)
                {
                    cRow[j] = TensorPrimitives.Dot(aRow, bS.Slice(j * K, K));
                }
            });

            return C;
        }

        private static FastTensor<float> MatMul_AT_B(FastTensor<float> A, FastTensor<float> B)
        {
            var K = A.GetDim(0);
            var N = A.GetDim(1);
            var M = B.GetDim(1);
            var C = new FastTensor<float>(true, N, M);

            Parallel.For(0, N, i =>
            {
                var cRow = C.AsSpan().Slice(i * M, M);
                var aS = A.AsSpan();
                var bS = B.AsSpan();

                for (var k = 0; k < K; k++)
                {
                    var aVal = aS[k * N + i];

                    if (aVal == 0)
                    {
                        continue;
                    }

                    TensorPrimitives.MultiplyAdd(bS.Slice(k * M, M), aVal, cRow, cRow);
                }
            });

            return C;
        }

        // ====================================================================
        // 2. CNN - CONV, POOL, GAP (NCHW)
        // ====================================================================

        public static AutogradNode Conv2D(ComputationGraph graph, AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.GetDim(0), kSqInC = k * k * inC;
            int colSizePerImg = kSqInC * outH * outW, inSize = inC * h * w, outSize = outC * outH * outW;

            using var workspace = new FastTensor<float>(false, batchSize, colSizePerImg);
            var resultData = new FastTensor<float>(batchSize, outC, outH, outW);
            using var weights2D = weights.Data.Reshape(outC, kSqInC);

            Parallel.For(0, batchSize, n =>
            {
                var colSpan = workspace.AsSpan().Slice(n * colSizePerImg, colSizePerImg);
                Im2Col(input.Data.AsSpan().Slice(n * inSize, inSize), inC, h, w, k, 1, 0, colSpan);
                MatMulRawSequential(weights2D.AsSpan(), colSpan, outC, kSqInC, outH * outW, resultData.AsSpan().Slice(n * outSize, outSize));
            });

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || weights.RequiresGrad);

            if (outputNode.RequiresGrad)
            {
                graph?.Record(OpCode.Conv2D, outputNode, input, weights, inC, outC, h, w, k);
            }

            return outputNode;
        }

        public static void Conv2DBackward(AutogradNode input, AutogradNode weights, AutogradNode output, int inC, int outC, int h, int w, int k)
        {
            if (!input.RequiresGrad && !weights.RequiresGrad) return;

            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.GetDim(0), kSqInC = k * k * inC;
            int colSizePerImg = kSqInC * outH * outW, inSize = inC * h * w, outSize = outC * outH * outW;
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
                () => weights.RequiresGrad ? new FastTensor<float>(true, outC, kSqInC) : null,
                (n, loopState, localDw) =>
                {
                    using var colData = new FastTensor<float>(false, kSqInC, K);
                    Im2Col(input.Data.AsSpan().Slice(n * inSize, inSize), inC, h, w, k, 1, 0, colData.AsSpan());

                    var outGradSlice = output.Grad.AsSpan().Slice(n * outSize, outSize);
                    using var outGradMat = new FastTensor<float>(false, outC, K);
                    outGradSlice.CopyTo(outGradMat.AsSpan());

                    if (weights.RequiresGrad)
                    {
                        var dwSpan = localDw.AsSpan();
                        var outGradSpan = outGradMat.AsSpan();
                        var colSpan = colData.AsSpan();

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
                        using var dCol = new FastTensor<float>(false, kSqInC, K);
                        MatMulRawSequential(weights2DTContig.AsSpan(), outGradMat.AsSpan(), kSqInC, outC, K, dCol.AsSpan());
                        Col2Im(dCol.AsSpan(), inC, h, w, k, 1, 0, input.Grad.AsSpan().Slice(n * inSize, inSize));
                    }

                    return localDw;
                },
                (localDw) =>
                {
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
                                        maxVal = val; maxIdx = absIdx;
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
        // 3. AKTYWACJE I REGULARYZACJA
        // ====================================================================

        public static AutogradNode ReLU(ComputationGraph graph, AutogradNode input)
        {
            var res = FastTensor<float>.SameShape(input.Data, clearMemory: false);
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
            if (!input.RequiresGrad) return;

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

                for (var i = 0; i < input.Data.Size; i++)
                {
                    if (Random.Shared.NextSingle() > probability)
                    {
                        resData.AsSpan()[i] = input.Data.AsSpan()[i] * scale;
                        mask.Data.AsSpan()[i] = scale;
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
        // 4. FUNKCJE STRATY (LOSS)
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

            var res = new FastTensor<float>(1, 1) { [0, 0] = totalLoss / rows };
            var output = new AutogradNode(res, logits.RequiresGrad);

            if (logits.RequiresGrad)
            {
                var probsNode = new AutogradNode(probsTensor, requiresGrad: false);
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
            using var diff = FastTensor<float>.SameShape(prediction.Data);
            TensorPrimitives.Subtract(prediction.Data.AsSpan(), target.Data.AsSpan(), diff.AsSpan());

            var mse = TensorPrimitives.Dot(diff.AsSpan(), diff.AsSpan()) / prediction.Data.Size;
            var res = new FastTensor<float>(1, 1) { [0, 0] = mse };
            var output = new AutogradNode(res, prediction.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MSELoss, output, prediction, target);
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
        // 5. BATCH NORM I NARZĘDZIA
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
            if (!input.RequiresGrad && !gamma.RequiresGrad && !beta.RequiresGrad) return;

            int N = input.Data.GetDim(0), C = input.Data.GetDim(1);

            var inS = input.Data.AsSpan();
            var outGradS = output.Grad.AsSpan();
            var meanS = (ReadOnlySpan<float>)mean.Data.AsSpan();
            var invStdS = (ReadOnlySpan<float>)invStd.Data.AsSpan();
            var gammaS = (ReadOnlySpan<float>)gamma.Data.AsSpan();

            const int StackAllocThreshold = 256;

            FastBuffer<float> xHatBuf = null;
            FastBuffer<float> coeffBuf = null;
            FastBuffer<float> termBuf = null;

            try
            {
                var xHatRow = C <= StackAllocThreshold ? stackalloc float[C] : (xHatBuf = new FastBuffer<float>(C)).AsSpan();
                var coeff = C <= StackAllocThreshold ? stackalloc float[C] : (coeffBuf = new FastBuffer<float>(C)).AsSpan();
                var term = C <= StackAllocThreshold ? stackalloc float[C] : (termBuf = new FastBuffer<float>(C)).AsSpan();
                var sumDy = C <= StackAllocThreshold ? stackalloc float[C] : new float[C];
                var sumDyXHat = C <= StackAllocThreshold ? stackalloc float[C] : new float[C];

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

                    if (beta.RequiresGrad) TensorPrimitives.Add(beta.Grad.AsSpan(), gradRow, beta.Grad.AsSpan());
                    if (gamma.RequiresGrad) TensorPrimitives.MultiplyAdd(gradRow, xHatRow, gamma.Grad.AsSpan(), gamma.Grad.AsSpan());
                }

                if (input.RequiresGrad)
                {
                    var inGradS = input.Grad.AsSpan();

                    for (var i = 0; i < N; i++)
                    {
                        var gradRow = (ReadOnlySpan<float>)outGradS.Slice(i * C, C);
                        var inRow = (ReadOnlySpan<float>)inS.Slice(i * C, C);
                        var inGradRow = inGradS.Slice(i * C, C);

                        TensorPrimitives.Subtract(inRow, meanS, xHatRow);
                        TensorPrimitives.Multiply(xHatRow, invStdS, xHatRow);

                        TensorPrimitives.Multiply(gradRow, (float)N, term);
                        TensorPrimitives.Subtract(term, sumDy, term);

                        TensorPrimitives.Multiply(xHatRow, sumDyXHat, xHatRow);
                        TensorPrimitives.Subtract(term, xHatRow, term);

                        TensorPrimitives.MultiplyAdd(coeff, term, inGradRow, inGradRow);
                    }
                }
            }
            finally
            {
                xHatBuf?.Dispose();
                coeffBuf?.Dispose();
                termBuf?.Dispose();
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

                                    if (startX > 0) output.Slice(outIdxY, startX).Clear();

                                    if (endX > startX)
                                    {
                                        var startJ = startX - padding + kw;
                                        var len = endX - startX;
                                        input.Slice(inputRowOffset + startJ, len).CopyTo(output.Slice(outIdxY + startX, len));
                                    }

                                    if (endX < outW) output.Slice(outIdxY + endX, outW - endX).Clear();
                                }
                                else
                                {
                                    for (var x = 0; x < outW; x++)
                                    {
                                        var j = x * stride - padding + kw;
                                        output[outIdxY + x] = (j >= 0 && j < width) ? input[inputRowOffset + j] : 0f;
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

            using var temp = FastTensor<float>.SameShape(prediction.Data, clearMemory: false);
            var tempS = temp.AsSpan();

            TensorPrimitives.Subtract(pS, tS, tempS);
            var sumMse = TensorPrimitives.SumOfSquares(tempS);

            TensorPrimitives.Multiply(pS, tS, tempS);
            TensorPrimitives.Min(tempS, 0f, tempS);

            var sumPenalty = TensorPrimitives.Sum(tempS) * -gamma;

            var totalLoss = sumMse + sumPenalty;
            var res = new FastTensor<float>(1, 1) { [0, 0] = totalLoss / pS.Length };
            var output = new AutogradNode(res, prediction.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(
                    OpCode.DirectionalLoss,
                    output,
                    prediction,
                    target,
                    BitConverter.SingleToInt32Bits(gamma)
                );
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
                var penaltyGrad = (pData[i] * tData[i] < 0f) ? -tData[i] * penaltyScale : 0f;
                pGrad[i] += baseGrad + penaltyGrad;
            }
        }
    }
}