using System.Collections.Concurrent;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Core
{
    public static class TensorMath
    {
        // ====================================================================
        // 1. PODSTAWOWA ALGEBRA
        // ====================================================================

        public static AutogradNode Add(AutogradNode left, AutogradNode right)
        {
            var resultData = new FastTensor<float>(left.Data.Shape);

            var lSpan = left.Data.AsSpan();
            var rSpan = right.Data.AsSpan();
            var resSpan = resultData.AsSpan();

            TensorPrimitives.Add(lSpan, rSpan, resSpan);

            var outputNode = new AutogradNode(resultData, left.RequiresGrad || right.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.Add, outputNode, left, right);

            return outputNode;
        }

        public static AutogradNode AddBias(AutogradNode input, AutogradNode bias)
        {
            var resultData = new FastTensor<float>(input.Data.Shape);

            var N = input.Data.Shape[0];
            var C = input.Data.Shape[1];

            var inSpan = input.Data.AsSpan();
            var bSpan = bias.Data.AsSpan();
            var resSpan = resultData.AsSpan();

            for (var i = 0; i < N; i++)
            {
                TensorPrimitives.Add(inSpan.Slice(i * C, C), bSpan, resSpan.Slice(i * C, C));
            }

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || bias.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.AddBias, outputNode, input, bias);

            return outputNode;
        }

        public static AutogradNode MatMul(AutogradNode left, AutogradNode right)
        {
            var resultData = MatMulRaw(left.Data, right.Data);

            var outputNode = new AutogradNode(resultData, left.RequiresGrad || right.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.MatMul, outputNode, left, right);

            return outputNode;
        }

        public static FastTensor<float> MatMulRaw(FastTensor<float> A, FastTensor<float> B)
        {
            var aRows = A.Shape[0];
            var aCols = A.Shape[1];
            var bCols = B.Shape[1];

            var C = new FastTensor<float>(aRows, bCols);

            var aSpan = A.AsSpan();
            var bSpan = B.AsSpan();
            var cSpan = C.AsSpan();

            for (var i = 0; i < aRows; i++)
            {
                var rowC = cSpan.Slice(i * bCols, bCols);
                rowC.Clear();

                for (var k = 0; k < aCols; k++)
                {
                    var valA = aSpan[i * aCols + k];
                    var rowB = bSpan.Slice(k * bCols, bCols);
                    TensorPrimitives.MultiplyAdd(rowB, valA, rowC, rowC);
                }
            }

            return C;
        }

        public static AutogradNode Linear(AutogradNode input, AutogradNode weights, AutogradNode bias)
        {
            var mm = MatMul(input, weights);
            return AddBias(mm, bias);
        }

        // ====================================================================
        // 2. AKTYWACJE, REGULARYZACJA I LOSS
        // ====================================================================

        public static AutogradNode ReLU(AutogradNode input)
        {
            var resultData = new FastTensor<float>(input.Data.Shape);
            TensorPrimitives.Max(input.Data.AsSpan(), 0f, resultData.AsSpan());

            var outputNode = new AutogradNode(resultData, input.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.ReLU, outputNode, input);

            return outputNode;
        }

        public static AutogradNode Dropout(AutogradNode input, float p, bool isTraining)
        {
            if (!isTraining) return input;

            var resultData = new FastTensor<float>(input.Data.Shape);
            var maskTensor = new AutogradNode(new FastTensor<float>(input.Data.Shape), requiresGrad: false);
            var scale = 1.0f / (1.0f - p);

            var inSpan = input.Data.AsSpan();
            var resSpan = resultData.AsSpan();
            var maskSpan = maskTensor.Data.AsSpan();

            for (var i = 0; i < inSpan.Length; i++)
            {
                if (Random.Shared.NextDouble() > p)
                {
                    maskSpan[i] = scale;
                    resSpan[i] = inSpan[i] * scale;
                }
                else
                {
                    maskSpan[i] = 0;
                    resSpan[i] = 0;
                }
            }

            var outputNode = new AutogradNode(resultData, input.RequiresGrad);
            if (input.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.Dropout, outputNode, input, maskTensor);
            else
                maskTensor.Dispose();

            return outputNode;
        }

        public static AutogradNode MSELoss(AutogradNode prediction, AutogradNode target)
        {
            var n = prediction.Data.Size;
            using var diffBuffer = new FastTensor<float>(n);
            var diffSpan = diffBuffer.AsSpan();

            TensorPrimitives.Subtract(prediction.Data.AsSpan(), target.Data.AsSpan(), diffSpan);
            var finalLoss = TensorPrimitives.SumOfSquares(diffSpan) / n;

            var resultMat = new FastTensor<float>(1, 1);
            resultMat[0, 0] = finalLoss;

            var outputNode = new AutogradNode(resultMat, prediction.RequiresGrad || target.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.MSELoss, outputNode, prediction, target);

            return outputNode;
        }

        public static AutogradNode SoftmaxCrossEntropy(AutogradNode logits, AutogradNode target)
        {
            var rows = logits.Data.Shape[0];
            var cols = logits.Data.Shape[1];
            var totalLoss = 0f;

            using var pRowBuf = new FastTensor<float>(cols);
            var pRow = pRowBuf.AsSpan();

            var lSpan = logits.Data.AsSpan();
            var tSpan = target.Data.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var lRow = lSpan.Slice(r * cols, cols);
                var tRow = tSpan.Slice(r * cols, cols);

                TensorPrimitives.SoftMax(lRow, pRow);

                for (var c = 0; c < cols; c++)
                {
                    if (tRow[c] > 0.5f) totalLoss -= MathF.Log(pRow[c] + 1e-15f);
                }
            }

            var resData = new FastTensor<float>(1, 1);
            resData[0, 0] = totalLoss / rows;

            var outputNode = new AutogradNode(resData, logits.RequiresGrad);
            if (logits.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.SoftmaxCrossEntropy, outputNode, logits, target);

            return outputNode;
        }

        // ====================================================================
        // 3. CNN OPERACJE (W PEŁNI PRZESTRZENNE - NCHW)
        // ====================================================================

        public static AutogradNode Conv2D(AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1;
            var batchSize = input.Data.Shape[0];
            var kSquareInC = k * k * inC;

            var resultData = new FastTensor<float>(batchSize, outC, outH, outW);

            var inSize = inC * h * w;
            var outSize = outC * outH * outW;

            // Zero-Copy Reshape na 2D dla MatMulRaw
            using var weights2D = weights.Data.Reshape(outC, kSquareInC);

            Parallel.For(0, batchSize, n =>
            {
                // POBIERAMY SPAN WEWNĄTRZ WĄTKU (BEZPIECZNE DLA REF STRUCT)
                var inSpan = input.Data.AsSpan();
                var resSpan = resultData.AsSpan();

                using var colData = new FastTensor<float>(kSquareInC, outH * outW);

                Im2Col(inSpan.Slice(n * inSize, inSize), inC, h, w, k, 1, 0, colData.AsSpan());

                using var batchResult = MatMulRaw(weights2D, colData);

                batchResult.AsSpan().CopyTo(resSpan.Slice(n * outSize, outSize));
            });

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || weights.RequiresGrad);

            if (outputNode.RequiresGrad)
            {
                ComputationGraph.Active.Record(OpCode.Conv2D, outputNode, input, weights,
                    i0: inC, i1: outC, i2: h, i3: w, i4: k);
            }

            return outputNode;
        }

        public static AutogradNode MaxPool2D(AutogradNode input, int channels, int inputH, int inputW, int poolSize)
        {
            var outputH = inputH / poolSize;
            var outputW = inputW / poolSize;
            var batchSize = input.Data.Shape[0];

            var resultData = new FastTensor<float>(batchSize, channels, outputH, outputW);
            var maxIndicesTensor = new AutogradNode(new FastTensor<float>(batchSize, channels, outputH, outputW), requiresGrad: false);

            Parallel.For(0, batchSize, n =>
            {
                for (var c = 0; c < channels; c++)
                {
                    for (var oh = 0; oh < outputH; oh++)
                    {
                        for (var ow = 0; ow < outputW; ow++)
                        {
                            var maxVal = float.MinValue;
                            var maxIdx = -1;

                            for (var ph = 0; ph < poolSize; ph++)
                            {
                                for (var pw = 0; pw < poolSize; pw++)
                                {
                                    var currentIdx = c * (inputH * inputW) + (oh * poolSize + ph) * inputW + (ow * poolSize + pw);
                                    var val = input.Data[n, c, oh * poolSize + ph, ow * poolSize + pw];

                                    if (val > maxVal)
                                    {
                                        maxVal = val; maxIdx = currentIdx;
                                    }
                                }
                            }
                            var outIdx = c * (outputH * outputW) + oh * outputW + ow;
                            resultData[n, c, oh, ow] = maxVal;
                            maxIndicesTensor.Data[n, c, oh, ow] = maxIdx;
                        }
                    }
                }
            });

            var outputNode = new AutogradNode(resultData, input.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.MaxPool2D, outputNode, input, maxIndicesTensor);
            else
                maxIndicesTensor.Dispose();

            return outputNode;
        }

        public static AutogradNode GlobalAveragePool2D(AutogradNode input, int channels, int h, int w)
        {
            var batchSize = input.Data.Shape[0];
            var spatialSize = h * w;
            var outputData = new FastTensor<float>(batchSize, channels);
            var invSpatialSize = 1f / spatialSize;

            var inSpan = input.Data.AsSpan();
            var outSpan = outputData.AsSpan();

            for (var b = 0; b < batchSize; b++)
            {
                for (var c = 0; c < channels; c++)
                {
                    var offset = b * (channels * spatialSize) + c * spatialSize;
                    var spatialSpan = inSpan.Slice(offset, spatialSize);

                    var sum = TensorPrimitives.Sum(spatialSpan);
                    outSpan[b * channels + c] = sum * invSpatialSize;
                }
            }

            var outputNode = new AutogradNode(outputData, input.RequiresGrad);

            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.GlobalAveragePool2D, outputNode, input, null, i0: channels, i1: h, i2: w);

            return outputNode;
        }

        public static AutogradNode BatchNorm1D(AutogradNode input, AutogradNode gamma, AutogradNode beta, FastTensor<float> runningMean, FastTensor<float> runningVar, float momentum, float eps, bool isTraining)
        {
            var N = input.Data.Shape[0];
            var C = input.Data.Shape[1];
            var resultData = new FastTensor<float>(N, C);

            if (!isTraining)
            {
                using var scaleBuf = new FastTensor<float>(C);
                using var shiftBuf = new FastTensor<float>(C);
                var scale = scaleBuf.AsSpan();
                var shift = shiftBuf.AsSpan();

                var gSpan = gamma.Data.AsSpan();
                var bSpan = beta.Data.AsSpan();
                var rmSpan = runningMean.AsSpan();
                var rvSpan = runningVar.AsSpan();

                for (var j = 0; j < C; j++)
                {
                    var invStd = 1f / MathF.Sqrt(rvSpan[j] + eps);
                    scale[j] = gSpan[j] * invStd;
                    shift[j] = bSpan[j] - (rmSpan[j] * scale[j]);
                }

                var inSpan = input.Data.AsSpan();
                var resSpan = resultData.AsSpan();

                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.MultiplyAdd(inSpan.Slice(i * C, C), scale, shift, resSpan.Slice(i * C, C));
                }

                return new AutogradNode(resultData, requiresGrad: false);
            }

            using var batchMeanBuf = new FastTensor<float>(C);
            using var batchVarBuf = new FastTensor<float>(C);
            var batchMean = batchMeanBuf.AsSpan();
            var batchVar = batchVarBuf.AsSpan();

            var invStdTensor = new AutogradNode(new FastTensor<float>(1, C), requiresGrad: false);
            var xHatTensor = new AutogradNode(new FastTensor<float>(N, C), requiresGrad: false);
            var invStdVec = invStdTensor.Data.AsSpan();

            batchMean.Clear();
            var iSpan = input.Data.AsSpan();

            for (var i = 0; i < N; i++)
                TensorPrimitives.Add(batchMean, iSpan.Slice(i * C, C), batchMean);

            TensorPrimitives.Multiply(batchMean, 1f / N, batchMean);

            batchVar.Clear();
            using var tempRowMat = new FastTensor<float>(C);
            var tempRow = tempRowMat.AsSpan();

            for (var i = 0; i < N; i++)
            {
                TensorPrimitives.Subtract(iSpan.Slice(i * C, C), batchMean, tempRow);
                TensorPrimitives.MultiplyAdd(tempRow, tempRow, batchVar, batchVar);
            }
            TensorPrimitives.Multiply(batchVar, 1f / N, batchVar);

            var rmSpanLocal = runningMean.AsSpan();
            var rvSpanLocal = runningVar.AsSpan();

            for (var j = 0; j < C; j++)
            {
                rmSpanLocal[j] = (1 - momentum) * rmSpanLocal[j] + momentum * batchMean[j];
                rvSpanLocal[j] = (1 - momentum) * rvSpanLocal[j] + momentum * batchVar[j];
                invStdVec[j] = 1f / MathF.Sqrt(batchVar[j] + eps);
            }

            var rSpan = resultData.AsSpan();
            var xHatSpan = xHatTensor.Data.AsSpan();
            var gSpanLocal = gamma.Data.AsSpan();
            var bSpanLocal = beta.Data.AsSpan();

            for (var i = 0; i < N; i++)
            {
                var rowIn = iSpan.Slice(i * C, C);
                var rowOut = rSpan.Slice(i * C, C);
                var rowXHat = xHatSpan.Slice(i * C, C);

                for (var j = 0; j < C; j++)
                {
                    var xHat = (rowIn[j] - batchMean[j]) * invStdVec[j];
                    rowXHat[j] = xHat;
                    rowOut[j] = gSpanLocal[j] * xHat + bSpanLocal[j];
                }
            }

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad);

            if (outputNode.RequiresGrad)
            {
                ComputationGraph.Active.Record(OpCode.BatchNorm1D, outputNode, input, null,
                    nodeContext: [gamma, beta, xHatTensor, invStdTensor]);
            }
            else
            {
                xHatTensor.Dispose();
                invStdTensor.Dispose();
            }

            return outputNode;
        }

        // ====================================================================
        // 4. BACKWARD OPERATIONS
        // ====================================================================

        public static void AddBackward(AutogradNode left, AutogradNode right, AutogradNode output)
        {
            if (left.RequiresGrad)
                TensorPrimitives.Add(left.Grad.AsSpan(), output.Grad.AsSpan(), left.Grad.AsSpan());

            if (right.RequiresGrad)
                TensorPrimitives.Add(right.Grad.AsSpan(), output.Grad.AsSpan(), right.Grad.AsSpan());
        }

        public static void AddBiasBackward(AutogradNode input, AutogradNode bias, AutogradNode output)
        {
            if (input.RequiresGrad)
                TensorPrimitives.Add(input.Grad.AsSpan(), output.Grad.AsSpan(), input.Grad.AsSpan());

            if (bias.RequiresGrad)
            {
                var bGrad = bias.Grad.AsSpan();
                var outGradSpan = output.Grad.AsSpan();
                var N = output.Grad.Shape[0];
                var C = output.Grad.Shape[1];

                for (var i = 0; i < N; i++)
                    TensorPrimitives.Add(bGrad, outGradSpan.Slice(i * C, C), bGrad);
            }
        }

        public static void MatMulBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                using var bT = b.Data.Transpose(0, 1).ToContiguous();
                using var gradA = MatMulRaw(output.Grad, bT);
                TensorPrimitives.Add(a.Grad.AsSpan(), gradA.AsSpan(), a.Grad.AsSpan());
            }
            if (b.RequiresGrad)
            {
                using var aT = a.Data.Transpose(0, 1).ToContiguous();
                using var gradB = MatMulRaw(aT, output.Grad);
                TensorPrimitives.Add(b.Grad.AsSpan(), gradB.AsSpan(), b.Grad.AsSpan());
            }
        }

        public static void ReluBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad) return;
            var inSpan = input.Data.AsSpan();
            var gradOut = output.Grad.AsSpan();
            var gradIn = input.Grad.AsSpan();

            for (var i = 0; i < inSpan.Length; i++)
            {
                if (inSpan[i] > 0) gradIn[i] += gradOut[i];
            }
        }

        public static void DropoutBackward(AutogradNode input, AutogradNode mask, AutogradNode output)
        {
            if (!input.RequiresGrad) return;
            var gradIn = input.Grad.AsSpan();
            var gradOut = output.Grad.AsSpan();
            var mSpan = mask.Data.AsSpan();

            TensorPrimitives.MultiplyAdd(gradOut, mSpan, gradIn, gradIn);
        }

        public static void MSELossBackward(AutogradNode prediction, AutogradNode target, AutogradNode output)
        {
            var n = prediction.Data.Size;
            var m = (2f / n) * output.Grad[0]; // Output grad to skalar

            if (prediction.RequiresGrad)
            {
                var pGrad = prediction.Grad.AsSpan();
                TensorPrimitives.MultiplyAdd(prediction.Data.AsSpan(), m, pGrad, pGrad);
                TensorPrimitives.MultiplyAdd(target.Data.AsSpan(), -m, pGrad, pGrad);
            }
            if (target.RequiresGrad)
            {
                var tGrad = target.Grad.AsSpan();
                TensorPrimitives.MultiplyAdd(prediction.Data.AsSpan(), -m, tGrad, tGrad);
                TensorPrimitives.MultiplyAdd(target.Data.AsSpan(), m, tGrad, tGrad);
            }
        }

        public static void SoftmaxCrossEntropyBackward(AutogradNode logits, AutogradNode target, AutogradNode output)
        {
            if (!logits.RequiresGrad) return;

            var rows = logits.Data.Shape[0];
            var cols = logits.Data.Shape[1];
            var scale = output.Grad.AsSpan()[0] / rows;

            using var pRowBuf = new FastTensor<float>(cols);
            using var diffBuf = new FastTensor<float>(cols);

            var pRow = pRowBuf.AsSpan();
            var diff = diffBuf.AsSpan();

            var lSpan = logits.Data.AsSpan();
            var tSpan = target.Data.AsSpan();
            var lgSpan = logits.Grad.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var lRow = lSpan.Slice(r * cols, cols);
                var tRow = tSpan.Slice(r * cols, cols);
                var lGrad = lgSpan.Slice(r * cols, cols);

                TensorPrimitives.SoftMax(lRow, pRow);

                TensorPrimitives.Subtract(pRow, tRow, diff);
                TensorPrimitives.MultiplyAdd(diff, scale, lGrad, lGrad);
            }
        }

        public static void Conv2DBackward(AutogradNode input, AutogradNode weights, AutogradNode output, int inC, int outC, int h, int w, int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1;
            var batchSize = input.Data.Shape[0];
            var kSquareInC = k * k * inC;
            var weightGradSize = outC * kSquareInC;

            var inSize = inC * h * w;
            var outSize = outC * outH * outW;

            var threadResults = new ConcurrentQueue<float[]>();

            Parallel.For<float[]>(0, batchSize,
                () => new float[weightGradSize],

                (int n, ParallelLoopState loopState, float[] threadLocalWGrad) =>
                {
                    // POBIERAMY SPAN WEWNĄTRZ WĄTKU
                    var inSpan = input.Data.AsSpan();
                    var outGradSpan = output.Grad.AsSpan();

                    using var gn = new FastTensor<float>(outC, outH * outW);
                    outGradSpan.Slice(n * outSize, outSize).CopyTo(gn.AsSpan());

                    if (weights.RequiresGrad)
                    {
                        using var colData = new FastTensor<float>(kSquareInC, outH * outW);
                        Im2Col(inSpan.Slice(n * inSize, inSize), inC, h, w, k, 1, 0, colData.AsSpan());

                        using var colT = colData.Transpose(0, 1).ToContiguous();
                        using var dW_batch = MatMulRaw(gn, colT);

                        TensorPrimitives.Add(threadLocalWGrad, dW_batch.AsSpan(), threadLocalWGrad);
                    }

                    if (input.RequiresGrad)
                    {
                        using var w2D = weights.Data.Reshape(outC, kSquareInC);
                        using var wT = w2D.Transpose(0, 1).ToContiguous();
                        using var dX_col = MatMulRaw(wT, gn);

                        Col2Im(dX_col.AsSpan(), inC, h, w, k, 1, 0, input.Grad.AsSpan().Slice(n * inSize, inSize));
                    }

                    return threadLocalWGrad;
                },

                (float[] threadLocalWGrad) =>
                {
                    threadResults.Enqueue(threadLocalWGrad);
                }
            );

            if (weights.RequiresGrad)
            {
                var globalWGrad = weights.Grad.AsSpan();
                foreach (var localGrad in threadResults)
                {
                    TensorPrimitives.Add(globalWGrad, localGrad, globalWGrad);
                }
            }
        }

        public static void MaxPool2DBackward(AutogradNode input, AutogradNode maxIndices, AutogradNode output)
        {
            var batchSize = input.Data.Shape[0];
            var channels = input.Data.Shape[1];
            var inH = input.Data.Shape[2];
            var inW = input.Data.Shape[3];

            var outH = output.Data.Shape[2];
            var outW = output.Data.Shape[3];

            var inSize = channels * inH * inW;
            var outSize = channels * outH * outW;

            Parallel.For(0, batchSize, n =>
            {
                // POBIERAMY SPAN WEWNĄTRZ WĄTKU
                var gradOutSpan = output.Grad.AsSpan();
                var gradInSpan = input.Grad.AsSpan();
                var idxSpan = maxIndices.Data.AsSpan();

                var goRow = gradOutSpan.Slice(n * outSize, outSize);
                var giRow = gradInSpan.Slice(n * inSize, inSize);
                var idxRow = idxSpan.Slice(n * outSize, outSize);

                for (var i = 0; i < goRow.Length; i++)
                {
                    var originalIdx = (int)idxRow[i];
                    giRow[originalIdx] += goRow[i];
                }
            });
        }

        public static void GlobalAvgPool2DBackward(AutogradNode input, AutogradNode output, int channels, int h, int w)
        {
            var batchSize = input.Data.Shape[0];
            var spatialSize = h * w;
            var invSpatialSize = 1f / spatialSize;

            var inGradSpan = input.Grad.AsSpan();
            var outGradSpan = output.Grad.AsSpan();

            for (var b = 0; b < batchSize; b++)
            {
                for (var c = 0; c < channels; c++)
                {
                    var gradOut = outGradSpan[b * channels + c];
                    var offset = b * (channels * spatialSize) + c * spatialSize;

                    var gInSpan = inGradSpan.Slice(offset, spatialSize);
                    TensorPrimitives.Add(gInSpan, gradOut * invSpatialSize, gInSpan);
                }
            }
        }

        public static void BatchNorm1DBackward(AutogradNode input, AutogradNode output, AutogradNode gamma, AutogradNode beta, AutogradNode xHatTensor, AutogradNode invStdTensor)
        {
            var N = input.Data.Shape[0];
            var C = input.Data.Shape[1];

            var gradOutSpan = output.Grad.AsSpan();
            var xHatSpan = xHatTensor.Data.AsSpan();

            using var dGammaBuf = new FastTensor<float>(C);
            using var dBetaBuf = new FastTensor<float>(C);

            var dGamma = dGammaBuf.AsSpan();
            var dBeta = dBetaBuf.AsSpan();

            for (var i = 0; i < N; i++)
            {
                var goRow = gradOutSpan.Slice(i * C, C);
                TensorPrimitives.MultiplyAdd(goRow, xHatSpan.Slice(i * C, C), dGamma, dGamma);
                TensorPrimitives.Add(dBeta, goRow, dBeta);
            }

            if (gamma.RequiresGrad) TensorPrimitives.Add(gamma.Grad.AsSpan(), dGamma, gamma.Grad.AsSpan());
            if (beta.RequiresGrad) TensorPrimitives.Add(beta.Grad.AsSpan(), dBeta, beta.Grad.AsSpan());

            if (input.RequiresGrad)
            {
                using var factorBuf = new FastTensor<float>(C);
                using var temp1Buf = new FastTensor<float>(C);
                using var termBuf = new FastTensor<float>(C);

                var factor = factorBuf.AsSpan();
                var temp1 = temp1Buf.AsSpan();
                var term = termBuf.AsSpan();

                TensorPrimitives.Multiply(gamma.Data.AsSpan(), invStdTensor.Data.AsSpan(), factor);
                TensorPrimitives.Multiply(factor, 1f / N, factor);

                var gradInSpan = input.Grad.AsSpan();

                for (var i = 0; i < N; i++)
                {
                    var giRow = gradInSpan.Slice(i * C, C);
                    var goRow = gradOutSpan.Slice(i * C, C);
                    var xHRow = xHatSpan.Slice(i * C, C);

                    TensorPrimitives.Multiply(goRow, N, term);
                    TensorPrimitives.Subtract(term, dBeta, term);

                    TensorPrimitives.Multiply(xHRow, dGamma, temp1);
                    TensorPrimitives.Subtract(term, temp1, term);

                    TensorPrimitives.MultiplyAdd(term, factor, giRow, giRow);
                }
            }
        }

        // ====================================================================
        // 5. CNN HELPERS (Im2Col / Col2Im)
        // ====================================================================

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
                                for (var x = 0; x < outW; x++)
                                {
                                    var j = x * stride - padding + kw;
                                    if (j >= 0 && j < width) output[outIdxY + x] = input[c * channelSize + i * width + j];
                                    else output[outIdxY + x] = 0f;
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
            gradInput.Clear();

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
                                for (var x = 0; x < outW; x++)
                                {
                                    var j = x * stride - padding + kw;
                                    if (j >= 0 && j < width)
                                    {
                                        gradInput[c * channelSize + i * width + j] += colData[outIdxY + x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        public static void ReshapeBackward(AutogradNode input, AutogradNode output)
        {
            if (input.RequiresGrad && input.Grad != null && output.Grad != null)
            {
                // Gradient płynie przez Reshape bez zmian wartości, dodajemy go do wejścia
                TensorPrimitives.Add(input.Grad.AsSpan(), output.Grad.AsSpan(), input.Grad.AsSpan());
            }
        }

        public static AutogradNode Reshape(AutogradNode input, params int[] newShape)
        {
            // Zero-Copy Reshape na danych
            var reshapedData = input.Data.Reshape(newShape);
            var outputNode = new AutogradNode(reshapedData, input.RequiresGrad);

            // Rejestrujemy operację, aby gradienty mogły przepłynąć wstecz
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.Reshape, outputNode, input);

            return outputNode;
        }

        // Dodaj też obsługę w ComputationGraph.ExecuteBackward:
        // case OpCode.Reshape: op.A.Grad.AsSpan().Add(op.Output.Grad.AsSpan()); break;
    }
}