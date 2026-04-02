using System.Numerics;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Core
{
    public static class TensorMath
    {
        // ====================================================================
        // 1. BROADCASTING & CORE
        // ====================================================================

        public static FastMatrixView<T> BroadcastRowVector<T>(Span<T> rowVector, int targetRows)
            where T : struct, IFloatingPointIeee754<T>
        {
            return new FastMatrixView<T>(rowVector, targetRows, rowVector.Length, 0, 1, 0);
        }

        public static AutogradNode Add(AutogradNode left, AutogradNode right)
        {
            var resultData = new FastMatrix<float>(left.Data.Rows, left.Data.Cols);

            // Używa niskopoziomowego TensorPrimitives (void Add) pod spodem
            Add(left.Data.AsView(), right.Data.AsView(), resultData.AsView());

            var outputNode = new AutogradNode(resultData, left.RequiresGrad || right.RequiresGrad);

            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.Add, outputNode, left, right);

            return outputNode;
        }

        public static void Add<T>(FastMatrixView<T> left, FastMatrixView<T> right, FastMatrixView<T> result)
            where T : struct, IFloatingPointIeee754<T>
        {
            for (var i = 0; i < result.Rows; i++)
                TensorPrimitives.Add(left.Row(i), right.Row(i), result.Row(i));
        }

        public static void MatMul<T>(FastMatrixView<T> A, FastMatrixView<T> B, FastMatrixView<T> C)
            where T : struct, IFloatingPointIeee754<T>
        {
            for (var i = 0; i < A.Rows; i++)
            {
                var rowC = C.Row(i);
                rowC.Clear();
                for (var k = 0; k < A.Cols; k++)
                    TensorPrimitives.MultiplyAdd(B.Row(k), A[i, k], rowC, rowC);
            }
        }

        public static void MatMulAdd<T>(FastMatrixView<T> A, FastMatrixView<T> B, FastMatrixView<T> C)
            where T : struct, IFloatingPointIeee754<T>
        {
            for (var i = 0; i < A.Rows; i++)
            {
                var rowC = C.Row(i);
                for (var k = 0; k < A.Cols; k++)
                    TensorPrimitives.MultiplyAdd(B.Row(k), A[i, k], rowC, rowC);
            }
        }

        public static FastMatrix<float> MatMulRaw(FastMatrixView<float> A, FastMatrixView<float> B)
        {
            var C = new FastMatrix<float>(A.Rows, B.Cols);
            MatMul(A, B, C.AsView());
            return C;
        }

        // ====================================================================
        // 2. FORWARD OPERATIONS (Tape)
        // ====================================================================

        public static AutogradNode AddBias(AutogradNode input, AutogradNode bias)
        {
            var resultData = new FastMatrix<float>(input.Data.Rows, input.Data.Cols);
            var broadcastedBias = BroadcastRowVector(bias.Data.AsSpan(), input.Data.Rows);
            Add(input.Data.AsView(), broadcastedBias, resultData.AsView());

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || bias.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.AddBias, outputNode, input, bias);

            return outputNode;
        }

        public static AutogradNode MatMul(AutogradNode left, AutogradNode right)
        {
            var resultData = new FastMatrix<float>(left.Data.Rows, right.Data.Cols);
            MatMul(left.Data.AsView(), right.Data.AsView(), resultData.AsView());

            var outputNode = new AutogradNode(resultData, left.RequiresGrad || right.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.MatMul, outputNode, left, right);

            return outputNode;
        }

        public static AutogradNode Linear(AutogradNode input, AutogradNode weights, AutogradNode bias)
        {
            var mm = MatMul(input, weights);
            return AddBias(mm, bias);
        }

        public static AutogradNode ReLU(AutogradNode input)
        {
            var resultData = new FastMatrix<float>(input.Data.Rows, input.Data.Cols);
            TensorPrimitives.Max(input.Data.AsReadOnlySpan(), 0f, resultData.AsSpan());

            var outputNode = new AutogradNode(resultData, input.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.ReLU, outputNode, input);

            return outputNode;
        }

        public static AutogradNode Dropout(AutogradNode input, float p, bool isTraining)
        {
            if (!isTraining) return input;

            var resultData = new FastMatrix<float>(input.Data.Rows, input.Data.Cols);
            var maskTensor = new AutogradNode(new FastMatrix<float>(input.Data.Rows, input.Data.Cols), requiresGrad: false);
            var scale = 1.0f / (1.0f - p);

            var inSpan = input.Data.AsReadOnlySpan();
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
            using var diffBuffer = new FastBuffer<float>(n); // Zero-Alloc: ArrayPool pod spodem
            var diffSpan = diffBuffer.AsSpan();

            TensorPrimitives.Subtract(prediction.Data.AsReadOnlySpan(), target.Data.AsReadOnlySpan(), diffSpan);
            var finalLoss = TensorPrimitives.SumOfSquares((ReadOnlySpan<float>)diffSpan) / n;

            var resultMat = new FastMatrix<float>(1, 1);
            resultMat[0, 0] = finalLoss;

            var outputNode = new AutogradNode(resultMat, prediction.RequiresGrad || target.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.MSELoss, outputNode, prediction, target);

            return outputNode;
        }

        public static AutogradNode SoftmaxCrossEntropy(AutogradNode logits, AutogradNode target)
        {
            var rows = logits.Data.Rows;
            var cols = logits.Data.Cols;
            var totalLoss = 0f;

            // Żadnego "new double[]". Używamy wynajętego bufora z puli.
            using var pRowBuf = new FastBuffer<float>(cols);
            var pRow = pRowBuf.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var row = logits.Data.ReadOnlyRow(r);
                var tRow = target.Data.ReadOnlyRow(r);

                TensorPrimitives.SoftMax(row, pRow);

                for (var c = 0; c < cols; c++)
                {
                    if (tRow[c] > 0.5) totalLoss -= MathF.Log(pRow[c] + 1e-15f);
                }
            }

            var resData = new FastMatrix<float>(1, 1);
            resData[0, 0] = totalLoss / rows;

            var outputNode = new AutogradNode(resData, logits.RequiresGrad);
            if (logits.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.SoftmaxCrossEntropy, outputNode, logits, target);

            return outputNode;
        }

        public static AutogradNode Conv2D(AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1;
            var batchSize = input.Data.Rows;
            var kSquareInC = k * k * inC;

            var resultData = new FastMatrix<float>(batchSize, outC * outH * outW);

            Parallel.For(0, batchSize, n =>
            {
                // Zero-Alloc wewnątrz wątku
                using var colData = new FastMatrix<float>(kSquareInC, outH * outW);

                Im2Col(input.Data.ReadOnlyRow(n), inC, h, w, k, 1, 0, colData.AsSpan());

                using var batchResult = MatMulRaw(weights.Data.AsView(), colData.AsView());

                // Kopiowanie wyniku splotu dla danego obrazu do głównej macierzy
                batchResult.AsSpan().CopyTo(resultData.Row(n));
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
            var batchSize = input.Data.Rows;

            var resultData = new FastMatrix<float>(batchSize, channels * outputH * outputW);
            var maxIndicesTensor = new AutogradNode(new FastMatrix<float>(batchSize, channels * outputH * outputW), requiresGrad: false);
            var maxIndices = maxIndicesTensor.Data;

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
                                    var val = input.Data[n, currentIdx];

                                    if (val > maxVal)
                                    {
                                        maxVal = val; maxIdx = currentIdx;
                                    }
                                }
                            }
                            var outIdx = c * (outputH * outputW) + oh * outputW + ow;
                            resultData[n, outIdx] = maxVal;
                            maxIndices[n, outIdx] = maxIdx;
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
            var batchSize = input.Data.Rows;
            var spatialSize = h * w;
            var outputData = new FastMatrix<float>(batchSize, channels);
            var invSpatialSize = 1f / spatialSize;

            for (var b = 0; b < batchSize; b++)
            {
                var row = input.Data.ReadOnlyRow(b);
                for (var c = 0; c < channels; c++)
                {
                    var offset = c * spatialSize;

                    // Wycinamy fragment pamięci (Span) dla danego kanału i sumujemy SIMD
                    var spatialSpan = row.Slice(offset, spatialSize);
                    var sum = TensorPrimitives.Sum(spatialSpan);

                    outputData[b, c] = sum * invSpatialSize;
                }
            }

            var outputNode = new AutogradNode(outputData, input.RequiresGrad);

            if (outputNode.RequiresGrad)
            {
                ComputationGraph.Active.Record(OpCode.GlobalAveragePool2D, outputNode, input, null,
                    i0: channels, i1: h, i2: w);
            }

            return outputNode;
        }

        public static AutogradNode BatchNorm1D(
            AutogradNode input,
            AutogradNode gamma,
            AutogradNode beta,
            FastMatrix<float> runningMean,
            FastMatrix<float> runningVar,
            float momentum,
            float eps,
            bool isTraining)
        {
            var N = input.Data.Rows;
            var C = input.Data.Cols;
            var resultData = new FastMatrix<float>(N, C);

            if (!isTraining)
            {
                using var scaleBuf = new FastBuffer<float>(C);
                using var shiftBuf = new FastBuffer<float>(C);
                var scale = scaleBuf.AsSpan();
                var shift = shiftBuf.AsSpan();

                var gSpan = gamma.Data.AsReadOnlySpan();
                var bSpan = beta.Data.AsReadOnlySpan();
                var rmSpan = runningMean.AsReadOnlySpan();
                var rvSpan = runningVar.AsReadOnlySpan();

                for (var j = 0; j < C; j++)
                {
                    var invStd = 1f / MathF.Sqrt(rvSpan[j] + eps);
                    scale[j] = gSpan[j] * invStd;
                    shift[j] = bSpan[j] - (rmSpan[j] * scale[j]);
                }

                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.MultiplyAdd(
                        input.Data.ReadOnlyRow(i),
                        scale,
                        shift,
                        resultData.Row(i)
                    );
                }

                return new AutogradNode(resultData, requiresGrad: false);
            }

            using var batchMeanBuf = new FastBuffer<float>(C);
            using var batchVarBuf = new FastBuffer<float>(C);
            var batchMean = batchMeanBuf.AsSpan();
            var batchVar = batchVarBuf.AsSpan();

            var invStdTensor = new AutogradNode(new FastMatrix<float>(1, C), requiresGrad: false);
            var xHatTensor = new AutogradNode(new FastMatrix<float>(N, C), requiresGrad: false);
            var invStdVec = invStdTensor.Data.AsSpan();

            batchMean.Clear();
            for (var i = 0; i < N; i++) TensorPrimitives.Add(batchMean, input.Data.ReadOnlyRow(i), batchMean);
            TensorPrimitives.Multiply(batchMean, 1f / N, batchMean);

            batchVar.Clear();
            using var tempRowMat = new FastMatrix<float>(1, C);
            var tempRow = tempRowMat.AsSpan();
            for (var i = 0; i < N; i++)
            {
                TensorPrimitives.Subtract(input.Data.ReadOnlyRow(i), (ReadOnlySpan<float>)batchMean, tempRow);
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

            for (var i = 0; i < N; i++)
            {
                var rowIn = input.Data.ReadOnlyRow(i);
                var rowOut = resultData.Row(i);
                var rowXHat = xHatTensor.Data.Row(i);
                var gSpanLocal = gamma.Data.AsReadOnlySpan();
                var bSpanLocal = beta.Data.AsReadOnlySpan();

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
        // 3. BACKWARD OPERATIONS
        // ====================================================================

        public static void AddBackward(AutogradNode left, AutogradNode right, AutogradNode output)
        {
            if (left.RequiresGrad)
            {
                TensorPrimitives.Add(left.Grad.AsReadOnlySpan(), output.Grad.AsReadOnlySpan(), left.Grad.AsSpan());
            }

            if (right.RequiresGrad)
            {
                TensorPrimitives.Add(right.Grad.AsReadOnlySpan(), output.Grad.AsReadOnlySpan(), right.Grad.AsSpan());
            }
        }

        public static void AddBiasBackward(AutogradNode input, AutogradNode bias, AutogradNode output)
        {
            if (input.RequiresGrad)
                TensorPrimitives.Add(input.Grad.AsReadOnlySpan(), output.Grad.AsReadOnlySpan(), input.Grad.AsSpan());

            if (bias.RequiresGrad)
            {
                var bGrad = bias.Grad.AsSpan();
                for (var r = 0; r < output.Grad.Rows; r++)
                    TensorPrimitives.Add(bGrad, output.Grad.ReadOnlyRow(r), bGrad);
            }
        }

        public static void MatMulBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                using var bT = b.Data.Transpose().ToContiguousFastMatrix();
                MatMulAdd(output.Grad.AsView(), bT.AsView(), a.Grad.AsView());
            }
            if (b.RequiresGrad)
            {
                using var aT = a.Data.Transpose().ToContiguousFastMatrix();
                MatMulAdd(aT.AsView(), output.Grad.AsView(), b.Grad.AsView());
            }
        }

        public static void ReluBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad) return;
            var inSpan = input.Data.AsReadOnlySpan();
            var gradOut = output.Grad.AsReadOnlySpan();
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
            var gradOut = output.Grad.AsReadOnlySpan();
            var mSpan = mask.Data.AsReadOnlySpan();

            TensorPrimitives.MultiplyAdd(gradOut, mSpan, gradIn, gradIn);
        }

        public static void MSELossBackward(AutogradNode prediction, AutogradNode target, AutogradNode output)
        {
            var n = prediction.Data.Size;
            var m = (2f / n) * output.Grad[0, 0];

            if (prediction.RequiresGrad)
            {
                var pGrad = prediction.Grad.AsSpan();
                TensorPrimitives.MultiplyAdd(prediction.Data.AsReadOnlySpan(), m, pGrad, pGrad);
                TensorPrimitives.MultiplyAdd(target.Data.AsReadOnlySpan(), -m, pGrad, pGrad);
            }
            if (target.RequiresGrad)
            {
                var tGrad = target.Grad.AsSpan();
                TensorPrimitives.MultiplyAdd(prediction.Data.AsReadOnlySpan(), -m, tGrad, tGrad);
                TensorPrimitives.MultiplyAdd(target.Data.AsReadOnlySpan(), m, tGrad, tGrad);
            }
        }

        public static void SoftmaxCrossEntropyBackward(AutogradNode logits, AutogradNode target, AutogradNode output)
        {
            if (!logits.RequiresGrad) return;

            var rows = logits.Data.Rows;
            var cols = logits.Data.Cols;
            var scale = output.Grad.AsReadOnlySpan()[0] / rows;

            using var pRowBuf = new FastBuffer<float>(cols);
            using var diffBuf = new FastBuffer<float>(cols);

            var pRow = pRowBuf.AsSpan();
            var diff = diffBuf.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var lRow = logits.Data.ReadOnlyRow(r);
                var tRow = target.Data.ReadOnlyRow(r);
                var lGrad = logits.Grad.Row(r);

                TensorPrimitives.SoftMax(lRow, pRow);

                TensorPrimitives.Subtract(pRow, tRow, diff);
                TensorPrimitives.MultiplyAdd(diff, scale, lGrad, lGrad);
            }
        }

        public static void Conv2DBackward(AutogradNode input, AutogradNode weights, AutogradNode output, int inC, int outC, int h, int w, int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1;
            var batchSize = input.Data.Rows;
            var kSquareInC = k * k * inC;

            Parallel.For(0, batchSize, n =>
            {
                using var gn = new FastMatrix<float>(outC, outH * outW);
                output.Grad.ReadOnlyRow(n).CopyTo(gn.AsSpan());

                if (weights.RequiresGrad)
                {
                    using var colData = new FastMatrix<float>(kSquareInC, outH * outW);
                    Im2Col(input.Data.ReadOnlyRow(n), inC, h, w, k, 1, 0, colData.AsSpan());

                    using var colT = colData.AsView().Transpose().ToContiguousFastMatrix();
                    using var dW_batch = MatMulRaw(gn.AsView(), colT.AsView());

                    // LOCK GWARANTUJE BEZPIECZNĄ AKUMULACJĘ GRADIENTU WAG Z WIELU WĄTKÓW
                    lock (weights.Grad)
                    {
                        TensorPrimitives.Add(weights.Grad.AsReadOnlySpan(), dW_batch.AsReadOnlySpan(), weights.Grad.AsSpan());
                    }
                }

                if (input.RequiresGrad)
                {
                    using var dX_col = MatMulRaw(weights.Data.AsView().Transpose(), gn.AsView());
                    Col2Im(dX_col.AsReadOnlySpan(), inC, h, w, k, 1, 0, input.Grad.Row(n));
                }
            });
        }

        public static void MaxPool2DBackward(AutogradNode input, AutogradNode maxIndices, AutogradNode output)
        {
            var batchSize = input.Data.Rows;

            Parallel.For(0, batchSize, n =>
            {
                var gradOut = output.Grad.ReadOnlyRow(n);
                var gradIn = input.Grad.Row(n);
                var idxRow = maxIndices.Data.ReadOnlyRow(n);

                for (var i = 0; i < gradOut.Length; i++)
                {
                    var originalIdx = (int)idxRow[i];
                    gradIn[originalIdx] += gradOut[i];
                }
            });
        }

        public static void GlobalAvgPool2DBackward(AutogradNode input, AutogradNode output, int channels, int h, int w)
        {
            var batchSize = input.Data.Rows;
            var spatialSize = h * w;
            var invSpatialSize = 1f / spatialSize;

            for (var b = 0; b < batchSize; b++)
            {
                for (var c = 0; c < channels; c++)
                {
                    var gradOut = output.Grad[b, c];
                    var offset = c * spatialSize;

                    var gradInSpan = input.Grad.Row(b).Slice(offset, spatialSize);
                    TensorPrimitives.Add(gradInSpan, gradOut * invSpatialSize, gradInSpan);
                }
            }
        }

        public static void BatchNorm1DBackward(AutogradNode input, AutogradNode output, AutogradNode gamma, AutogradNode beta, AutogradNode xHatTensor, AutogradNode invStdTensor)
        {
            var N = input.Data.Rows;
            var C = input.Data.Cols;
            var gradOut = output.Grad;
            var xHatMat = xHatTensor.Data;

            using var dGammaBuf = new FastBuffer<float>(C);
            using var dBetaBuf = new FastBuffer<float>(C);

            var dGamma = dGammaBuf.AsSpan();
            var dBeta = dBetaBuf.AsSpan();

            for (var i = 0; i < N; i++)
            {
                var gradOutRow = gradOut.ReadOnlyRow(i);
                TensorPrimitives.MultiplyAdd(gradOutRow, xHatMat.ReadOnlyRow(i), dGamma, dGamma);
                TensorPrimitives.Add(dBeta, gradOutRow, dBeta);
            }

            if (gamma.RequiresGrad)
            {
                TensorPrimitives.Add(gamma.Grad.AsSpan(), dGamma, gamma.Grad.AsSpan());
            }

            if (beta.RequiresGrad)
            {
                TensorPrimitives.Add(beta.Grad.AsSpan(), dBeta, beta.Grad.AsSpan());
            }

            if (input.RequiresGrad)
            {
                using var factorBuf = new FastBuffer<float>(C);
                using var temp1Buf = new FastBuffer<float>(C);
                using var termBuf = new FastBuffer<float>(C);

                var factor = factorBuf.AsSpan();
                var temp1 = temp1Buf.AsSpan();
                var term = termBuf.AsSpan();

                TensorPrimitives.Multiply(gamma.Data.AsReadOnlySpan(), invStdTensor.Data.AsReadOnlySpan(), factor);
                TensorPrimitives.Multiply(factor, 1f / N, factor);

                for (var i = 0; i < N; i++)
                {
                    var gradInRow = input.Grad.Row(i);
                    var gradOutRow = gradOut.ReadOnlyRow(i);
                    var xHatRow = xHatMat.ReadOnlyRow(i);

                    TensorPrimitives.Multiply(gradOutRow, N, term);
                    TensorPrimitives.Subtract(term, dBeta, term);

                    TensorPrimitives.Multiply(xHatRow, dGamma, temp1);
                    TensorPrimitives.Subtract(term, temp1, term);

                    TensorPrimitives.MultiplyAdd(term, factor, gradInRow, gradInRow);
                }
            }
        }

        // ====================================================================
        // 4. CNN HELPERS (Im2Col / Col2Im)
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
                            for (var x = 0; x < outW; x++)
                            {
                                var j = x * stride - padding + kw;
                                var outIdx = rowOffset + y * outW + x;
                                if (i >= 0 && i < height && j >= 0 && j < width)
                                    output[outIdx] = input[c * channelSize + i * width + j];
                                else
                                    output[outIdx] = 0f;
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
                            for (var x = 0; x < outW; x++)
                            {
                                var j = x * stride - padding + kw;
                                var colIdx = rowOffset + y * outW + x;

                                if (i >= 0 && i < height && j >= 0 && j < width)
                                {
                                    gradInput[c * channelSize + i * width + j] += colData[colIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}