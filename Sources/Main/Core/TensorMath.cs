using System.Numerics;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Multi-threaded computational engine. 
    /// Features: SIMD via TensorPrimitives, O(1) Broadcasting, 
    /// CPU Multi-threading (Parallel.For), and ArrayPool safety.
    /// Operates on high-precision FP64 (double).
    /// </summary>
    public static class TensorMath
    {
        // ====================================================================
        // 1. BROADCASTING FACTORIES
        // ====================================================================

        public static FastMatrixView<T> BroadcastRowVector<T>(Span<T> rowVector, int targetRows)
            where T : struct, IFloatingPointIeee754<T>
        {
            return new FastMatrixView<T>(
                data: rowVector,
                rows: targetRows,
                cols: rowVector.Length,
                rowStride: 0,
                colStride: 1,
                offset: 0);
        }

        // ====================================================================
        // 2. CORE MATH ENGINE
        // ====================================================================

        public static void Add<T>(FastMatrixView<T> left, FastMatrixView<T> right, FastMatrixView<T> result)
            where T : struct, IFloatingPointIeee754<T>
        {
            if (left.Rows != right.Rows || left.Cols != right.Cols || left.Rows != result.Rows || left.Cols != result.Cols)
                throw new ArgumentException("Shape mismatch.");

            for (var i = 0; i < result.Rows; i++)
            {
                TensorPrimitives.Add(left.Row(i), right.Row(i), result.Row(i));
            }
        }

        public static void MatMul<T>(FastMatrixView<T> A, FastMatrixView<T> B, FastMatrixView<T> C)
            where T : struct, IFloatingPointIeee754<T>
        {
            if (A.Cols != B.Rows || A.Rows != C.Rows || B.Cols != C.Cols)
                throw new ArgumentException("Shape mismatch in MatMul.");

            if (B.ColStride != 1)
                throw new ArgumentException("MatMul requires contiguous rows in B.", nameof(B));

            for (var i = 0; i < A.Rows; i++)
            {
                var rowC = C.Row(i);
                rowC.Clear();

                for (var k = 0; k < A.Cols; k++)
                {
                    var a_ik = A[i, k];
                    var rowB = B.Row(k);
                    TensorPrimitives.MultiplyAdd(rowB, a_ik, rowC, rowC);
                }
            }
        }

        public static void MatMulAdd<T>(FastMatrixView<T> A, FastMatrixView<T> B, FastMatrixView<T> C)
            where T : struct, IFloatingPointIeee754<T>
        {
            if (A.Cols != B.Rows || A.Rows != C.Rows || B.Cols != C.Cols)
                throw new ArgumentException("Shape mismatch in MatMulAdd.");

            if (B.ColStride != 1)
                throw new ArgumentException("MatMulAdd requires contiguous rows in B.", nameof(B));

            for (var i = 0; i < A.Rows; i++)
            {
                var rowC = C.Row(i);

                for (var k = 0; k < A.Cols; k++)
                {
                    var a_ik = A[i, k];
                    var rowB = B.Row(k);
                    TensorPrimitives.MultiplyAdd(rowB, a_ik, rowC, rowC);
                }
            }
        }

        public static FastMatrix<double> MatMulRaw(FastMatrixView<double> A, FastMatrixView<double> B)
        {
            var C = new FastMatrix<double>(A.Rows, B.Cols);
            MatMul(A, B, C.AsView());
            return C;
        }

        public static FastMatrix<double> MatMulRaw(FastMatrix<double> A, FastMatrix<double> B)
        {
            if (A.Cols != B.Rows)
                throw new ArgumentException($"Shape mismatch: {A.Cols} != {B.Rows}");

            var C = new FastMatrix<double>(A.Rows, B.Cols);

            Parallel.For(0, A.Rows, i =>
            {
                var rowA = A.Row(i);
                var rowC = C.Row(i);

                for (var k = 0; k < A.Cols; k++)
                {
                    var valA = rowA[k];
                    var rowB = B.Row(k);
                    TensorPrimitives.MultiplyAdd(rowB, valA, rowC, rowC);
                }
            });
            return C;
        }

        // ====================================================================
        // 3. AUTOGRAD OPERATIONS
        // ====================================================================

        public static AutogradNode Add(AutogradNode left, AutogradNode right)
        {
            var resultData = new FastMatrix<double>(left.Data.Rows, left.Data.Cols);
            Add(left.Data.AsView(), right.Data.AsView(), resultData.AsView());

            Action<AutogradNode> backward = (resultNode) =>
            {
                var gradC = resultNode.Grad.AsView();
                if (left.RequiresGrad) Add(left.Grad.AsView(), gradC, left.Grad.AsView());
                if (right.RequiresGrad) Add(right.Grad.AsView(), gradC, right.Grad.AsView());
            };

            return AutogradNode.CreateOperationResult(resultData, [left, right], backward);
        }

        public static AutogradNode AddBias(AutogradNode input, AutogradNode bias)
        {
            var resultData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);
            var broadcastedBias = BroadcastRowVector(bias.Data.AsSpan(), input.Data.Rows);
            Add(input.Data.AsView(), broadcastedBias, resultData.AsView());

            Action<AutogradNode> backward = (resultNode) =>
            {
                var gradC = resultNode.Grad.AsView();

                if (input.RequiresGrad)
                    Add(input.Grad.AsView(), gradC, input.Grad.AsView());

                if (bias.RequiresGrad)
                {
                    var biasGradSpan = bias.Grad.AsSpan();
                    for (var r = 0; r < resultNode.Grad.Rows; r++)
                    {
                        var rowGrad = resultNode.Grad.Row(r);
                        TensorPrimitives.Add(biasGradSpan, rowGrad, biasGradSpan);
                    }
                }
            };

            return AutogradNode.CreateOperationResult(resultData, [input, bias], backward);
        }

        public static AutogradNode MatMul(AutogradNode left, AutogradNode right)
        {
            var resultData = new FastMatrix<double>(left.Data.Rows, right.Data.Cols);
            MatMul(left.Data.AsView(), right.Data.AsView(), resultData.AsView());

            Action<AutogradNode> backward = (resultNode) =>
            {
                var gradC = resultNode.Grad.AsView();

                if (left.RequiresGrad)
                {
                    using var rightT = right.Data.Transpose().ToContiguousFastMatrix();
                    MatMulAdd(gradC, rightT.AsView(), left.Grad.AsView());
                }

                if (right.RequiresGrad)
                {
                    MatMulAdd(left.Data.Transpose(), gradC, right.Grad.AsView());
                }
            };

            return AutogradNode.CreateOperationResult(resultData, [left, right], backward);
        }

        // ====================================================================
        // 4. ACTIVATION FUNCTIONS
        // ====================================================================

        public static AutogradNode ReLU(AutogradNode input)
        {
            var resultData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);
            TensorPrimitives.Max(input.Data.AsReadOnlySpan(), 0.0, resultData.AsSpan());

            Action<AutogradNode> backward = (resultNode) =>
            {
                if (!input.RequiresGrad) return;

                var inSpan = input.Data.AsReadOnlySpan();
                var gradOutSpan = resultNode.Grad.AsReadOnlySpan();
                var gradInSpan = input.Grad.AsSpan();

                for (var i = 0; i < inSpan.Length; i++)
                {
                    if (inSpan[i] > 0) gradInSpan[i] += gradOutSpan[i];
                }
            };

            return AutogradNode.CreateOperationResult(resultData, [input], backward);
        }

        public static AutogradNode Dropout(AutogradNode input, double p, bool isTraining)
        {
            if (!isTraining) return input;

            var resultData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);

            // WĘZEŁ-WIDMO
            var maskTensor = new AutogradNode(new FastMatrix<double>(input.Data.Rows, input.Data.Cols), requiresGrad: false);
            var mask = maskTensor.Data;
            var scale = 1.0 / (1.0 - p);

            var inSpan = input.Data.AsReadOnlySpan();
            var resSpan = resultData.AsSpan();
            var maskSpan = mask.AsSpan();

            for (var i = 0; i < inSpan.Length; i++)
            {
                if (Random.Shared.NextDouble() > p)
                {
                    maskSpan[i] = scale;
                    resSpan[i] = inSpan[i] * scale;
                }
                else
                {
                    maskSpan[i] = 0.0;
                    resSpan[i] = 0.0;
                }
            }

            var needsGrad = input.RequiresGrad;
            var deps = new List<AutogradNode> { input };

            if (needsGrad)
            {
                deps.Add(maskTensor);
            }
            else
            {
                maskTensor.Dispose();
            }

            Action<AutogradNode> backward = (resultNode) =>
            {
                var gradOut = resultNode.Grad.AsReadOnlySpan();
                var gradIn = input.Grad.AsSpan();
                var mSpan = mask.AsReadOnlySpan();

                for (var i = 0; i < gradIn.Length; i++)
                {
                    gradIn[i] += gradOut[i] * mSpan[i];
                }
            };

            return AutogradNode.CreateOperationResult(resultData, deps, backward);
        }

        // ====================================================================
        // 5. LOSS FUNCTIONS
        // ====================================================================

        public static AutogradNode MSE(AutogradNode predictions, AutogradNode targets)
        {
            if (predictions.Data.Rows != targets.Data.Rows || predictions.Data.Cols != targets.Data.Cols)
                throw new ArgumentException("Shape mismatch between predictions and targets in MSE.");

            var n = predictions.Data.Rows * predictions.Data.Cols;
            var resultData = new FastMatrix<double>(1, 1);

            var predSpanForward = predictions.Data.AsReadOnlySpan();
            var targetSpanForward = targets.Data.AsReadOnlySpan();

            var dist = TensorPrimitives.Distance(predSpanForward, targetSpanForward);
            resultData[0, 0] = (dist * dist) / n;

            Action<AutogradNode> backward = (resultNode) =>
            {
                var gradC = resultNode.Grad[0, 0];
                var factor = (2.0 / n) * gradC;

                var pData = predictions.Data.AsReadOnlySpan();
                var tData = targets.Data.AsReadOnlySpan();

                if (predictions.RequiresGrad)
                {
                    var pGrad = predictions.Grad.AsSpan();
                    for (var i = 0; i < pData.Length; i++)
                        pGrad[i] += factor * (pData[i] - tData[i]);
                }

                if (targets.RequiresGrad)
                {
                    var tGrad = targets.Grad.AsSpan();
                    for (var i = 0; i < tData.Length; i++)
                        tGrad[i] += factor * (tData[i] - pData[i]);
                }
            };

            return AutogradNode.CreateOperationResult(resultData, [predictions, targets], backward);
        }

        public static AutogradNode SoftmaxCrossEntropy(AutogradNode logits, AutogradNode target)
        {
            var resData = new FastMatrix<double>(1, 1);
            var probsTensor = new AutogradNode(new FastMatrix<double>(logits.Data.Rows, logits.Data.Cols), requiresGrad: false);
            var probs = probsTensor.Data;

            double totalLoss = 0;
            var lossLock = new object();

            Parallel.For(0, logits.Data.Rows, r =>
            {
                var row = logits.Data.Row(r);
                var pRow = probs.Row(r);

                TensorPrimitives.SoftMax(row, pRow);

                double localLoss = 0;
                for (var c = 0; c < row.Length; c++)
                {
                    if (target.Data[r, c] > 0.5)
                        localLoss -= Math.Log(pRow[c] + 1e-15);
                }

                lock (lossLock) { totalLoss += localLoss; }
            });

            resData[0, 0] = totalLoss / logits.Data.Rows;

            var needsGrad = logits.RequiresGrad;
            var deps = new List<AutogradNode> { logits, target };

            if (needsGrad) deps.Add(probsTensor);
            else probsTensor.Dispose();

            Action<AutogradNode> backward = (resultNode) =>
            {
                var scale = resultNode.Grad[0, 0] / logits.Data.Rows;

                Parallel.For(0, logits.Data.Rows, r =>
                {
                    var lGrad = logits.Grad.Row(r);
                    var pRow = probs.ReadOnlyRow(r);
                    var tRow = target.Data.ReadOnlyRow(r);

                    for (var c = 0; c < lGrad.Length; c++)
                    {
                        lGrad[c] += (pRow[c] - tRow[c]) * scale;
                    }
                });
            };

            return AutogradNode.CreateOperationResult(resData, deps, backward);
        }

        // ====================================================================
        // 6. CNN OPERATIONS
        // ====================================================================

        public static void Im2Col(
            ReadOnlySpan<double> input,
            int channels, int height, int width,
            int kSize, int stride, int padding,
            Span<double> output)
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
                                    output[outIdx] = 0.0;
                            }
                        }
                    }
                }
            }
        }

        public static void Col2Im(
            ReadOnlySpan<double> colData,
            int channels, int height, int width,
            int kSize, int stride, int padding,
            Span<double> gradInput)
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
                                    var inputIdx = c * channelSize + i * width + j;
                                    gradInput[inputIdx] += colData[colIdx];
                                }
                            }
                        }
                    }
                }
            }
        }

        public static AutogradNode MaxPool2D(AutogradNode input, int channels, int inputH, int inputW, int poolSize)
        {
            var outputH = inputH / poolSize;
            var outputW = inputW / poolSize;
            var batchSize = input.Data.Rows;

            var resultData = new FastMatrix<double>(batchSize, channels * outputH * outputW);
            var maxIndicesTensor = new AutogradNode(new FastMatrix<double>(batchSize, channels * outputH * outputW), requiresGrad: false);
            var maxIndices = maxIndicesTensor.Data;

            Parallel.For(0, batchSize, n =>
            {
                for (var c = 0; c < channels; c++)
                {
                    for (var oh = 0; oh < outputH; oh++)
                    {
                        for (var ow = 0; ow < outputW; ow++)
                        {
                            var maxVal = double.MinValue;
                            var maxIdx = -1;

                            for (var ph = 0; ph < poolSize; ph++)
                            {
                                for (var pw = 0; pw < poolSize; pw++)
                                {
                                    var currentIdx = c * (inputH * inputW) + (oh * poolSize + ph) * inputW + (ow * poolSize + pw);
                                    var val = input.Data[n, currentIdx];
                                    if (val > maxVal)
                                    {
                                        maxVal = val;
                                        maxIdx = currentIdx;
                                    }
                                }
                            }
                            var outIdx = c * (outputH * outputW) + oh * outputW + ow;
                            resultData[n, outIdx] = maxVal;
                            maxIndices[n, outIdx] = maxIdx; // Rzutowanie int -> double
                        }
                    }
                }
            });

            var deps = new List<AutogradNode> { input };
            if (input.RequiresGrad) deps.Add(maxIndicesTensor);
            else maxIndicesTensor.Dispose();

            Action<AutogradNode> backward = (resultNode) =>
            {
                Parallel.For(0, batchSize, n =>
                {
                    var gradOut = resultNode.Grad.Row(n);
                    var gradIn = input.Grad.Row(n);

                    for (var i = 0; i < gradOut.Length; i++)
                    {
                        var originalIdx = (int)maxIndices[n, i]; // Rzutowanie double -> int
                        gradIn[originalIdx] += gradOut[i];
                    }
                });
            };

            return AutogradNode.CreateOperationResult(resultData, deps, backward);
        }

        public static AutogradNode Conv2D(AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1;
            var batchSize = input.Data.Rows;
            var kSquareInC = k * k * inC;

            var resultData = new FastMatrix<double>(batchSize, outC * outH * outW);

            // Tablica Węzłów-Widm, zamiast tablicy czystych FastMatrix
            var colTensors = new AutogradNode[batchSize];

            Parallel.For(0, batchSize, n =>
            {
                colTensors[n] = new AutogradNode(new FastMatrix<double>(kSquareInC, outH * outW), requiresGrad: false);
                Im2Col(input.Data.Row(n), inC, h, w, k, 1, 0, colTensors[n].Data.AsSpan());

                using var batchResult = MatMulRaw(weights.Data.AsView(), colTensors[n].Data.AsView());
                batchResult.AsSpan().CopyTo(resultData.Row(n));
            });

            var needsGrad = weights.RequiresGrad || input.RequiresGrad;
            var deps = new List<AutogradNode> { input, weights };

            if (needsGrad)
            {
                deps.AddRange(colTensors);
            }
            else
            {
                for (var n = 0; n < batchSize; n++) colTensors[n].Dispose();
            }

            Action<AutogradNode> backward = (resultNode) =>
            {
                var gradOutput = resultNode.Grad;

                for (var n = 0; n < batchSize; n++)
                {
                    using var gn = new FastMatrix<double>(outC, outH * outW);
                    gradOutput.Row(n).CopyTo(gn.AsSpan());

                    if (weights.RequiresGrad)
                    {
                        using var colT = colTensors[n].Data.AsView().Transpose().ToContiguousFastMatrix();
                        using var dW_batch = MatMulRaw(gn.AsView(), colT.AsView());
                        TensorPrimitives.Add(weights.Grad.AsSpan(), dW_batch.AsSpan(), weights.Grad.AsSpan());
                    }

                    if (input.RequiresGrad)
                    {
                        var weightsT = weights.Data.AsView().Transpose();
                        using var dX_col = MatMulRaw(weightsT, gn.AsView());
                        Col2Im(dX_col.AsSpan(), inC, h, w, k, 1, 0, input.Grad.Row(n));
                    }
                }
            };

            return AutogradNode.CreateOperationResult(resultData, deps, backward);
        }

        public static AutogradNode Linear(AutogradNode input, AutogradNode weights, AutogradNode bias)
        {
            var resultData = new FastMatrix<double>(input.Data.Rows, weights.Data.Cols);
            MatMul(input.Data.AsView(), weights.Data.AsView(), resultData.AsView());

            var broadcastedBias = BroadcastRowVector(bias.Data.AsSpan(), input.Data.Rows);
            Add(resultData.AsView(), broadcastedBias, resultData.AsView());

            Action<AutogradNode> backward = (resultNode) =>
            {
                var gradC = resultNode.Grad.AsView();

                if (input.RequiresGrad)
                {
                    using var weightsT = weights.Data.Transpose().ToContiguousFastMatrix();
                    MatMulAdd(gradC, weightsT.AsView(), input.Grad.AsView());
                }

                if (weights.RequiresGrad)
                {
                    MatMulAdd(input.Data.Transpose(), gradC, weights.Grad.AsView());
                }

                if (bias.RequiresGrad)
                {
                    var biasGradSpan = bias.Grad.AsSpan();
                    for (var r = 0; r < resultNode.Grad.Rows; r++)
                    {
                        var rowGrad = resultNode.Grad.Row(r);
                        TensorPrimitives.Add(biasGradSpan, rowGrad, biasGradSpan);
                    }
                }
            };

            return AutogradNode.CreateOperationResult(resultData, [input, weights, bias], backward);
        }

        // ====================================================================
        // 8. NORMALIZACJA (BATCH NORM)
        // ====================================================================

        public static AutogradNode BatchNorm1D(
            AutogradNode input,
            AutogradNode gamma,
            AutogradNode beta,
            FastMatrix<double> runningMean,
            FastMatrix<double> runningVar,
            double momentum,
            double eps,
            bool isTraining)
        {
            var N = input.Data.Rows;
            var C = input.Data.Cols;

            var resultData = new FastMatrix<double>(N, C);

            if (!isTraining)
            {
                // Inference: Parallel.For przelatuje po wierszach - idealny Cache Hit
                Parallel.For(0, N, i =>
                {
                    var rowIn = input.Data.ReadOnlyRow(i);
                    var rowOut = resultData.Row(i);

                    var gSpan = gamma.Data.AsReadOnlySpan();
                    var bSpan = beta.Data.AsReadOnlySpan();
                    var rmSpan = runningMean.AsReadOnlySpan();
                    var rvSpan = runningVar.AsReadOnlySpan();

                    for (var j = 0; j < C; j++)
                    {
                        var invStd = 1.0 / Math.Sqrt(rvSpan[j] + eps);
                        var xHat = (rowIn[j] - rmSpan[j]) * invStd;
                        rowOut[j] = gSpan[j] * xHat + bSpan[j];
                    }
                });

                return AutogradNode.CreateOperationResult(resultData, [input], (_) => { });
            }

            // --- TRENING ---
            using var batchMeanMat = new FastMatrix<double>(1, C);
            using var batchVarMat = new FastMatrix<double>(1, C);

            var invStdTensor = new AutogradNode(new FastMatrix<double>(1, C), requiresGrad: false);
            var xHatTensor = new AutogradNode(new FastMatrix<double>(N, C), requiresGrad: false);

            var batchMean = batchMeanMat.AsSpan();
            var batchVar = batchVarMat.AsSpan();
            var invStdVec = invStdTensor.Data.AsSpan();

            // 1. Średnia: Row-Major + Sprzętowe SIMD (100% Cache Friendly)
            batchMean.Clear();
            for (var i = 0; i < N; i++)
            {
                // Zamiast wchodzić w głąb kolumn, dodajemy wektorowo całe wiersze
                TensorPrimitives.Add(batchMean, input.Data.ReadOnlyRow(i), batchMean);
            }

            var invN = 1.0 / N;
            TensorPrimitives.Multiply(batchMean, invN, batchMean);

            // 2. Wariancja: Row-Major + Sprzętowe FMA (Fused Multiply-Add)
            batchVar.Clear();
            using var tempRowMat = new FastMatrix<double>(1, C);
            var tempRow = tempRowMat.AsSpan();
            var batchMeanRead = batchMeanMat.AsReadOnlySpan();

            for (var i = 0; i < N; i++)
            {
                // tempRow = input_row - batchMean
                TensorPrimitives.Subtract(input.Data.ReadOnlyRow(i), batchMeanRead, tempRow);

                // batchVar = (tempRow * tempRow) + batchVar (Wykonane w jednym cyklu CPU!)
                TensorPrimitives.MultiplyAdd(tempRow, tempRow, batchVar, batchVar);
            }
            TensorPrimitives.Multiply(batchVar, invN, batchVar);

            // 3. Statystyki EMA i InvStd
            var rmSpanLocal = runningMean.AsSpan();
            var rvSpanLocal = runningVar.AsSpan();

            for (var j = 0; j < C; j++)
            {
                rmSpanLocal[j] = (1 - momentum) * rmSpanLocal[j] + momentum * batchMean[j];
                rvSpanLocal[j] = (1 - momentum) * rvSpanLocal[j] + momentum * batchVar[j];

                invStdVec[j] = 1.0 / Math.Sqrt(batchVar[j] + eps);
            }

            // 4. Normalizacja i Skalowanie (Przelot wierszami)
            var xHatMat = xHatTensor.Data;
            Parallel.For(0, N, i =>
            {
                var rowIn = input.Data.ReadOnlyRow(i);
                var rowOut = resultData.Row(i);
                var rowXHat = xHatMat.Row(i);

                var bMeanRead = batchMeanMat.AsReadOnlySpan();
                var invStdRead = invStdTensor.Data.AsReadOnlySpan();
                var gSpanLocal = gamma.Data.AsReadOnlySpan();
                var bSpanLocal = beta.Data.AsReadOnlySpan();

                for (var j = 0; j < C; j++)
                {
                    var xHat = (rowIn[j] - bMeanRead[j]) * invStdRead[j];
                    rowXHat[j] = xHat;
                    rowOut[j] = gSpanLocal[j] * xHat + bSpanLocal[j];
                }
            });

            var needsGrad = input.RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad;
            var deps = new List<AutogradNode> { input, gamma, beta };

            if (needsGrad)
            {
                deps.Add(xHatTensor);
                deps.Add(invStdTensor);
            }
            else
            {
                xHatTensor.Dispose();
                invStdTensor.Dispose();
            }

            Action<AutogradNode> backward = (resultNode) =>
            {
                var gradOut = resultNode.Grad;

                using var dGammaMat = new FastMatrix<double>(1, C);
                using var dBetaMat = new FastMatrix<double>(1, C);

                var dGamma = dGammaMat.AsSpan();
                var dBeta = dBetaMat.AsSpan();

                // KROK A: Cache-friendly Backward dla wag (Row-Major + SIMD)
                for (var i = 0; i < N; i++)
                {
                    var gradOutRow = gradOut.ReadOnlyRow(i);
                    var xHatRow = xHatMat.ReadOnlyRow(i);

                    // dGamma += gradOutRow * xHatRow
                    TensorPrimitives.MultiplyAdd(gradOutRow, xHatRow, dGamma, dGamma);

                    // dBeta += gradOutRow
                    TensorPrimitives.Add(dBeta, gradOutRow, dBeta);
                }

                if (gamma.RequiresGrad)
                {
                    var gGrad = gamma.Grad.AsSpan();
                    TensorPrimitives.Add(gGrad, dGamma, gGrad);
                }

                if (beta.RequiresGrad)
                {
                    var bGrad = beta.Grad.AsSpan();
                    TensorPrimitives.Add(bGrad, dBeta, bGrad);
                }

                if (input.RequiresGrad)
                {
                    // Krok B: Równoległe, wierszowe cofanie błędu do wejścia
                    Parallel.For(0, N, i =>
                    {
                        var gradInRow = input.Grad.Row(i);
                        var gradOutRow = gradOut.ReadOnlyRow(i);
                        var xHatRow = xHatMat.ReadOnlyRow(i);

                        var gSpanLocal = gamma.Data.AsReadOnlySpan();
                        var invStdSpanLocal = invStdTensor.Data.AsReadOnlySpan();
                        var dGammaRead = dGammaMat.AsReadOnlySpan();
                        var dBetaRead = dBetaMat.AsReadOnlySpan();

                        for (var j = 0; j < C; j++)
                        {
                            var dx = (gSpanLocal[j] * invStdSpanLocal[j] / N) * (N * gradOutRow[j] - dBetaRead[j] - xHatRow[j] * dGammaRead[j]);
                            gradInRow[j] += dx;
                        }
                    });
                }
            };

            return AutogradNode.CreateOperationResult(resultData, deps, backward);
        }

        public static AutogradNode GlobalAveragePool2D(AutogradNode input, int channels, int h, int w)
        {
            var batchSize = input.Data.Rows;
            var spatialSize = h * w;
            var outputData = new FastMatrix<double>(batchSize, channels);

            Parallel.For(0, batchSize, b =>
            {
                for (var c = 0; c < channels; c++)
                {
                    double sum = 0;
                    var offset = c * spatialSize;

                    for (var i = 0; i < spatialSize; i++)
                    {
                        sum += input.Data[b, offset + i];
                    }
                    outputData[b, c] = sum / spatialSize;
                }
            });

            var result = new AutogradNode(outputData, input.RequiresGrad);

            if (input.RequiresGrad)
            {
                result._dependencies.Add(input);
                result._backwardAction = (node) =>
                {
                    var invSpatialSize = 1.0 / spatialSize;

                    Parallel.For(0, batchSize, b =>
                    {
                        for (var c = 0; c < channels; c++)
                        {
                            var gradOut = node.Grad[b, c];
                            var offset = c * spatialSize;

                            for (var i = 0; i < spatialSize; i++)
                            {
                                input.Grad[b, offset + i] += gradOut * invSpatialSize;
                            }
                        }
                    });
                };
            }

            return result;
        }
    }
}