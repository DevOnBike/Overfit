using System.Numerics;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit
{
    /// <summary>
    /// Computational engine operating on views. 
    /// Implements allocation-free Broadcasting, parallel SIMD operations, and Autograd graph building.
    /// Memory leaks from ArrayPool exhaustion are strictly prevented via conditional disposal.
    /// </summary>
    public static class TensorMath
    {
        // ====================================================================
        // 1. BROADCASTING FACTORIES
        // ====================================================================

        /// <summary>
        /// Creates a virtual [targetRows x Cols] matrix from a 1D [Cols] vector.
        /// ZERO memory copying. Clones the row downwards in O(1) time.
        /// </summary>
        public static FastMatrixView<T> BroadcastRowVector<T>(Span<T> rowVector, int targetRows)
            where T : struct, IFloatingPointIeee754<T>
        {
            return new FastMatrixView<T>(
            data: rowVector,
            rows: targetRows,
            cols: rowVector.Length,
            rowStride: 0, // BROADCASTING MAGIC: Step down = stand still in memory
            colStride: 1,
            offset: 0);
        }

        // ====================================================================
        // 2. MATH ENGINE (SIMD Execution)
        // ====================================================================

        public static void Add<T>(FastMatrixView<T> left, FastMatrixView<T> right, FastMatrixView<T> result)
            where T : struct, IFloatingPointIeee754<T>
        {
            if (left.Rows != right.Rows || left.Cols != right.Cols || left.Rows != result.Rows || left.Cols != result.Cols)
                throw new ArgumentException("Shape mismatch. Views must have identical virtual dimensions.");

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
                throw new ArgumentException("MatMul requires contiguous rows in B (ColStride == 1).", nameof(B));

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
                throw new ArgumentException("MatMulAdd requires contiguous rows in B (ColStride == 1).", nameof(B));

            for (var i = 0; i < A.Rows; i++)
            {
                var rowC = C.Row(i);
                for (var k = 0; k < A.Cols; k++)
                {
                    var a_ik = A[i, k];
                    var rowB = B.Row(k);
                    TensorPrimitives.MultiplyAdd(rowB, a_ik, rowC, rowC); // Accumulates automatically
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
            for (var i = 0; i < A.Rows; i++)
            {
                var rowA = A.Row(i);
                var rowC = C.Row(i);

                for (var k = 0; k < A.Cols; k++)
                {
                    var valA = rowA[k];
                    var rowB = B.Row(k);
                    TensorPrimitives.MultiplyAdd(rowB, valA, rowC, rowC);
                }
            }
            return C;
        }

        // ====================================================================
        // 3. AUTOGRAD OPERATIONS
        // ====================================================================

        public static Tensor Add(Tensor left, Tensor right)
        {
            var resultData = new FastMatrix<double>(left.Data.Rows, left.Data.Cols);
            Add(left.Data.AsView(), right.Data.AsView(), resultData.AsView());

            Action<Tensor> backward = (resultNode) => {
                var gradC = resultNode.Grad.AsView();

                if (left.RequiresGrad) Add(left.Grad.AsView(), gradC, left.Grad.AsView());
                if (right.RequiresGrad) Add(right.Grad.AsView(), gradC, right.Grad.AsView());
            };

            return Tensor.CreateOperationResult(resultData, new List<Tensor>
            {
                left,
                right
            }, backward);
        }

        public static Tensor AddBias(Tensor input, Tensor bias)
        {
            if (bias.Data.Rows != 1 || bias.Data.Cols != input.Data.Cols)
                throw new ArgumentException("Bias must be a 1xD vector.");

            var resultData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);
            var broadcastedBias = BroadcastRowVector(bias.Data.AsSpan(), input.Data.Rows);
            Add(input.Data.AsView(), broadcastedBias, resultData.AsView());

            Action<Tensor> backward = (resultNode) => {
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

            return Tensor.CreateOperationResult(resultData, new List<Tensor>
            {
                input,
                bias
            }, backward);
        }

        public static Tensor MatMul(Tensor left, Tensor right)
        {
            var resultData = new FastMatrix<double>(left.Data.Rows, right.Data.Cols);
            MatMul(left.Data.AsView(), right.Data.AsView(), resultData.AsView());

            Action<Tensor> backward = (resultNode) => {
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

            return Tensor.CreateOperationResult(resultData, new List<Tensor>
            {
                left,
                right
            }, backward);
        }

        // ====================================================================
        // 4. ACTIVATION FUNCTIONS
        // ====================================================================

        public static Tensor ReLU(Tensor input)
        {
            var resultData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);
            TensorPrimitives.Max(input.Data.AsReadOnlySpan(), 0.0, resultData.AsSpan());

            Action<Tensor> backward = (resultNode) => {
                if (!input.RequiresGrad) return;

                var inSpan = input.Data.AsReadOnlySpan();
                var gradOutSpan = resultNode.Grad.AsReadOnlySpan();
                var gradInSpan = input.Grad.AsSpan();

                for (var i = 0; i < inSpan.Length; i++)
                {
                    if (inSpan[i] > 0) gradInSpan[i] += gradOutSpan[i];
                }
            };

            return Tensor.CreateOperationResult(resultData, new List<Tensor>
            {
                input
            }, backward);
        }

        public static Tensor Softmax(Tensor input)
        {
            var resData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);

            for (var r = 0; r < input.Data.Rows; r++)
            {
                TensorPrimitives.SoftMax(input.Data.Row(r), resData.Row(r));
            }

            Action<Tensor> backward = (resultNode) => {
                if (!input.RequiresGrad) return;

                var outGrad = resultNode.Grad;
                var outVal = resultNode.Data;
                var inGrad = input.Grad;

                for (var r = 0; r < outVal.Rows; r++)
                {
                    var gradRow = outGrad.Row(r);
                    var valRow = outVal.Row(r);
                    var targetGradRow = inGrad.Row(r);

                    var dot = TensorPrimitives.Dot(gradRow, valRow);

                    for (var c = 0; c < valRow.Length; c++)
                    {
                        targetGradRow[c] += valRow[c] * (gradRow[c] - dot);
                    }
                }
            };

            return Tensor.CreateOperationResult(resData, new List<Tensor>
            {
                input
            }, backward);
        }

        public static Tensor Dropout(Tensor input, double p, bool isTraining)
        {
            if (!isTraining) return input;

            var resultData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);
            var mask = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);
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

            bool needsGrad = input.RequiresGrad;

            // IMMEDIATE DISPOSAL IF NO GRADIENT REQUIRED TO PREVENT LEAKS
            if (!needsGrad) mask.Dispose();

            Action<Tensor> backward = (resultNode) => {
                if (!needsGrad) return;

                var gradOut = resultNode.Grad.AsReadOnlySpan();
                var gradIn = input.Grad.AsSpan();
                var mSpan = mask.AsReadOnlySpan();

                for (var i = 0; i < gradIn.Length; i++)
                {
                    gradIn[i] += gradOut[i] * mSpan[i];
                }

                mask.Dispose();
            };

            return Tensor.CreateOperationResult(resultData, new List<Tensor>
            {
                input
            }, backward);
        }

        // ====================================================================
        // 5. LOSS FUNCTIONS
        // ====================================================================

        public static Tensor MSE(Tensor predictions, Tensor targets)
        {
            if (predictions.Data.Rows != targets.Data.Rows || predictions.Data.Cols != targets.Data.Cols)
                throw new ArgumentException("Shape mismatch between predictions and targets in MSE.");

            var n = predictions.Data.Rows * predictions.Data.Cols;
            var resultData = new FastMatrix<double>(1, 1);

            var predSpanForward = predictions.Data.AsReadOnlySpan();
            var targetSpanForward = targets.Data.AsReadOnlySpan();

            var dist = TensorPrimitives.Distance(predSpanForward, targetSpanForward);
            resultData[0, 0] = (dist * dist) / n;

            Action<Tensor> backward = (resultNode) => {
                var gradC = resultNode.Grad[0, 0];
                var factor = (2.0 / n) * gradC;

                var pData = predictions.Data.AsReadOnlySpan();
                var tData = targets.Data.AsReadOnlySpan();

                if (predictions.RequiresGrad)
                {
                    var pGrad = predictions.Grad.AsSpan();
                    for (var i = 0; i < pData.Length; i++) pGrad[i] += factor * (pData[i] - tData[i]);
                }

                if (targets.RequiresGrad)
                {
                    var tGrad = targets.Grad.AsSpan();
                    for (var i = 0; i < tData.Length; i++) tGrad[i] += factor * (tData[i] - pData[i]);
                }
            };

            return Tensor.CreateOperationResult(resultData, new List<Tensor>
            {
                predictions,
                targets
            }, backward);
        }

        public static Tensor CrossEntropy(Tensor prediction, Tensor target)
        {
            var resData = new FastMatrix<double>(1, 1);
            double loss = 0;
            var epsilon = 1e-15;

            for (var r = 0; r < prediction.Data.Rows; r++)
            {
                for (var c = 0; c < prediction.Data.Cols; c++)
                {
                    if (target.Data[r, c] > 0.5)
                        loss -= Math.Log(prediction.Data[r, c] + epsilon);
                }
            }
            resData[0, 0] = loss / prediction.Data.Rows;

            Action<Tensor> backward = (resultNode) => {
                if (!prediction.RequiresGrad) return;

                var scale = resultNode.Grad[0, 0] / prediction.Data.Rows;
                for (var r = 0; r < prediction.Data.Rows; r++)
                {
                    for (var c = 0; c < prediction.Data.Cols; c++)
                    {
                        prediction.Grad[r, c] += (prediction.Data[r, c] - target.Data[r, c]) * scale;
                    }
                }
            };

            return Tensor.CreateOperationResult(resData, new List<Tensor>
            {
                prediction,
                target
            }, backward);
        }

        public static Tensor SoftmaxCrossEntropy(Tensor logits, Tensor target)
        {
            var resData = new FastMatrix<double>(1, 1);
            var probs = new FastMatrix<double>(logits.Data.Rows, logits.Data.Cols);
            double totalLoss = 0;

            for (var r = 0; r < logits.Data.Rows; r++)
            {
                var row = logits.Data.Row(r);
                var pRow = probs.Row(r);

                TensorPrimitives.SoftMax(row, pRow);

                for (var c = 0; c < row.Length; c++)
                {
                    if (target.Data[r, c] > 0.5)
                        totalLoss -= Math.Log(pRow[c] + 1e-15);
                }
            }

            resData[0, 0] = totalLoss / logits.Data.Rows;

            bool needsGrad = logits.RequiresGrad;

            // IMMEDIATE DISPOSAL IF NO GRADIENT REQUIRED TO PREVENT LEAKS
            if (!needsGrad) probs.Dispose();

            Action<Tensor> backward = (resultNode) => {
                if (!needsGrad) return;

                var scale = resultNode.Grad[0, 0] / logits.Data.Rows;

                for (var r = 0; r < logits.Data.Rows; r++)
                {
                    var lGrad = logits.Grad.Row(r);
                    var pRow = probs.ReadOnlyRow(r);
                    var tRow = target.Data.ReadOnlyRow(r);

                    for (var c = 0; c < lGrad.Length; c++)
                    {
                        lGrad[c] += (pRow[c] - tRow[c]) * scale;
                    }
                }

                probs.Dispose();
            };

            return Tensor.CreateOperationResult(resData, new List<Tensor>
            {
                logits,
                target
            }, backward);
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
                                    output[outIdx] = 0;
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

        public static Tensor MaxPool2D(Tensor input, int channels, int inputH, int inputW, int poolSize)
        {
            var outputH = inputH / poolSize;
            var outputW = inputW / poolSize;
            var batchSize = input.Data.Rows;

            var resultData = new FastMatrix<double>(batchSize, channels * outputH * outputW);
            var maxIndices = new int[batchSize, channels * outputH * outputW];

            for (var n = 0; n < batchSize; n++)
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
                            maxIndices[n, outIdx] = maxIdx;
                        }
                    }
                }
            }

            Action<Tensor> backward = (resultNode) => {
                if (!input.RequiresGrad) return;

                for (var n = 0; n < batchSize; n++)
                {
                    var gradOut = resultNode.Grad.Row(n);
                    var gradIn = input.Grad.Row(n);

                    for (var i = 0; i < gradOut.Length; i++)
                    {
                        var originalIdx = maxIndices[n, i];
                        gradIn[originalIdx] += gradOut[i];
                    }
                }
            };

            return Tensor.CreateOperationResult(resultData, new List<Tensor>
            {
                input
            }, backward);
        }

        public static Tensor Conv2D(Tensor input, Tensor weights, int inC, int outC, int h, int w, int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1;
            var batchSize = input.Data.Rows;
            var kSquareInC = k * k * inC;

            var colMatrices = new FastMatrix<double>[batchSize];
            var resultData = new FastMatrix<double>(batchSize, outC * outH * outW);

            for (var n = 0; n < batchSize; n++)
            {
                colMatrices[n] = new FastMatrix<double>(kSquareInC, outH * outW);
                Im2Col(input.Data.Row(n), inC, h, w, k, 1, 0, colMatrices[n].AsSpan());

                using var batchResult = MatMulRaw(weights.Data.AsView(), colMatrices[n].AsView());
                batchResult.AsSpan().CopyTo(resultData.Row(n));
            }

            bool needsGrad = weights.RequiresGrad || input.RequiresGrad;

            // IMMEDIATE DISPOSAL IF NO GRADIENT REQUIRED TO PREVENT LEAKS
            if (!needsGrad)
            {
                for (var n = 0; n < batchSize; n++) colMatrices[n].Dispose();
            }

            Action<Tensor> backward = (resultNode) => {
                if (!needsGrad) return;

                var gradOutput = resultNode.Grad;

                for (var n = 0; n < batchSize; n++)
                {
                    using var gn = new FastMatrix<double>(outC, outH * outW);
                    gradOutput.Row(n).CopyTo(gn.AsSpan());

                    if (weights.RequiresGrad)
                    {
                        using var colT = colMatrices[n].AsView().Transpose().ToContiguousFastMatrix();
                        using var dW_batch = MatMulRaw(gn.AsView(), colT.AsView());
                        TensorPrimitives.Add(weights.Grad.AsSpan(), dW_batch.AsSpan(), weights.Grad.AsSpan());
                    }

                    if (input.RequiresGrad)
                    {
                        var weightsT = weights.Data.AsView().Transpose();
                        using var dX_col = MatMulRaw(weightsT, gn.AsView());
                        Col2Im(dX_col.AsSpan(), inC, h, w, k, 1, 0, input.Grad.Row(n));
                    }

                    colMatrices[n].Dispose();
                }
            };

            return Tensor.CreateOperationResult(resultData, new List<Tensor>
            {
                input,
                weights
            }, backward);
        }

        // ====================================================================
        // 7. UTILS
        // ====================================================================

        public static double GetMax(ReadOnlySpan<double> span)
        {
            var max = span[0];
            for (var i = 1; i < span.Length; i++)
            {
                if (span[i] > max) max = span[i];
            }
            return max;
        }
    }
}