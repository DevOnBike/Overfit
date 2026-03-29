using System.Numerics;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit
{
    /// <summary>
    /// Computational engine operating on views. 
    /// Implements allocation-free Broadcasting, parallel SIMD operations, and Autograd graph building.
    /// </summary>
    public static class TensorMath
    {
        // ====================================================================
        // 1. BROADCASTING FACTORIES (Zero-stride magic)
        // ====================================================================

        /// <summary>
        /// Creates a virtual [targetRows x Cols] matrix from a 1D [Cols] vector.
        /// ZERO memory copying. Clones the row downwards in O(1) time.
        /// Used in ML to add Bias to an entire Batch of activations.
        /// </summary>
        public static FastMatrixView<T> BroadcastRowVector<T>(Span<T> rowVector, int targetRows)
            where T : struct, IFloatingPointIeee754<T>
        {
            return new FastMatrixView<T>(
            data: rowVector,
            rows: targetRows,
            cols: rowVector.Length,
            rowStride: 0, // <--- BROADCASTING MAGIC: Step down = stand still in memory
            colStride: 1, // Step right = move 1 element in the vector
            offset: 0);
        }

        // ====================================================================
        // 2. MATH ENGINE (SIMD Execution)
        // ====================================================================

        /// <summary>
        /// Adds two views together: result = left + right.
        /// Powerful SIMD acceleration. Natively supports Broadcasting!
        /// </summary>
        public static void Add<T>(FastMatrixView<T> left, FastMatrixView<T> right, FastMatrixView<T> result)
            where T : struct, IFloatingPointIeee754<T>
        {
            if (left.Rows != right.Rows || left.Cols != right.Cols || left.Rows != result.Rows || left.Cols != result.Cols)
            {
                throw new ArgumentException("Shape mismatch. Views must have identical virtual dimensions.");
            }

            // We process the matrix row by row. 
            // Why not the whole thing at once? Because broadcasted or transposed views
            // are not one giant, contiguous 2D block of memory. BUT their individual rows ARE contiguous!
            for (var i = 0; i < result.Rows; i++)
            {
                // If 'right' is a broadcasted Bias vector (RowStride == 0),
                // then right.Row(i) for every 'i' returns EXACTLY THE SAME physical Span!
                // Your CPU will love this, as this small Span will never leave the L1 Cache.
                var leftRow = left.Row(i);
                var rightRow = right.Row(i);
                var resultRow = result.Row(i);

                // Fire up hardware AVX-512 on the extracted Spans
                TensorPrimitives.Add(leftRow, rightRow, resultRow);
            }
        }

        // ====================================================================
        // 3. GEMM (General Matrix Multiply)
        // ====================================================================

        /// <summary>
        /// Executes C = A * B.
        /// Uses the "Linear Combination of Rows" optimization (Cache-friendly) 
        /// and FMA (Fused Multiply-Add) hardware vector instructions.
        /// </summary>
        public static void MatMul<T>(FastMatrixView<T> A, FastMatrixView<T> B, FastMatrixView<T> C)
            where T : struct, IFloatingPointIeee754<T>
        {
            if (A.Cols != B.Rows || A.Rows != C.Rows || B.Cols != C.Cols)
            {
                throw new ArgumentException("Shape mismatch in MatMul.");
            }

            if (B.ColStride != 1)
            {
                throw new ArgumentException("MatMul requires contiguous rows in B (ColStride == 1). For transposed B, materialize the view first.", nameof(B));
            }

            // Outer loop goes over the rows of A and C
            for (var i = 0; i < A.Rows; i++)
            {
                var rowC = C.Row(i);

                // Zero out the result row before accumulation
                rowC.Clear();

                // Inner loop goes over columns of A (and rows of B)
                for (var k = 0; k < A.Cols; k++)
                {
                    // Fetch a single scalar from matrix A
                    var a_ik = A[i, k];

                    // Fetch a CONTIGUOUS row from matrix B
                    var rowB = B.Row(k);

                    // Fire up hardware AVX-512 FMA: rowC = (rowB * a_ik) + rowC
                    TensorPrimitives.MultiplyAdd(rowB, a_ik, rowC, rowC);
                }
            }
        }

        /// <summary>
        /// Executes C += A * B (Accumulating GEMM).
        /// Used strictly in backpropagation to accumulate gradients without overwriting them.
        /// </summary>
        public static void MatMulAdd<T>(FastMatrixView<T> A, FastMatrixView<T> B, FastMatrixView<T> C)
            where T : struct, IFloatingPointIeee754<T>
        {
            if (A.Cols != B.Rows || A.Rows != C.Rows || B.Cols != C.Cols)
            {
                throw new ArgumentException("Shape mismatch in MatMulAdd.");
            }

            if (B.ColStride != 1)
            {
                throw new ArgumentException("MatMulAdd requires contiguous rows in B (ColStride == 1). Materialize the view first.", nameof(B));
            }

            for (var i = 0; i < A.Rows; i++)
            {
                var rowC = C.Row(i);

                // NO CLEARING HERE! We accumulate on top of existing values in rowC.
                for (var k = 0; k < A.Cols; k++)
                {
                    var a_ik = A[i, k];
                    var rowB = B.Row(k);
                    TensorPrimitives.MultiplyAdd(rowB, a_ik, rowC, rowC);
                }
            }
        }

        // ====================================================================
        // 4. AUTOGRAD OPERATIONS
        // ====================================================================

        /// <summary>
        /// Vectorized addition of two Tensors WITH AUTOGRAD SUPPORT.
        /// </summary>
        public static Tensor Add(Tensor left, Tensor right)
        {
            // 1. FORWARD PASS
            var resultData = new FastMatrix<double>(left.Data.Rows, left.Data.Cols);
            Add(left.Data.AsView(), right.Data.AsView(), resultData.AsView());

            // 2. BACKWARD PASS REGISTRATION (Clean Architecture: node injects itself)
            Action<Tensor> backward = (resultNode) => {
                var gradC = resultNode.Grad.AsView();

                if (left.RequiresGrad)
                {
                    // Chain rule: gradA += gradC
                    Add(left.Grad.AsView(), gradC, left.Grad.AsView());
                }

                if (right.RequiresGrad)
                {
                    // Chain rule: gradB += gradC
                    Add(right.Grad.AsView(), gradC, right.Grad.AsView());
                }
            };

            // 3. Create the computation graph node
            return Tensor.CreateOperationResult(
            data: resultData,
            dependencies: new List<Tensor>
            {
                left,
                right
            },
            backwardAction: backward
            );
        }

        /// <summary>
        /// Vectorized GEMM of two Tensors WITH AUTOGRAD SUPPORT (C = A * B).
        /// </summary>
        public static Tensor MatMul(Tensor left, Tensor right)
        {
            // 1. FORWARD PASS
            var resultData = new FastMatrix<double>(left.Data.Rows, right.Data.Cols);
            MatMul(left.Data.AsView(), right.Data.AsView(), resultData.AsView());

            // 2. BACKWARD PASS REGISTRATION (Clean Architecture: node injects itself)
            Action<Tensor> backward = (resultNode) => {
                var gradC = resultNode.Grad.AsView();

                if (left.RequiresGrad)
                {
                    // Math: gradA += gradC * B^T
                    // Catch: B^T is on the right, so it MUST be contiguous for SIMD. We materialize it.
                    using var rightT = right.Data.Transpose().ToContiguousFastMatrix();
                    MatMulAdd(gradC, rightT.AsView(), left.Grad.AsView());
                }

                if (right.RequiresGrad)
                {
                    // Math: gradB += A^T * gradC
                    // No catch here! A^T is on the left, scalar reads are fine. gradC is already contiguous.
                    MatMulAdd(left.Data.Transpose(), gradC, right.Grad.AsView());
                }
            };

            // 3. Create the computation graph node
            return Tensor.CreateOperationResult(
            data: resultData,
            dependencies: new List<Tensor>
            {
                left,
                right
            },
            backwardAction: backward
            );
        }

        // ====================================================================
        // 5. ACTIVATION FUNCTIONS
        // ====================================================================

        /// <summary>
        /// Rectified Linear Unit (ReLU) activation function.
        /// Math: f(x) = max(0, x).
        /// SIMD-accelerated forward pass.
        /// </summary>
        public static Tensor ReLU(Tensor input)
        {
            var resultData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);

            // 1. FORWARD PASS: Hardware-accelerated max(0, x)
            TensorPrimitives.Max(input.Data.AsReadOnlySpan(), 0.0, resultData.AsSpan());

            // 2. BACKWARD PASS REGISTRATION
            Action<Tensor> backward = (resultNode) => {
                if (input.RequiresGrad)
                {
                    var inSpan = input.Data.AsReadOnlySpan();
                    var gradOutSpan = resultNode.Grad.AsReadOnlySpan();
                    var gradInSpan = input.Grad.AsSpan();

                    // Math: Derivative of ReLU is 1 if x > 0, else 0.
                    // grad_input += grad_output * (input > 0 ? 1 : 0)
                    // RyuJIT will heavily optimize this tight Span loop.
                    for (var i = 0; i < inSpan.Length; i++)
                    {
                        if (inSpan[i] > 0)
                        {
                            gradInSpan[i] += gradOutSpan[i];
                        }
                    }
                }
            };

            return Tensor.CreateOperationResult(
            data: resultData,
            dependencies: new List<Tensor>
            {
                input
            },
            backwardAction: backward
            );
        }

        // ====================================================================
        // 6. LOSS FUNCTIONS
        // ====================================================================

        /// <summary>
        /// Mean Squared Error (MSE) Loss.
        /// Math: L = (1/N) * sum((predictions - targets)^2)
        /// Returns a 1x1 Tensor representing the scalar loss.
        /// </summary>
        public static Tensor MSE(Tensor predictions, Tensor targets)
        {
            if (predictions.Data.Rows != targets.Data.Rows || predictions.Data.Cols != targets.Data.Cols)
            {
                throw new ArgumentException("Shape mismatch between predictions and targets in MSE.");
            }

            var n = predictions.Data.Rows * predictions.Data.Cols;
            var resultData = new FastMatrix<double>(1, 1);

            // 1. FORWARD PASS: Wyciągamy Spany lokalnie tylko na potrzeby Forward
            var predSpanForward = predictions.Data.AsReadOnlySpan();
            var targetSpanForward = targets.Data.AsReadOnlySpan();

            var dist = TensorPrimitives.Distance(predSpanForward, targetSpanForward);
            resultData[0, 0] = (dist * dist) / n;

            // 2. BACKWARD PASS REGISTRATION
            Action<Tensor> backward = (resultNode) => {
                var gradC = resultNode.Grad[0, 0];
                var factor = (2.0 / n) * gradC;

                // FIX: Wyciągamy Spany od nowa WEWNĄTRZ lambdy!
                // Dzięki temu żyją one krótko, bezpiecznie na stosie podczas wykonywania .Backward()
                var pData = predictions.Data.AsReadOnlySpan();
                var tData = targets.Data.AsReadOnlySpan();

                if (predictions.RequiresGrad)
                {
                    var pGrad = predictions.Grad.AsSpan();

                    for (var i = 0; i < pData.Length; i++)
                    {
                        pGrad[i] += factor * (pData[i] - tData[i]);
                    }
                }

                if (targets.RequiresGrad)
                {
                    var tGrad = targets.Grad.AsSpan();

                    for (var i = 0; i < tData.Length; i++)
                    {
                        tGrad[i] += factor * (tData[i] - pData[i]);
                    }
                }
            };

            return Tensor.CreateOperationResult(
            data: resultData,
            dependencies: new List<Tensor>
            {
                predictions,
                targets
            },
            backwardAction: backward
            );
        }

        /// <summary>
        /// Dodaje wektor Biasu do każdego wiersza macierzy wejściowej (Broadcasting).
        /// Wspiera Autograd: gradient biasu jest sumowany po wszystkich wierszach (Batchu).
        /// </summary>
        public static Tensor AddBias(Tensor input, Tensor bias)
        {
            if (bias.Data.Rows != 1 || bias.Data.Cols != input.Data.Cols)
            {
                throw new ArgumentException("Bias musi być wektorem 1xD, gdzie D to liczba kolumn wejścia.");
            }

            // 1. FORWARD PASS z bezalokacyjnym Broadcastingiem
            var resultData = new FastMatrix<double>(input.Data.Rows, input.Data.Cols);
            var broadcastedBias = BroadcastRowVector(bias.Data.AsSpan(), input.Data.Rows);
            Add(input.Data.AsView(), broadcastedBias, resultData.AsView());

            // 2. BACKWARD PASS REGISTRATION
            Action<Tensor> backward = (resultNode) => {
                var gradC = resultNode.Grad.AsView();

                // Gradient wejścia przepływa bez zmian (jak w zwykłym dodawaniu)
                if (input.RequiresGrad)
                {
                    Add(input.Grad.AsView(), gradC, input.Grad.AsView());
                }

                // Gradient biasu to SUMA gradientów po wszystkich wierszach (Batchu)
                if (bias.RequiresGrad)
                {
                    var biasGradSpan = bias.Grad.AsSpan();
                    for (var r = 0; r < resultNode.Grad.Rows; r++)
                    {
                        var rowGrad = resultNode.Grad.Row(r);
                        // SIMD Akumulacja: biasGrad += rowGrad
                        TensorPrimitives.Add(biasGradSpan, rowGrad, biasGradSpan);
                    }
                }
            };

            return Tensor.CreateOperationResult(
            data: resultData,
            dependencies: new List<Tensor>
            {
                input,
                bias
            },
            backwardAction: backward
            );
        }

        public static Tensor MaxPool2D(Tensor input, int channels, int inputH, int inputW, int poolSize)
        {
            var outputH = inputH / poolSize;
            var outputW = inputW / poolSize;
            var batchSize = input.Data.Rows;

            var resultData = new FastMatrix<double>(batchSize, channels * outputH * outputW);
            // Zapamiętujemy indeksy MAX, żeby wiedzieć, gdzie posłać gradient w Backward
            var maxIndices = new int[batchSize, channels * outputH * outputW];

            // --- FORWARD ---
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

            // --- BACKWARD ---
            Action<Tensor> backward = (resultNode) => {
                if (!input.RequiresGrad) return;

                for (var n = 0; n < batchSize; n++)
                {
                    var gradOut = resultNode.Grad.Row(n);
                    var gradIn = input.Grad.Row(n);

                    for (var i = 0; i < gradOut.Length; i++)
                    {
                        // Gradient płynie tylko do tego piksela, który był "największy"
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

        public static void Im2Col(
            ReadOnlySpan<double> input,
            int channels, int height, int width,
            int kSize, int stride, int padding,
            Span<double> output)
        {
            var outH = (height + 2 * padding - kSize) / stride + 1;
            var outW = (width + 2 * padding - kSize) / stride + 1;
            var channelSize = height * width;

            // Każda kolumna w 'output' to jedno rozciągnięte okienko filtra
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
                                    output[outIdx] = 0; // Padding
                            }
                        }
                    }
                }
            }
        }

        public static FastMatrix<double> MatMulRaw(FastMatrixView<double> A, FastMatrixView<double> B)
        {
            // C zawsze tworzymy jako nową, ciągłą macierz
            var C = new FastMatrix<double>(A.Rows, B.Cols);
            // Wykorzystujemy istniejący, zoptymalizowany silnik MatMul
            MatMul(A, B, C.AsView());

            return C;
        }

        /// <summary>
        /// Najszybsze możliwe mnożenie macierzy na surowych danych FastMatrix.
        /// Nie rejestruje operacji w Autogradzie.
        /// Oblicza: C = A * B
        /// </summary>
        public static FastMatrix<double> MatMulRaw(FastMatrix<double> A, FastMatrix<double> B)
        {
            if (A.Cols != B.Rows)
            {
                throw new ArgumentException($"Błędne wymiary macierzy: {A.Cols} != {B.Rows}");
            }

            var C = new FastMatrix<double>(A.Rows, B.Cols);

            // Optymalizacja pod kątem cache-friendly: 
            // Przechodzimy po wierszach A i wierszach B, używając TensorPrimitives.MultiplyAdd
            for (var i = 0; i < A.Rows; i++)
            {
                var rowA = A.Row(i);
                var rowC = C.Row(i);

                for (var k = 0; k < A.Cols; k++)
                {
                    var valA = rowA[k];
                    var rowB = B.Row(k);

                    // SIMD: rowC += valA * rowB
                    System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(rowB, valA, rowC, rowC);
                }
            }

            return C;
        }

        public static Tensor Conv2D(Tensor input, Tensor weights, int inC, int outC, int h, int w, int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1; // Poprawiono: w zamiast h
            var batchSize = input.Data.Rows;
            var kSquareInC = k * k * inC;

            // --- FORWARD ---
            var colMatrices = new FastMatrix<double>[batchSize];
            var resultData = new FastMatrix<double>(batchSize, outC * outH * outW);

            for (var n = 0; n < batchSize; n++)
            {
                colMatrices[n] = new FastMatrix<double>(kSquareInC, outH * outW);
                Im2Col(input.Data.Row(n), inC, h, w, k, 1, 0, colMatrices[n].AsSpan());

                // Używamy .AsView() dla zgodności z nowym MatMulRaw
                using var batchResult = MatMulRaw(weights.Data.AsView(), colMatrices[n].AsView());
                batchResult.AsSpan().CopyTo(resultData.Row(n));
            }

            // --- BACKWARD ---
            Action<Tensor> backward = (resultNode) => {
                var gradOutput = resultNode.Grad;

                for (var n = 0; n < batchSize; n++)
                {
                    using var gn = new FastMatrix<double>(outC, outH * outW);
                    gradOutput.Row(n).CopyTo(gn.AsSpan());

                    // 1. Gradient Wag (dW = gradOutput * colMatrix^T)
                    if (weights.RequiresGrad)
                    {
                        // colT jest po prawej -> Musi być ciągła (materializacja)[cite: 2]
                        using var colT = colMatrices[n].AsView().Transpose().ToContiguousFastMatrix();
                        using var dW_batch = MatMulRaw(gn.AsView(), colT.AsView());

                        TensorPrimitives.Add(weights.Grad.AsSpan(), dW_batch.AsSpan(), weights.Grad.AsSpan());
                    }

                    // 2. Gradient Wejścia (dX_col = weights^T * gradOutput)
                    if (input.RequiresGrad)
                    {
                        // weightsT jest po lewej -> Może zostać jako widok (dostęp skalarami jest OK)[cite: 2]
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

        public static void Col2Im(
            ReadOnlySpan<double> colData,
            int channels, int height, int width,
            int kSize, int stride, int padding,
            Span<double> gradInput)
        {
            gradInput.Clear(); // Zaczynamy od zera, bo będziemy akumulować (+=)
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
    }
}