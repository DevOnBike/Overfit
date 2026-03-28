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
                offset: 0
            );
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
            Action<Tensor> backward = (resultNode) =>
            {
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
                dependencies: new List<Tensor> { left, right },
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
            Action<Tensor> backward = (resultNode) =>
            {
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
                dependencies: new List<Tensor> { left, right },
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
            Action<Tensor> backward = (resultNode) =>
            {
                if (input.RequiresGrad)
                {
                    var inSpan = input.Data.AsReadOnlySpan();
                    var gradOutSpan = resultNode.Grad.AsReadOnlySpan();
                    var gradInSpan = input.Grad.AsSpan();

                    // Math: Derivative of ReLU is 1 if x > 0, else 0.
                    // grad_input += grad_output * (input > 0 ? 1 : 0)
                    // RyuJIT will heavily optimize this tight Span loop.
                    for (int i = 0; i < inSpan.Length; i++)
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
                dependencies: new List<Tensor> { input },
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

            int n = predictions.Data.Rows * predictions.Data.Cols;
            var resultData = new FastMatrix<double>(1, 1);

            // 1. FORWARD PASS: Wyciągamy Spany lokalnie tylko na potrzeby Forward
            var predSpanForward = predictions.Data.AsReadOnlySpan();
            var targetSpanForward = targets.Data.AsReadOnlySpan();

            double dist = TensorPrimitives.Distance(predSpanForward, targetSpanForward);
            resultData[0, 0] = (dist * dist) / n;

            // 2. BACKWARD PASS REGISTRATION
            Action<Tensor> backward = (resultNode) =>
            {
                double gradC = resultNode.Grad[0, 0];
                double factor = (2.0 / n) * gradC;

                // FIX: Wyciągamy Spany od nowa WEWNĄTRZ lambdy!
                // Dzięki temu żyją one krótko, bezpiecznie na stosie podczas wykonywania .Backward()
                var pData = predictions.Data.AsReadOnlySpan();
                var tData = targets.Data.AsReadOnlySpan();

                if (predictions.RequiresGrad)
                {
                    var pGrad = predictions.Grad.AsSpan();
                    
                    for (int i = 0; i < pData.Length; i++)
                    {
                        pGrad[i] += factor * (pData[i] - tData[i]);
                    }
                }

                if (targets.RequiresGrad)
                {
                    var tGrad = targets.Grad.AsSpan();
                    
                    for (int i = 0; i < tData.Length; i++)
                    {
                        tGrad[i] += factor * (tData[i] - pData[i]);
                    }
                }
            };

            return Tensor.CreateOperationResult(
                data: resultData,
                dependencies: new List<Tensor> { predictions, targets },
                backwardAction: backward
            );
        }
    }
}