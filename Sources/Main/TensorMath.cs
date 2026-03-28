using System.Numerics;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit
{
    /// <summary>
    /// Computational engine operating on views. 
    /// Implements allocation-free Broadcasting and parallel SIMD operations.
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
    }
}