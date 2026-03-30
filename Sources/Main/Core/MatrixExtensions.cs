using System;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Core
{
    public static class MatrixExtensions
    {
        public static int ArgMax(this FastMatrix<double> matrix, int row = 0)
        {
            // Magia .NET: Sprzętowe SIMD (AVX2/AVX-512) robiące dokładnie to samo,
            // ale sprawdzające po 4-8 liczb w jednym takcie zegara!
            return TensorPrimitives.IndexOfMax(matrix.ReadOnlyRow(row));
        }
    }
}