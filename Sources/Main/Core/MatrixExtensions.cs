// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Core
{
    public static class MatrixExtensions
    {
        public static int ArgMax(this TensorView<float> tensor, int row = 0)
        {
            // Omijamy wywołania metod i alokacje - bierzemy płaski fragment pamięci danego wiersza.
            // Magia .NET: Sprzętowe SIMD (AVX2/AVX-512) robiące dokładnie to samo,
            // ale sprawdzające po 4-8 liczb w jednym takcie zegara!
            var cols = tensor.GetDim(1);
            var rowSpan = tensor.AsReadOnlySpan().Slice(row * cols, cols);

            return TensorPrimitives.IndexOfMax(rowSpan);
        }
    }
}