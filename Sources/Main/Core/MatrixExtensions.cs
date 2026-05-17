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
            // We bypass method calls and allocations — we take the flat memory slice of the given row.
            // .NET magic: hardware SIMD (AVX2/AVX-512) doing exactly the same thing,
            // but checking 4-8 values per clock cycle!
            var cols = tensor.GetDim(1);
            var rowSpan = tensor.AsReadOnlySpan().Slice(row * cols, cols);

            return TensorPrimitives.IndexOfMax(rowSpan);
        }
    }
}