// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Core
{
    public static class MatrixExtensions
    {
        public static int ArgMax(this TensorView<float> tensor, int row = 0)
        {
            // Flat memory slice of the row → the single vectorized argmax (TensorPrimitives.IndexOfMax,
            // AVX2/AVX-512, ~4-8 values per clock) via MathUtils — one source of truth across the engine.
            var cols = tensor.GetDim(1);
            var rowSpan = tensor.AsReadOnlySpan().Slice(row * cols, cols);

            return MathUtils.ArgMax(rowSpan);
        }
    }
}