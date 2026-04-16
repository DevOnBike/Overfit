// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // CONSTANTS
        // ====================================================================

        private const long ParallelThreshold = 4096;
        private const int StackAllocThreshold = 1024;
        private const int BatchSequentialThreshold = 32;

        // ====================================================================
        // SOFTMAX
        // ====================================================================

        public static void StableSoftmax(ReadOnlySpan<float> input, Span<float> output)
        {
            var maxVal = TensorPrimitives.Max(input);
            var shifted = output;

            TensorPrimitives.Subtract(input, maxVal, shifted);
            TensorPrimitives.Exp(shifted, shifted);

            var sumExp = TensorPrimitives.Sum(shifted);
            TensorPrimitives.Divide(shifted, sumExp, output);
        }

        // ====================================================================
        // ALLOCATOR
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static FastTensor<float> AllocateLike(AutogradNode node, bool clearMemory = true)
        {
            var v = node.DataView;
            return v.Rank switch
            {
                1 => new FastTensor<float>(v.GetDim(0), clearMemory),
                2 => new FastTensor<float>(v.GetDim(0), v.GetDim(1), clearMemory),
                3 => new FastTensor<float>(v.GetDim(0), v.GetDim(1), v.GetDim(2), clearMemory),
                4 => new FastTensor<float>(v.GetDim(0), v.GetDim(1), v.GetDim(2), v.GetDim(3), clearMemory),
                _ => throw new InvalidOperationException("Nieobsługiwany wymiar")
            };
        }
    }
}
