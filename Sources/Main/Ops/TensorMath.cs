// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Wpinamy Core

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // CONSTANTS
        // ====================================================================

        private const long ParallelThreshold = 4096;
        private const int StackAllocThreshold = 1024;
        private const int BatchSequentialThreshold = 128;

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
        // ZERO-ALLOC NODE FACTORY (DOD)
        // ====================================================================

        /// <summary>
        /// Centralna metoda tworzÄ…ca nowe wÄ™zÅ‚y na taÅ›mie.
        /// Wycina pamiÄ™Ä‡ z Areny (jeÅ›li podano Graf) i opakowuje w lekki AutogradNode.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static AutogradNode AllocateNode(ComputationGraph? graph, TensorShape shape, bool requiresGrad, bool clearMemory = true)
        {
            if (graph != null)
            {
                // Route through graph factory so Ownership is set from birth.
                return requiresGrad
                    ? graph.CreateTemporary(shape, requiresGrad: true,  clearMemory: clearMemory)
                    : graph.CreateAuxiliary(shape, clearMemory: clearMemory);
            }

            // No graph (inference path): standalone allocation, not graph-owned.
            var storage = new TensorStorage<float>(shape.Size, clearMemory);
            return new AutogradNode(storage, shape, requiresGrad)
            {
                Ownership = AutogradNodeOwnership.ExternalBorrowed,
            };
        }
    }
}