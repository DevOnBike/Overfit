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
        // ZERO-ALLOC NODE FACTORY (DOD)
        // ====================================================================

        /// <summary>
        /// Centralna metoda tworząca nowe węzły na taśmie.
        /// Wycina pamięć z Areny (jeśli podano Graf) i opakowuje w lekki AutogradNode.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static AutogradNode AllocateNode(ComputationGraph? graph, TensorShape shape, bool requiresGrad, bool clearMemory = true)
        {
            TensorStorage<float> storage;

            if (graph != null)
            {
                storage = graph.AllocateIntermediate(shape.Size);
                if (clearMemory)
                {
                    storage.AsSpan().Clear();
                }
            }
            else
            {
                storage = new TensorStorage<float>(shape.Size, clearMemory);
            }

            // Zwracamy od razu gotowy, bezpieczny węzeł
            return new AutogradNode(storage, shape, requiresGrad);
        }
    }
}