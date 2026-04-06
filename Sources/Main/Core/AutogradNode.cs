// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Represents a node in the computation graph, holding data and its associated gradient.
    /// </summary>
    public sealed class AutogradNode : IDisposable
    {
        public FastTensor<float> Data { get; }
        public FastTensor<float> Grad { get; private set; }
        public bool RequiresGrad { get; set; }

        public AutogradNode(FastTensor<float> data, bool requiresGrad = true)
        {
            Data = data;
            RequiresGrad = requiresGrad;

            if (requiresGrad)
            {
                // Gradients must match data shape. Memory is cleared to ensure zero-start.
                Grad = FastTensor<float>.SameShape(data, clearMemory: true);
            }
        }

        /// <summary>
        /// Retrieves the first scalar value (typically used for Loss nodes).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Forward() => Data[0];

        public void Dispose()
        {
            Data?.Dispose();
            Grad?.Dispose();
        }
    }
}