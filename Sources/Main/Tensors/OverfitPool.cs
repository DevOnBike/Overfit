// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// Independent, GC-immune memory pool.
    /// Prevents cached tensors from being evicted during Gen2 garbage collections.
    /// </summary>
    internal static class OverfitPool<T> where T : struct
    {
        // Oversized ArrayPool with no ConcurrentBag overhead!
        // Retains up to 1024 arrays per bucket, eliminating the thread-stealing problem.
        public static readonly ArrayPool<T> Shared = ArrayPool<T>.Create(1024 * 1024 * 64, 1024);
    }
}