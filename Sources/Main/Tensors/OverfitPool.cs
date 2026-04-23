// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Collections.Concurrent;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// Niezależna, odporna na Garbage Collectora pula pamięci (GC-Immune Pool).
    /// Zatrzymuje wyrzucanie zbuforowanych tensorów podczas zbiórek pamięci Gen2.
    /// </summary>
    internal static class OverfitPool<T> where T : struct
    {
        // Gigantyczny ArrayPool bez narzutu ConcurrentBag!
        // Utrzyma 1024 tablice per bucket, rozwiązując problem kradzieży wątków.
        public static readonly ArrayPool<T> Shared = ArrayPool<T>.Create(1024 * 1024 * 64, 1024);
    }
}