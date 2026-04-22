// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// Globalna, powiększona pula pamięci dla silnika Overfit.
    /// Zatrzymuje zjawisko "Bucket Exhaustion" dla dużych sieci konwolucyjnych.
    /// </summary>
    internal static class OverfitPool<T> where T : struct
    {
        // 1024 tablice na rozmiar (zamiast 50) i wielkie rozmiary do 64M elementów
        public static readonly ArrayPool<T> Shared = ArrayPool<T>.Create(1024 * 1024 * 64, 1024);
    }
}