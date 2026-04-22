// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// Niezależna, odporna na Garbage Collectora pula pamięci (GC-Immune Pool).
    /// Zatrzymuje wyrzucanie zbuforowanych tensorów podczas zbiórek pamięci Gen2.
    /// </summary>
    internal static class OverfitPool<T> where T : struct
    {
        private static readonly ConcurrentDictionary<int, ConcurrentBag<T[]>> _buckets = new();

        public static T[] Rent(int size)
        {
            if (_buckets.TryGetValue(size, out var bag) && bag.TryTake(out var array))
            {
                return array;
            }
            
            return new T[size];
        }

        public static void Return(T[] array)
        {
            if (array == null)
            {
                return;
            }
            
            var bag = _buckets.GetOrAdd(array.Length, _ => new ConcurrentBag<T[]>());
            bag.Add(array);
        }
    }
}