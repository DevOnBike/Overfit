// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Ekstremalnie lekki widok na pamięć tensora. 
    /// Żyje TYLKO na stosie. Nigdy nie alokuje pamięci na stercie.
    /// </summary>
    public readonly ref struct TensorView<T> where T : struct
    {
        // Pamięć, na którą patrzymy (pod spodem może to być ArrayPool, stackalloc, lub unmanaged memory)
        private readonly Span<T> _data;

        public readonly int Rank;
        public readonly int Size;
        public readonly int Offset;

        // Metadane inline (dla wydajności Rank <= 4)
        private readonly int _s0, _s1, _s2, _s3;
        private readonly int _st0, _st1, _st2, _st3;

        // Konstruktor inicjujący widok (wykonywany w całości na stosie)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView(Span<T> data, int s0, int s1)
        {
            _data = data;
            Rank = 2;
            Size = s0 * s1;
            Offset = 0;

            _s0 = s0; _s1 = s1;
            _s2 = 0; _s3 = 0;

            _st0 = s1; _st1 = 1;
            _st2 = 0; _st3 = 0;
        }

        // Zwracamy czysty Span do obliczeń w TensorPrimitives
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan() => _data;

        // ========================================================================
        // MAGIA ZERO-ALLOCATION: Operacje zwracają nowe struktury, a nie klasy!
        // ========================================================================

        /// <summary>
        /// Zwraca przetransponowany widok w czasie O(1). 
        /// Żadnego kopiowania danych, żadnej alokacji RAMu!
        /// </summary>
        public TensorView<T> Transpose2D()
        {
            if (Rank != 2) throw new InvalidOperationException("To jest dedykowane dla 2D");

            // Tworzymy nową strukturę przekazując TEN SAM Span, ale zamieniając wymiary i kroki (strides)
            var transposedView = new TensorView<T>(); // Pusty konstruktor by ominąć sprawdzanie

            // W prawdziwym kodzie użyjesz prywatnego konstruktora z pełną listą _s0.._s3
            // Gdzie np. nowe _s0 = stare _s1, a nowe _st0 = stare _st1.

            return transposedView;
        }

        /// <summary>
        /// Zwraca płaski, jednowymiarowy widok na te same dane.
        /// </summary>
        public TensorView<T> Flatten()
        {
            return new TensorView<T>(_data, Size, 1); // Traktujemy jako wektor 1D
        }
    }
}
