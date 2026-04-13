// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Ekstremalnie lekki, wszechstronny widok na pamięć tensora. 
    /// Przejmuje całą logikę wymiarów, kroków (strides) i nawigacji po danych.
    /// Żyje TYLKO na stosie (zero-allocation).
    /// </summary>
    public readonly ref struct TensorView<T> where T : struct
    {
        private readonly Span<T> _data;

        public readonly int Rank;
        public readonly int Size;
        public readonly int Offset;
        public readonly bool IsContiguous;

        private readonly int _s0, _s1, _s2, _s3;
        private readonly int _st0, _st1, _st2, _st3;

        // ========================================================================
        // KONSTRUKTORY DLA WIDOKÓW BAZOWYCH (Ciągłych)
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView(Span<T> data, int s0)
            : this(data, 1, s0, 0, true, s0, 0, 0, 0, 1, 0, 0, 0) { }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView(Span<T> data, int s0, int s1)
            : this(data, 2, s0 * s1, 0, true, s0, s1, 0, 0, s1, 1, 0, 0) { }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView(Span<T> data, int s0, int s1, int s2)
            : this(data, 3, s0 * s1 * s2, 0, true, s0, s1, s2, 0, s1 * s2, s2, 1, 0) { }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView(Span<T> data, int s0, int s1, int s2, int s3)
            : this(data, 4, s0 * s1 * s2 * s3, 0, true, s0, s1, s2, s3, s1 * s2 * s3, s2 * s3, s3, 1) { }

        // Prywatny, główny konstruktor do tworzenia skomplikowanych (np. odwróconych) widoków
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TensorView(Span<T> data, int rank, int size, int offset, bool isContiguous,
                           int s0, int s1, int s2, int s3,
                           int st0, int st1, int st2, int st3)
        {
            _data = data; Rank = rank; Size = size; Offset = offset; IsContiguous = isContiguous;
            _s0 = s0; _s1 = s1; _s2 = s2; _s3 = s3;
            _st0 = st0; _st1 = st1; _st2 = st2; _st3 = st3;
        }

        // ========================================================================
        // DOSTĘP DO DANYCH
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            return !IsContiguous ? throw new InvalidOperationException("Tensor nie jest ciągły w pamięci! Użyj manualnych indeksów.") : _data.Slice(Offset, Size);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            return AsSpan();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetDim(int index)
        {
            return index switch { 0 => _s0, 1 => _s1, 2 => _s2, 3 => _s3, _ => 0 };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetStride(int index)
        {
            return index switch { 0 => _st0, 1 => _st1, 2 => _st2, 3 => _st3, _ => 0 };
        }

        // Indexery przeniesione z FastTensor
        public ref T this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return ref _data[Offset + i * _st0];
            }
        }

        public ref T this[int i, int j]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return ref _data[Offset + i * _st0 + j * _st1];
            }
        }

        public ref T this[int i, int j, int k]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return ref _data[Offset + i * _st0 + j * _st1 + k * _st2];
            }
        }

        // ========================================================================
        // TRANSFORMACJE (ZERO-ALLOCATION)
        // ========================================================================

        public TensorView<T> Reshape(int newS0, int newS1)
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Nie można zmienić kształtu nieciągłego widoku.");
            }
            
            if (newS0 * newS1 != Size)
            {
                throw new ArgumentException($"Nowy rozmiar {newS0 * newS1} nie pasuje do obecnego {Size}");
            }

            return new TensorView<T>(_data, newS0, newS1);
        }

        public TensorView<T> Transpose2D()
        {
            if (Rank != 2)
            {
                throw new InvalidOperationException("Funkcja dedykowana dla 2D.");
            }

            // Zamieniamy miejscami wymiary (s) i kroki (st). Ustawiamy isContiguous = false.
            return new TensorView<T>(
                _data, Rank, Size, Offset, false,
                _s1, _s0, _s2, _s3,
                _st1, _st0, _st2, _st3
            );
        }

        public TensorView<T> Slice(int offsetIndex, int sliceLength)
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Slice wymaga ciągłej pamięci bazowej.");
            }

            var slicedSpan = _data.Slice(Offset + offsetIndex, sliceLength);
            return new TensorView<T>(slicedSpan, sliceLength);
        }

        public TensorView<T> Flatten()
        {
            return !IsContiguous ? throw new InvalidOperationException("Flatten wymaga ciągłej pamięci bazowej.") : new TensorView<T>(_data, Size);
        }
    }
}