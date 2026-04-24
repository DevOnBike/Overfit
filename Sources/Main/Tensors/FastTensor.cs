// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// Klasa zarządzająca wyłącznie cyklem życia pamięci na stercie (Heap).
    /// Nie wykonuje żadnych operacji logicznych ani matematycznych.
    /// </summary>
    public sealed class FastTensor<T> : IDisposable where T : struct
    {
        private T[] _data;
        private int _disposed;

        public readonly int Rank;
        public readonly int Size;

        // Zapamiętujemy tylko bazowy kształt, żeby wiedzieć, jaki widok wydać
        private readonly int _s0, _s1, _s2, _s3;

        // ========================================================================
        // KONSTRUKTORY ALOKUJĄCE
        // ========================================================================

        public FastTensor(int s0, bool clearMemory = true)
        {
            Rank = 1;
            Size = s0;

            _s0 = s0;

            Allocate(clearMemory);
        }

        public FastTensor(int s0, int s1, bool clearMemory = true)
        {
            Rank = 2;
            Size = s0 * s1;

            _s0 = s0;
            _s1 = s1;

            Allocate(clearMemory);
        }

        public FastTensor(int s0, int s1, int s2, bool clearMemory = true)
        {
            Rank = 3;
            Size = s0 * s1 * s2;
            _s0 = s0;
            _s1 = s1;
            _s2 = s2;

            Allocate(clearMemory);
        }

        public FastTensor(int s0, int s1, int s2, int s3, bool clearMemory = true)
        {
            Rank = 4;
            Size = s0 * s1 * s2 * s3;
            _s0 = s0;
            _s1 = s1;
            _s2 = s2;
            _s3 = s3;

            Allocate(clearMemory);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Allocate(bool clearMemory)
        {
            _data = ArrayPool<T>.Shared.Rent(Size);

            if (clearMemory)
            {
                _data.AsSpan(0, Size).Clear();
            }
        }

        // ========================================================================
        // WYDAWANIE WIDOKU (Główna rola tej klasy)
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView<T> GetView()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            return Rank switch
            {
                1 => new TensorView<T>(_data.AsSpan(0, Size), _s0),
                2 => new TensorView<T>(_data.AsSpan(0, Size), _s0, _s1),
                3 => new TensorView<T>(_data.AsSpan(0, Size), _s0, _s1, _s2),
                4 => new TensorView<T>(_data.AsSpan(0, Size), _s0, _s1, _s2, _s3),

                _ => throw new NotImplementedException("Obsługa max 4 wymiarów")
            };
        }

        /// <summary>
        /// Zwraca widok 2D z nadpisanym kształtem na te same dane.
        /// Używane przez AutogradNode w trybie aliasowania (Reshape zero-copy).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView<T> GetViewAs(int s0, int s1)
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            if (s0 * s1 != Size)
            {
                throw new ArgumentException($"Nowy kształt ({s0}×{s1}={s0 * s1}) nie pasuje do rozmiaru tensora ({Size}).");
            }

            return new TensorView<T>(_data.AsSpan(0, Size), s0, s1);
        }

        // ========================================================================
        // ZARZĄDZANIE PAMIĘCIĄ
        // ========================================================================

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0 && _data != null)
            {
                ArrayPool<T>.Shared.Return(_data);

                _data = null;
            }
        }

        public static FastTensor<T> SameShape(FastTensor<T> template, bool clearMemory = true)
        {
            return template.Rank switch
            {
                1 => new FastTensor<T>(template._s0, clearMemory),
                2 => new FastTensor<T>(template._s0, template._s1, clearMemory),
                3 => new FastTensor<T>(template._s0, template._s1, template._s2, clearMemory),
                4 => new FastTensor<T>(template._s0, template._s1, template._s2, template._s3, clearMemory),

                _ => throw new InvalidOperationException()
            };
        }

        // ========================================================================
        // MATERIALIZACJA WIDOKÓW
        // ========================================================================

        /// <summary>
        /// Tworzy nowy, ciągły fizyczny tensor z dowolnego (nawet transponowanego) widoku.
        /// </summary>
        public static FastTensor<T> FromView(TensorView<T> view)
        {
            // 1. Alokujemy nowy, pusty magazyn pod wymiary widoku
            var materializedTensor = view.Rank switch
            {
                1 => new FastTensor<T>(view.GetDim(0), clearMemory: false),
                2 => new FastTensor<T>(view.GetDim(0), view.GetDim(1), clearMemory: false),
                3 => new FastTensor<T>(view.GetDim(0), view.GetDim(1), view.GetDim(2), clearMemory: false),
                4 => new FastTensor<T>(view.GetDim(0), view.GetDim(1), view.GetDim(2), view.GetDim(3), clearMemory: false),

                _ => throw new InvalidOperationException("Nieobsługiwany wymiar")
            };

            if (view.IsContiguous)
            {
                view.AsReadOnlySpan().CopyTo(materializedTensor.GetView().AsSpan());

                return materializedTensor;
            }

            // 3. Jeśli widok został "poszatkowany" (np. przez Transpose2D), 
            var targetSpan = materializedTensor.GetView().AsSpan();
            var index = 0;

            if (view.Rank == 2)
            {
                for (var i = 0; i < view.GetDim(0); i++)
                {
                    for (var j = 0; j < view.GetDim(1); j++)
                    {
                        targetSpan[index++] = view[i, j];
                    }
                }
            }
            else
            {
                throw new NotImplementedException("todo: Kopiowanie nieciągłych widoków > 2D nie jest zaimplementowane.");
            }

            return materializedTensor;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddInPlace(FastTensor<T> other)
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);
    
            var target = GetView().AsSpan();
            var source = other.GetView().AsReadOnlySpan();

            if (target.Length != source.Length)
            {
                throw new ArgumentException("Tensory muszą mieć ten sam rozmiar do operacji In-Place.");
            }

            // Jeśli T to float, używamy zoptymalizowanych prymitywów .NET
            if (typeof(T) == typeof(float))
            {
                var targetFloat = MemoryMarshal.Cast<T, float>(target);
                var sourceFloat = MemoryMarshal.Cast<T, float>(source);
                
                TensorPrimitives.Add(targetFloat, sourceFloat, targetFloat);
            }
            else
            {
                // Fallback dla innych typów
                for (var i = 0; i < target.Length; i++)
                {
                    // Wymagałoby to generycznej matematyki w .NET 7+, 
                    // dla uproszczenia załóżmy float lub zaimplementuj per typ
                }
            }
        }
    }
}