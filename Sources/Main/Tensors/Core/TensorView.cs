using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// PURE DOD: Soczewka (Lens) nakładana na surowy Span pamięci.
    /// Żyje tylko na stosie (ref struct). Łączy Span, Kształt i Kroki.
    /// </summary>
    public readonly ref struct TensorView<T> where T : unmanaged
    {
        private readonly Span<T> _data;

        public readonly TensorShape Shape;
        public readonly TensorStrides Strides;
        public readonly int Offset;

        // Właściwości pomocnicze
        public int Rank => Shape.Rank;
        public int Size => Shape.Size;
        public bool IsContiguous => Strides.IsContiguous(Shape);

        // ========================================================================
        // KONSTRUKTORY
        // ========================================================================

        /// <summary>
        /// Podstawowy konstruktor dla widoku ciągłego.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView(Span<T> data, TensorShape shape)
        {
            _data = data;
            Shape = shape;
            Strides = TensorStrides.Contiguous(shape);
            Offset = 0;
        }

        /// <summary>
        /// Zaawansowany konstruktor dla widoków po transformacjach (np. Transpose).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TensorView(Span<T> data, TensorShape shape, TensorStrides strides, int offset)
        {
            _data = data;
            Shape = shape;
            Strides = strides;
            Offset = offset;
        }

        // ========================================================================
        // DOSTĘP DO PAMIĘCI
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Tensor nie jest ciągły w pamięci! Użyj manualnych indeksów lub zmaterializuj widok.");
            }

            return _data.Slice(Offset, Size);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan() => AsSpan();

        // ========================================================================
        // INDEXERY (Używają nowej struktury TensorStrides)
        // ========================================================================

        public ref T this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _data[Offset + Strides.GetOffset(i)];
        }

        public ref T this[int i, int j]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _data[Offset + Strides.GetOffset(i, j)];
        }

        public ref T this[int i, int j, int k]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _data[Offset + Strides.GetOffset(i, j, k)];
        }

        // ========================================================================
        // TRANSFORMACJE (ZERO-ALLOCATION)
        // ========================================================================

        public TensorView<T> Reshape(TensorShape newShape)
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Nie można zmienić kształtu nieciągłego widoku.");
            }

            if (newShape.Size != Size)
            {
                throw new ArgumentException($"Nowy rozmiar {newShape.Size} nie pasuje do obecnego {Size}");
            }

            return new TensorView<T>(_data, newShape);
        }

        public TensorView<T> Transpose2D()
        {
            if (Rank != 2)
            {
                throw new InvalidOperationException("Funkcja dedykowana dla macierzy 2D.");
            }

            // Kształt odwrócony (D1, D0), Strides odwrócone (S1, S0)
            var newShape = new TensorShape(Shape.D1, Shape.D0);
            var newStrides = Strides.Transpose2D();

            return new TensorView<T>(_data, newShape, newStrides, Offset);
        }

        public TensorView<T> Slice(int offsetIndex, int sliceLength)
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Slice wymaga ciągłej pamięci bazowej.");
            }

            var slicedSpan = _data.Slice(Offset + offsetIndex, sliceLength);
            
            return new TensorView<T>(slicedSpan, new TensorShape(sliceLength));
        }

        public TensorView<T> Flatten()
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Flatten wymaga ciągłej pamięci bazowej.");
            }

            return new TensorView<T>(_data, new TensorShape(Size));
        }
        
        // ========================================================================
        // KOMPATYBILNOŚĆ WSTECZNA (Zgodność z dawnym API)
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetDim(int index)
        {
            return Shape[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetStride(int index)
        {
            return Strides[index];
        }
    }
}