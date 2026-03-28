using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit
{
    /// <summary>
    /// Bezalokacyjny widok na pamięć macierzy.
    /// Definiuje kształt (Rows, Cols) oraz fizyczny układ w pamięci (Strides, Offset).
    /// </summary>
    public readonly ref struct FastMatrixView<T>
    {
        private readonly Span<T> _data;

        public int Rows { get; }
        public int Cols { get; }

        // Strides (Kroki) - o ile elementów przeskoczyć w surowej tablicy, 
        // aby przejść do następnego wiersza lub kolumny.
        public int RowStride { get; }
        public int ColStride { get; }
        public int Offset { get; }

        /// <summary>
        /// Zwraca true, jeśli pamięć w widoku układa się w jeden ciągły blok (Row-Major).
        /// Pozwala to na pełną akcelerację SIMD całego bloku.
        /// </summary>
        public bool IsContiguous => ColStride == 1 && RowStride == Cols;

        public FastMatrixView(Span<T> data, int rows, int cols, int rowStride, int colStride, int offset)
        {
            _data = data;
            Rows = rows;
            Cols = cols;
            RowStride = rowStride;
            ColStride = colStride;
            Offset = offset;
        }

        public ref T this[int row, int col]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _data[Offset + row * RowStride + col * ColStride];
        }

        // ====================================================================
        // MAGIA WIDOKÓW - Te operacje kosztują O(1) i 0 bajtów alokacji!
        // ====================================================================

        /// <summary>
        /// Transpozycja macierzy w czasie O(1). 
        /// Po prostu zamienia miejscami wymiary i kroki. Zero kopiowania pamięci!
        /// </summary>
        public FastMatrixView<T> Transpose()
        {
            return new FastMatrixView<T>(_data, rows: Cols, cols: Rows, rowStride: ColStride, colStride: RowStride, offset: Offset);
        }

        /// <summary>
        /// Wycinanie podmacierzy (Slicing) w czasie O(1).
        /// Modyfikuje tylko Offset i wymiary. Zero kopiowania!
        /// </summary>
        public FastMatrixView<T> Slice(int startRow, int startCol, int rows, int cols)
        {
            if (startRow + rows > Rows || startCol + cols > Cols)
            {
                throw new ArgumentOutOfRangeException("Slice exceeds view dimensions.");
            }

            var newOffset = Offset + startRow * RowStride + startCol * ColStride;

            return new FastMatrixView<T>(_data, rows: rows, cols: cols, rowStride: RowStride, colStride: ColStride, offset: newOffset);
        }

        /// <summary>
        /// Pobiera wiersz jako ciągły Span (jeśli to możliwe).
        /// Jeśli macierz jest transponowana, wiersze przestają być ciągłe w pamięci.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> Row(int row)
        {
            if (ColStride != 1)
            {
                throw new InvalidOperationException("Cannot return a contiguous Span for a row when ColStride != 1 (e.g., in a transposed view).");
            }

            var start = Offset + row * RowStride;
            
            return _data.Slice(start, Cols);
        }
    }
}