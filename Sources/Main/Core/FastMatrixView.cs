using System.Numerics;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Bezalokacyjny widok na pamięć macierzy (ref struct — zero heap alloc).
    /// </summary>
    public readonly ref struct FastMatrixView<T> where T : struct, IFloatingPointIeee754<T>
    {
        private readonly Span<T> _data;

        public int Rows { get; }
        public int Cols { get; }
        public int RowStride { get; }
        public int ColStride { get; }
        public int Offset { get; }

        public bool IsContiguous => ColStride == 1 && RowStride == Cols;

        public FastMatrixView(Span<T> data, int rows, int cols, int rowStride, int colStride, int offset)
        {
            _data = data;
            Rows = rows; Cols = cols;
            RowStride = rowStride; ColStride = colStride;
            Offset = offset;
        }

        public ref T this[int row, int col]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _data[Offset + row * RowStride + col * ColStride];
        }

        public FastMatrixView<T> Transpose() =>
            new(_data, rows: Cols, cols: Rows, rowStride: ColStride, colStride: RowStride, offset: Offset);

        public FastMatrixView<T> Slice(int startRow, int startCol, int rows, int cols)
        {
            if (startRow + rows > Rows || startCol + cols > Cols)
            {
                throw new ArgumentOutOfRangeException("Slice exceeds view dimensions.");
            }

            return new(_data, rows, cols, RowStride, ColStride,
                Offset + startRow * RowStride + startCol * ColStride);
        }

        // ── Row access ───────────────────────────────────────────────────────

        /// <summary>Ciągły Span wiersza — tylko gdy ColStride==1 (ciągły widok).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> Row(int row)
        {
            if (ColStride != 1)
            {
                throw new InvalidOperationException("Cannot return contiguous Span for a row when ColStride != 1 (transposed view).");
            }
            
            return _data.Slice(Offset + row * RowStride, Cols);
        }

        /// <summary>ReadOnly Span wiersza — tylko gdy ColStride==1.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> ReadOnlyRow(int row)
        {
            if (ColStride != 1)
            {
                throw new InvalidOperationException("Cannot return contiguous ReadOnlySpan for a row when ColStride != 1.");
            }
            
            return _data.Slice(Offset + row * RowStride, Cols);
        }

        // ── Materializacja ───────────────────────────────────────────────────

        /// <summary>
        /// Materializuje widok do ciągłej FastMatrix.
        /// Dla ciągłych widoków: jeden CopyTo (memcpy).
        /// Dla transponowanych: per-row copy — unika N² scalar ops.
        /// Caller odpowiada za Dispose().
        /// </summary>
        public FastMatrix<T> ToContiguousFastMatrix()
        {
            var result = new FastMatrix<T>(Rows, Cols);

            if (IsContiguous)
            {
                // Ciągły blok — jeden AVX memcpy
                _data.Slice(Offset, Size).CopyTo(result.AsSpan());
            }
            else if (ColStride == 1)
            {
                // Wiersze ciągłe, ale między nimi jest padding (np. Slice)
                // Kopiujemy wiersz po wierszu przez SIMD CopyTo
                for (var r = 0; r < Rows; r++)
                {
                    _data.Slice(Offset + r * RowStride, Cols).CopyTo(result.Row(r));
                }
            }
            else
            {
                // Transponowany — kolumny stają się wierszami, scalar fallback
                for (var r = 0; r < Rows; r++)
                {
                    for (var c = 0; c < Cols; c++)
                    {
                        result[r, c] = _data[Offset + r * RowStride + c * ColStride];
                    }
                }
            }

            return result;
        }

        private int Size => Rows * Cols;
    }
}