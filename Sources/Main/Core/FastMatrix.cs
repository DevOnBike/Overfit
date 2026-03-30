using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Core
{
    public class FastMatrix<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
    {
        private T[] _data;
        private bool _disposed;

        public int Cols { get; }
        public int Rows { get; }
        public int Size { get; }

        public FastMatrix(int rows, int cols)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);

            Rows = rows;
            Cols = cols;

            Size = checked(rows * cols);
            _data = ArrayPool<T>.Shared.Rent(Size);

            _data.AsSpan(0, Size).Clear();
        }

        public ref T this[int row, int col]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return ref _data![row * Cols + col];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> Row(int row)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _data!.AsSpan(row * Cols, Cols);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> ReadOnlyRow(int row)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _data!.AsSpan(row * Cols, Cols);
        }

        public Span<T> AsSpan()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _data!.AsSpan(0, Size);
        }

        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _data!.AsSpan(0, Size);
        }

        public TensorSpan<T> AsTensor()
        {
            return new(AsSpan(), [Rows, Cols], default);
        }

        public ReadOnlyTensorSpan<T> AsReadOnlyTensor()
        {
            return new(AsReadOnlySpan(), [Rows, Cols], default);
        }

        public void Clear()
        {
            AsSpan().Clear();
        }

        public void CopyFrom(ReadOnlySpan<T> source)
        {
            if (source.Length != Size)
            {
                throw new ArgumentException($"Expected {Size} elements, got {source.Length}.");
            }
            source.CopyTo(AsSpan());
        }

        // ── SIMD  (TensorPrimitives .NET 10) ─────────────────────────

        /// <summary>this += other (in-place, element-wise). SIMD-accelerated.</summary>
        public void Add(FastMatrix<T> other)
        {
            ThrowIfShapeMismatch(other);
            TensorPrimitives.Add(AsReadOnlySpan(), other.AsReadOnlySpan(), AsSpan());
        }

        /// <summary>this -= other (in-place, element-wise). SIMD-accelerated.</summary>
        public void Subtract(FastMatrix<T> other)
        {
            ThrowIfShapeMismatch(other);
            TensorPrimitives.Subtract(AsReadOnlySpan(), other.AsReadOnlySpan(), AsSpan());
        }

        /// <summary>this *= scalar (in-place). SIMD-accelerated.</summary>
        public void MultiplyScalar(T scalar)
        {
            TensorPrimitives.Multiply(AsReadOnlySpan(), scalar, AsSpan());
        }

        /// <summary>Suma kwadratów wszystkich elementów (‖A‖_F²). Używane do Mahalanobis distance.</summary>
        public T SumOfSquares()
        {
            // Metoda Dot dla generics zwraca typ T
            return TensorPrimitives.Dot(AsReadOnlySpan(), AsReadOnlySpan());
        }

        /// <summary>Norma Frobeniusa ‖A‖_F = sqrt(Σ aᵢⱼ²). SIMD-accelerated.</summary>
        public T FrobeniusNorm()
        {
            return TensorPrimitives.Norm(AsReadOnlySpan());
        }

        public void Softmax()
        {
            TensorPrimitives.SoftMax(AsReadOnlySpan(), AsSpan());
        }

        // ── (Strides & Views) ─────────────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FastMatrixView<T> AsView()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            return new FastMatrixView<T>(AsSpan(), Rows, Cols, rowStride: Cols, colStride: 1, offset: 0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FastMatrixView<T> Transpose()
        {
            return AsView().Transpose();
        }

        /// <summary>
        /// Zwraca wycinek macierzy (tzw. okno lub ROI) w czasie O(1).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FastMatrixView<T> Slice(int startRow, int startCol, int rows, int cols)
        {
            return AsView().Slice(startRow, startCol, rows, cols);
        }

        /// <summary>
        /// Materializes the view into a new, contiguous FastMatrix.
        /// Essential for Autograd when a transposed view needs to be the right-hand operand in SIMD GEMM.
        /// Caller is responsible for disposing the returned FastMatrix.
        /// </summary>
        public FastMatrix<T> ToContiguousFastMatrix()
        {
            var result = new FastMatrix<T>(Rows, Cols);

            AsReadOnlySpan().CopyTo(result.AsSpan());

            return result;
        }

        // ── Dispose Pattern ───────────────────────────────────────────
        // Wzorzec z virtual Dispose(bool) — wymagany gdy klasa nie jest sealed.
        // Podklasy nadpisują Dispose(bool) aby zwolnić własne zasoby,
        // następnie wywołują base.Dispose(disposing) do zwolnienia _data.

        /// <summary>
        /// Nadpisz tę metodę w podklasie aby zwolnić własne zasoby.
        /// Zawsze wywołuj base.Dispose(disposing) na końcu.
        /// </summary>
        /// <param name="disposing">
        /// true  — wywołane przez Dispose(); zwolnij zasoby managed i unmanaged.
        /// false — wywołane przez finalizer; zwalniaj tylko unmanaged (managed może być już zebrany przez GC).
        /// </param>
        protected virtual void Dispose(bool disposing)
        {
            // Atomowa zamiana: tylko pierwszy wywołujący przechodzi dalej.
            // Chroni przed double-return do ArrayPool z dwóch wątków jednocześnie.
            if (Interlocked.Exchange(ref _disposed, true))
            {
                return;
            }

            if (disposing)
            {
                // Managed resources: zwróć bufor do ArrayPool.
                var rented = Interlocked.Exchange(ref _data, null!);

                if (rented != null)
                {
                    ArrayPool<T>.Shared.Return(rented);
                }
            }

            // Unmanaged resources: brak — ArrayPool to managed wrapper.
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(disposing: true);
            // Informuje GC że finalizer jest zbędny — obiekt już posprzątany.
            GC.SuppressFinalize(this);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ThrowIfShapeMismatch(FastMatrix<T> other)
        {
            if (other.Rows != Rows || other.Cols != Cols)
            {
                throw new ArgumentException($"Shape mismatch: expected {Rows}×{Cols}, got {other.Rows}×{other.Cols}.", nameof(other));
            }
        }
    }
}