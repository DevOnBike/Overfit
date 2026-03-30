using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
namespace DevOnBike.Overfit.Core
{
    public sealed class FastMatrix<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
    {
        private T[] _data;
        private readonly int _size;
        private bool _disposed;

        public int Cols { get; }
        public int Rows { get; }

        public FastMatrix(int rows, int cols)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);

            Rows = rows;
            Cols = cols;

            _size = checked(rows * cols);
            _data = ArrayPool<T>.Shared.Rent(_size);

            _data.AsSpan(0, _size).Clear();
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
            return _data!.AsSpan(0, _size);
        }

        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _data!.AsSpan(0, _size);
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
            if (source.Length != _size)
            {
                throw new ArgumentException($"Expected {_size} elements, got {source.Length}.");
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
            
            for (var r = 0; r < Rows; r++)
            {
                for (var c = 0; c < Cols; c++)
                {
                    result[r, c] = this[r, c];
                }
            }
            
            return result;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, true))
            {
                return;
            }

            var rented = Interlocked.Exchange(ref _data, null);

            if (rented != null)
            {
                ArrayPool<T>.Shared.Return(rented);
            }

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