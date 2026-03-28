using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit
{
    public sealed class FastMatrix : IDisposable
    {
        private double[] _data;
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
            _data = ArrayPool<double>.Shared.Rent(_size);

            _data.AsSpan(0, _size).Clear();
        }

        public ref double this[int row, int col]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _data[row * Cols + col];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<double> Row(int row)
        {
            return _data.AsSpan(row * Cols, Cols);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<double> ReadOnlyRow(int row)
        {
            return _data.AsSpan(row * Cols, Cols);
        }

        public Span<double> AsSpan()
        {
            return _data.AsSpan(0, _size);
        }

        public ReadOnlySpan<double> AsReadOnlySpan()
        {
            return _data.AsSpan(0, _size);
        }

        /// <summary>Eksponuje macierz jako TensorSpan do operacji zapis/odczyt. Zero alokacji.</summary>
        public TensorSpan<double> AsTensor()
        {
            return new(AsSpan(), [Rows, Cols], default);
        }

        /// <summary>Eksponuje macierz jako ReadOnlyTensorSpan do bezpiecznych operacji odczytu. Zero alokacji.</summary>
        public ReadOnlyTensorSpan<double> AsReadOnlyTensor()
        {
            return new(AsReadOnlySpan(), [Rows, Cols], default);
        }

        public void Clear()
        {
            AsSpan().Clear();
        }

        // ── SIMD operacje (TensorPrimitives .NET 10) ─────────────────────────

        /// <summary>this += other (in-place, element-wise). SIMD-accelerated.</summary>
        public void Add(FastMatrix other)
        {
            ThrowIfShapeMismatch(other);

            TensorPrimitives.Add(AsReadOnlySpan(), other.AsReadOnlySpan(), AsSpan());
        }

        /// <summary>this -= other (in-place, element-wise). SIMD-accelerated.</summary>
        public void Subtract(FastMatrix other)
        {
            ThrowIfShapeMismatch(other);

            TensorPrimitives.Subtract(AsReadOnlySpan(), other.AsReadOnlySpan(), AsSpan());
        }

        /// <summary>this *= scalar (in-place). SIMD-accelerated.</summary>
        public void MultiplyScalar(double scalar)
        {
            TensorPrimitives.Multiply(AsReadOnlySpan(), scalar, AsSpan());
        }

        /// <summary>Suma kwadratów wszystkich elementów (‖A‖_F²). Używane do Mahalanobis distance.</summary>
        public double SumOfSquares()
        {
            return TensorPrimitives.Dot(AsReadOnlySpan(), AsReadOnlySpan());
        }

        /// <summary>Norma Frobeniusa ‖A‖_F = sqrt(Σ aᵢⱼ²). SIMD-accelerated.</summary>
        public double FrobeniusNorm()
        {
            return TensorPrimitives.Norm(AsReadOnlySpan());
        }

        /// <summary>Softmax in-place na płaskiej reprezentacji. Używane w normalizacji gamma w Baum-Welch.</summary>
        public void Softmax()
        {
            TensorPrimitives.SoftMax(AsReadOnlySpan(), AsSpan());
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
                ArrayPool<double>.Shared.Return(rented);
            }

            GC.SuppressFinalize(this);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ThrowIfShapeMismatch(FastMatrix other)
        {
            if (other.Rows != Rows || other.Cols != Cols)
            {
                throw new ArgumentException($"Shape mismatch: expected {Rows}×{Cols}, got {other.Rows}×{other.Cols}.", nameof(other));
            }
        }
    }
}