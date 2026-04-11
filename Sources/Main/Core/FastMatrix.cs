// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    ///     Represents a high-performance 2D matrix structure optimized for numerical computing and machine learning.
    /// </summary>
    public class FastMatrix<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
    {
        private T[] _data;
        private bool _disposed;

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

        public int Cols { get; }
        public int Rows { get; }
        public int Size { get; }

        public ref T this[int row, int col]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return ref _data![row * Cols + col];
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
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
            return new TensorSpan<T>(AsSpan(), [Rows, Cols], default);
        }

        public ReadOnlyTensorSpan<T> AsReadOnlyTensor()
        {
            return new ReadOnlyTensorSpan<T>(AsReadOnlySpan(), [Rows, Cols], default);
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

        public T SumOfSquares()
        {
            return TensorPrimitives.Dot(AsReadOnlySpan(), AsReadOnlySpan());
        }

        public T FrobeniusNorm()
        {
            return TensorPrimitives.Norm(AsReadOnlySpan());
        }

        public void Softmax()
        {
            TensorPrimitives.SoftMax(AsReadOnlySpan(), AsSpan());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FastMatrixView<T> AsView()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            return new FastMatrixView<T>(AsSpan(), Rows, Cols, Cols, 1, 0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FastMatrixView<T> Transpose()
        {
            return AsView().Transpose();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public FastMatrixView<T> Slice(int startRow, int startCol, int rows, int cols)
        {
            return AsView().Slice(startRow, startCol, rows, cols);
        }

        /// <summary>
        ///     Materializes the view into a new, contiguous FastMatrix.
        ///     Essential for Autograd when a transposed view needs to be the right-hand operand in SIMD GEMM.
        ///     Caller is responsible for disposing the returned FastMatrix.
        /// </summary>
        public FastMatrix<T> ToContiguousFastMatrix()
        {
            var result = new FastMatrix<T>(Rows, Cols);

            AsReadOnlySpan().CopyTo(result.AsSpan());

            return result;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (Interlocked.Exchange(ref _disposed, true))
            {
                return;
            }

            if (disposing)
            {
                var rented = Interlocked.Exchange(ref _data, null!);

                if (rented != null)
                {
                    ArrayPool<T>.Shared.Return(rented);
                }
            }

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