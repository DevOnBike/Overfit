using System;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// Stack-only lens over raw tensor memory.
    /// Combines base span, shape, strides and offset without owning memory.
    /// Supports up to 4 dimensions.
    /// </summary>
    public readonly ref struct TensorView<T> where T : unmanaged
    {
        private readonly Span<T> _data;

        public readonly TensorShape Shape;
        public readonly TensorStrides Strides;
        public readonly int Offset;

        public int Rank
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Shape.Rank;
        }

        public int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Shape.Size;
        }

        public bool IsContiguous
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Strides.IsContiguous(Shape);
        }

        /// <summary>
        /// Creates a contiguous tensor view over the supplied span.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView(Span<T> data, TensorShape shape)
        {
            if (!shape.IsValid)
            {
                throw new ArgumentException("Shape is invalid.", nameof(shape));
            }

            if (data.Length < shape.Size)
            {
                throw new ArgumentException(
                    $"Span length {data.Length} is smaller than required tensor size {shape.Size}.",
                    nameof(data));
            }

            _data = data;
            Shape = shape;
            Strides = TensorStrides.Contiguous(shape);
            Offset = 0;
        }

        /// <summary>
        /// Creates a view with explicitly supplied shape, strides and base offset.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TensorView(Span<T> data, TensorShape shape, TensorStrides strides, int offset)
        {
            if (!shape.IsValid)
            {
                throw new ArgumentException("Shape is invalid.", nameof(shape));
            }

            if (offset < 0 || offset > data.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(offset));
            }

            _data = data;
            Shape = shape;
            Strides = strides;
            Offset = offset;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException(
                    "Tensor view is not contiguous. Materialize it first or use indexed access.");
            }

            return _data.Slice(Offset, Size);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan() => AsSpan();


        public ref T this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                ValidateIndex(i);
                return ref _data[Offset + Strides.GetOffset(i)];
            }
        }


        public ref T this[int i, int j]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                ValidateIndex(i, j);
                return ref _data[Offset + Strides.GetOffset(i, j)];
            }
        }


        public ref T this[int i, int j, int k]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                ValidateIndex(i, j, k);
                return ref _data[Offset + Strides.GetOffset(i, j, k)];
            }
        }

        public ref T this[int i, int j, int k, int l]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                ValidateIndex(i, j, k, l);
                return ref _data[Offset + Strides.GetOffset(i, j, k, l)];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView<T> Reshape(TensorShape newShape)
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Cannot reshape a non-contiguous tensor view.");
            }

            if (!newShape.IsValid)
            {
                throw new ArgumentException("New shape is invalid.", nameof(newShape));
            }

            if (newShape.Size != Size)
            {
                throw new ArgumentException(
                    $"New shape size {newShape.Size} does not match current size {Size}.",
                    nameof(newShape));
            }

            return new TensorView<T>(_data.Slice(Offset, Size), newShape);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView<T> Transpose2D()
        {
            if (Rank != 2)
            {
                throw new InvalidOperationException("Transpose2D is valid only for rank-2 tensors.");
            }

            var newShape = new TensorShape(Shape.D1, Shape.D0);
            var newStrides = Strides.Transpose2D();
            return new TensorView<T>(_data, newShape, newStrides, Offset);
        }

        /// <summary>
        /// Returns a contiguous 1D slice over the current contiguous view.
        /// This is a linear slice, not a general N-dimensional slice.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView<T> SliceContiguous1D(int offsetIndex, int sliceLength)
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("SliceContiguous1D requires a contiguous tensor view.");
            }

            if (offsetIndex < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(offsetIndex));
            }

            if (sliceLength < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(sliceLength));
            }

            if (offsetIndex + sliceLength > Size)
            {
                throw new ArgumentOutOfRangeException(nameof(sliceLength));
            }

            var slicedSpan = _data.Slice(Offset + offsetIndex, sliceLength);

            return new TensorView<T>(slicedSpan, new TensorShape(sliceLength));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorView<T> Flatten()
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Flatten requires a contiguous tensor view.");
            }

            return new TensorView<T>(_data.Slice(Offset, Size), new TensorShape(Size));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetDim(int index) => Shape[index];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetStride(int index) => Strides[index];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateIndex(int i)
        {
            if ((uint)i >= (uint)Shape.D0)
            {
                throw new ArgumentOutOfRangeException(nameof(i));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateIndex(int i, int j)
        {
            if ((uint)i >= (uint)Shape.D0)
            {
                throw new ArgumentOutOfRangeException(nameof(i));
            }

            if ((uint)j >= (uint)Shape.D1)
            {
                throw new ArgumentOutOfRangeException(nameof(j));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateIndex(int i, int j, int k)
        {
            if ((uint)i >= (uint)Shape.D0)
            {
                throw new ArgumentOutOfRangeException(nameof(i));
            }

            if ((uint)j >= (uint)Shape.D1)
            {
                throw new ArgumentOutOfRangeException(nameof(j));
            }

            if ((uint)k >= (uint)Shape.D2)
            {
                throw new ArgumentOutOfRangeException(nameof(k));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateIndex(int i, int j, int k, int l)
        {
            if ((uint)i >= (uint)Shape.D0)
            {
                throw new ArgumentOutOfRangeException(nameof(i));
            }

            if ((uint)j >= (uint)Shape.D1)
            {
                throw new ArgumentOutOfRangeException(nameof(j));
            }

            if ((uint)k >= (uint)Shape.D2)
            {
                throw new ArgumentOutOfRangeException(nameof(k));
            }

            if ((uint)l >= (uint)Shape.D3)
            {
                throw new ArgumentOutOfRangeException(nameof(l));
            }
        }
    }
}