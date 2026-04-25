// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// Stack-only mutable lens over tensor memory.
    /// Does not own memory; only describes how to interpret it via shape, strides and offset.
    /// Supports up to 4 dimensions.
    /// </summary>
    public readonly ref struct TensorSpan<T> where T : unmanaged
    {
        private readonly Span<T> _data;

        public readonly TensorShape Shape;
        public readonly TensorStrides Strides;
        public readonly int Offset;

        // HPC OPTIMIZATION: Cached at creation to avoid recalculating on every Span access
        public readonly bool IsContiguous;

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

        public bool IsEmpty
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Size == 0;
        }

        /// <summary>
        /// Creates a contiguous tensor span over the supplied span.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorSpan(Span<T> data, TensorShape shape)
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

            // Ponieważ używamy metody Contiguous, wiemy na 100%, że pamięć jest ciągła
            IsContiguous = true;
        }

        /// <summary>
        /// Creates a tensor span with explicitly supplied shape, strides and base offset.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TensorSpan(Span<T> data, TensorShape shape, TensorStrides strides, int offset)
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

            // Obliczane tylko raz przy tworzeniu niestandardowego widoku (np. przy Transpose)
            IsContiguous = strides.IsContiguous(shape);
        }

        /// <summary>
        /// Returns a contiguous span over the tensor contents.
        /// Valid only for contiguous tensor spans.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Tensor span is not contiguous. Materialize it first or use indexed access.");
            }

            return _data.Slice(Offset, Size);
        }

        /// <summary>
        /// Returns a contiguous readonly span over the tensor contents.
        /// Valid only for contiguous tensor spans.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            return AsSpan();
        }

        /// <summary>
        /// Tries to get a contiguous span view over the tensor contents.
        /// Returns false for non-contiguous tensor spans.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetSpan(out Span<T> span)
        {
            if (IsContiguous)
            {
                span = _data.Slice(Offset, Size);
                return true;
            }

            span = default;
            return false;
        }

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

        /// <summary>
        /// Reinterprets the current tensor span with a new shape.
        /// Valid only for contiguous tensor spans.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorSpan<T> Reshape(TensorShape newShape)
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Cannot reshape a non-contiguous tensor span.");
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

            return new TensorSpan<T>(_data.Slice(Offset, Size), newShape);
        }

        /// <summary>
        /// Returns a transposed 2D view without copying data.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorSpan<T> Transpose2D()
        {
            if (Rank != 2)
            {
                throw new InvalidOperationException("Transpose2D is valid only for rank-2 tensors.");
            }

            var newShape = new TensorShape(Shape.D1, Shape.D0);
            var newStrides = Strides.Transpose2D();

            return new TensorSpan<T>(_data, newShape, newStrides, Offset);
        }

        /// <summary>
        /// Returns a contiguous 1D slice over the current contiguous tensor span.
        /// This is a linear slice, not a general N-dimensional slice.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorSpan<T> SliceContiguous1D(int offsetIndex, int sliceLength)
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("SliceContiguous1D requires a contiguous tensor span.");
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

            return new TensorSpan<T>(slicedSpan, new TensorShape(sliceLength));
        }

        /// <summary>
        /// Flattens the tensor span into a contiguous 1D view.
        /// Valid only for contiguous tensor spans.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorSpan<T> Flatten()
        {
            if (!IsContiguous)
            {
                throw new InvalidOperationException("Flatten requires a contiguous tensor span.");
            }

            return new TensorSpan<T>(_data.Slice(Offset, Size), new TensorShape(Size));
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