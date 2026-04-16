using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// Immutable, stack-allocated representation of tensor strides.
    /// </summary>
    public readonly record struct TensorStrides
    {
        public readonly int S0;
        public readonly int S1;
        public readonly int S2;
        public readonly int S3;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorStrides(int s0, int s1 = 0, int s2 = 0, int s3 = 0)
        {
            S0 = s0;
            S1 = s1;
            S2 = s2;
            S3 = s3;
        }

        public int this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return index switch
                {
                    0 => S0,
                    1 => S1,
                    2 => S2,
                    3 => S3,
                    _ => 0
                };
            }
        }

        /// <summary>
        /// Computes contiguous row-major strides for the given shape.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TensorStrides Contiguous(TensorShape shape)
        {
            return shape.Rank switch
            {
                1 => new TensorStrides(1),
                2 => new TensorStrides(shape.D1, 1),
                3 => new TensorStrides(shape.D1 * shape.D2, shape.D2, 1),
                4 => new TensorStrides(shape.D1 * shape.D2 * shape.D3, shape.D2 * shape.D3, shape.D3, 1),

                _ => throw new NotSupportedException($"Unsupported rank: {shape.Rank}")
            };
        }

        /// <summary>
        /// Returns strides for a transposed 2D view (swaps S0 and S1).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorStrides Transpose2D()
        {
            return new TensorStrides(S1, S0, S2, S3);
        }

        /// <summary>
        /// Checks if these strides represent contiguous memory for the given shape.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsContiguous(TensorShape shape)
        {
            var expected = Contiguous(shape);

            return S0 == expected.S0 && S1 == expected.S1 && S2 == expected.S2 && S3 == expected.S3;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetOffset(int i0)
        {
            return i0 * S0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetOffset(int i0, int i1)
        {
            return i0 * S0 + i1 * S1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetOffset(int i0, int i1, int i2)
        {
            return i0 * S0 + i1 * S1 + i2 * S2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetOffset(int i0, int i1, int i2, int i3)
        {
            return i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3;
        }

        public override string ToString()
        {
            return $"Strides({S0}, {S1}, {S2}, {S3})";
        }
    }
}