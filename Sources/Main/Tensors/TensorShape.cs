using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors
{

    /// <summary>
    /// Immutable, stack-allocated representation of tensor dimensions.
    /// Supports up to 4 dimensions (batch, channels, height, width).
    /// </summary>
    public readonly record struct TensorShape
    {
        public readonly int D0;
        public readonly int D1;
        public readonly int D2;
        public readonly int D3;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorShape(int d0, int d1 = 1, int d2 = 1, int d3 = 1)
        {
            D0 = d0;
            D1 = d1;
            D2 = d2;
            D3 = d3;
        }

        public int Rank
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return D3 > 1 ? 4 :
                    D2 > 1 ? 3 :
                    D1 > 1 ? 2 : 1;
            }
        }

        public int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return D0 * D1 * D2 * D3;
            }
        }

        public int this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return index switch { 0 => D0, 1 => D1, 2 => D2, 3 => D3, _ => 1 };
            }
        }

        public bool IsValid
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return D0 > 0 && D1 > 0 && D2 > 0 && D3 > 0;
            }
        }

        // Implicit conversions
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator TensorShape(int d0)
        {
            return new TensorShape(d0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator TensorShape((int d0, int d1) dims)
        {
            return new TensorShape(dims.d0, dims.d1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator TensorShape((int d0, int d1, int d2) dims)
        {
            return new TensorShape(dims.d0, dims.d1, dims.d2);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator TensorShape((int d0, int d1, int d2, int d3) dims)
        {
            return new TensorShape(dims.d0, dims.d1, dims.d2, dims.d3);
        }

        // Shape operations
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorShape WithD0(int newD0)
        {
            return new TensorShape(newD0, D1, D2, D3);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public TensorShape Flatten2D()
        {
            return new TensorShape(D0, D1 * D2 * D3);
        }

        public override string ToString()
        {
            return Rank switch
            {
                1 => $"({D0})",
                2 => $"({D0}, {D1})",
                3 => $"({D0}, {D1}, {D2})",
                4 => $"({D0}, {D1}, {D2}, {D3})",
                _ => $"({D0}, {D1}, {D2}, {D3})"
            };
        }

        // Factory methods
        public static TensorShape Scalar => new(1);

        public static TensorShape Vector(int length)
        {
            return new TensorShape(length);
        }
        public static TensorShape Matrix(int rows, int cols)
        {
            return new TensorShape(rows, cols);
        }
    }
}