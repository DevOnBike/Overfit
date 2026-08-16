// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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
                return D3 > 1 ? 4 : D2 > 1 ? 3 : D1 > 1 ? 2 : 1;
            }
        }

        /// <summary>
        /// Elements the shape describes.
        ///
        /// <para><b>Deliberately unchecked, and this is a measured decision rather than an oversight — do
        /// not "tidy" it into <c>checked</c>.</b> It was made <c>checked</c> on 2026-08-03 together with
        /// the four other unchecked dimension products in this codebase, and reverted on 2026-08-04 after
        /// the two halves of that change turned out to be different arguments wearing one label.</para>
        ///
        /// <para><b>Where the dimensions come from is the whole distinction.</b>
        /// <c>GgufTensorInfo.ElementCount</c>, <c>OnnxTensor.ElementCount</c> and the ONNX graph importer
        /// multiply numbers read <b>out of a model file</b>, which may be truncated or crafted; a wrapped
        /// product there is small, positive, plausible, and can pass a later shape check while the real
        /// layout disagrees — silently wrong weights. Those three stay checked. This one multiplies
        /// dimensions that came from <b>the caller's own architecture code</b>, and overflowing an
        /// <see cref="int"/> needs a product above 2.1 billion elements: a single 8.6 GB tensor, asked for
        /// by a literal somebody wrote, on hardware that could not allocate it.</para>
        ///
        /// <para><b>Measured, so the trade is on record.</b> <c>TensorGuardCostBenchmark</c> puts the check
        /// at ~0.11 ns per call — 1.34x this property, which is sub-nanosecond, so the ratio is large and
        /// the quantity is not. End-to-end it is invisible: an A/B over a full small-CNN
        /// forward+backward moved 2 of 6 arms the <i>wrong</i> way and up to 18% either way, which is the
        /// benchmark's own run-to-run spread, and the arithmetic bound is ~0.1% even at ten thousand calls
        /// per pass. It was reverted for having no benefit here, not for costing too much.</para>
        ///
        /// <para><b>And it introduced a defect of its own.</b> <c>checked</c> evaluates left to right, so a
        /// shape with large leading dimensions and a zero on the end threw despite a correct result of
        /// zero — a false positive that did not exist before.</para>
        ///
        /// <para>Note that <c>OVERFIT028</c> cannot see this either way: it scans <c>new T[...]</c>
        /// array-creation syntax, so a property getter is invisible to it. That gap is why the file-driven
        /// products above were found by reading rather than by the analyser, and it is recorded as an open
        /// item in <c>ROADMAP.md</c>.</para>
        /// </summary>
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
                return index switch
                {
                    0 => D0,
                    1 => D1,
                    2 => D2,
                    3 => D3,
                    _ => 1
                };
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