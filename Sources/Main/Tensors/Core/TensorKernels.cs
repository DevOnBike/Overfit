// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// Pure DOD: stateless kernels operating directly on spans.
    /// </summary>
    public static class TensorKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(TensorSpan<float> left, TensorSpan<float> right, TensorSpan<float> destination)
        {
            // 1. Fail-Fast (Reject non-contiguous memory)
            TensorKernelGuards.ValidateContiguous(left);
            TensorKernelGuards.ValidateContiguous(right);
            TensorKernelGuards.ValidateContiguous(destination);

            // 2. Shape Validation (Before we descend to raw bytes)
            TensorKernelGuards.ValidateSameShape(left, right);
            TensorKernelGuards.ValidateSameShape(left, destination);

            // 3. Descend to the fastest path
            Add(left.AsReadOnlySpan(), right.AsReadOnlySpan(), destination.AsSpan());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddInPlace(Span<float> target, ReadOnlySpan<float> source)
        {
            TensorKernelGuards.ValidateSameLength(source, target);
            TensorKernelGuards.ValidateInputOutputSpanNonOverlapping(source, target);

            TensorPrimitives.Add(target, source, target);
        }

        /// <summary>
        /// Element-wise sum into a separate destination.
        ///
        /// <para><b>The overlap guard was missing here and present on every sibling</b> — <c>AddInPlace</c>,
        /// the <c>TensorSpan</c> overload above, <c>Multiply</c>, <c>Scale</c> and <c>Relu</c> all had it.
        /// A partially overlapping destination does not fault; <c>TensorPrimitives</c> processes in vector
        /// blocks and simply reads inputs it has already overwritten, so the result is arithmetically wrong
        /// and silent. Wrong numbers out of a kernel do not announce themselves — they arrive as a loss
        /// curve that merely looks worse than it should, which is days of looking in the wrong place.</para>
        ///
        /// <para>Exact aliasing (<c>destination</c> identical to an input) is what <c>AddInPlace</c> is for
        /// and is allowed there; it is only the partial overlap that has no correct answer.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
        {
            TensorKernelGuards.ValidateSameLengthAndDestination(left, right, destination);
            TensorKernelGuards.ValidateInputOutputSpanNonOverlapping(left, right, destination);

            TensorPrimitives.Add(left, right, destination);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
        {
            TensorKernelGuards.ValidateSameLengthAndDestination(left, right, destination);
            TensorKernelGuards.ValidateInputOutputSpanNonOverlapping(left, right, destination);

            TensorPrimitives.Multiply(left, right, destination);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Scale(ReadOnlySpan<float> source, float scalar, Span<float> destination)
        {
            TensorKernelGuards.ValidateDestinationLength(source, destination);
            TensorKernelGuards.ValidateInputOutputSpanNonOverlapping(source, destination);

            TensorPrimitives.Multiply(source, scalar, destination);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Relu(ReadOnlySpan<float> source, Span<float> destination)
        {
            TensorKernelGuards.ValidateDestinationLength(source, destination);
            TensorKernelGuards.ValidateInputOutputSpanNonOverlapping(source, destination);

            TensorPrimitives.Max(source, 0f, destination);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Relu(TensorSpan<float> source, TensorSpan<float> destination)
        {
            TensorKernelGuards.ValidateContiguous(source);
            TensorKernelGuards.ValidateContiguous(destination);
            TensorKernelGuards.ValidateSameShape(source, destination);

            Relu(source.AsReadOnlySpan(), destination.AsSpan());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
        {
            TensorKernelGuards.ValidateSameLength(left, right);
            return TensorPrimitives.Dot(left, right);
        }
    }
}