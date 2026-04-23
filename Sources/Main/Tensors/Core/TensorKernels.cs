using System;
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
        public static void AddInPlace(Span<float> target, ReadOnlySpan<float> source)
        {
            TensorKernelGuards.ValidateSameLength(source, target);
            TensorKernelGuards.ValidateInputOutputSpanNonOverlapping(source, target);

            TensorPrimitives.Add(target, source, target);
        }

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
        public static float Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
        {
            TensorKernelGuards.ValidateSameLength(left, right);
            return TensorPrimitives.Dot(left, right);
        }
    }
}