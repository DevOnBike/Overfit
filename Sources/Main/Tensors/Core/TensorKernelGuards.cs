using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    internal static class TensorKernelGuards
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateSameLength<T>(
            ReadOnlySpan<T> left,
            ReadOnlySpan<T> right)
        {
            if (left.Length != right.Length)
            {
                throw new ArgumentException($"Span lengths must match. Left={left.Length}, Right={right.Length}.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateDestinationLength<T>(
            ReadOnlySpan<T> source,
            Span<T> destination)
        {
            if (destination.Length < source.Length)
            {
                throw new ArgumentException($"Destination too short. Source={source.Length}, Destination={destination.Length}.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateSameLengthAndDestination<T>(
            ReadOnlySpan<T> left,
            ReadOnlySpan<T> right,
            Span<T> destination)
        {
            ValidateSameLength(left, right);

            if (destination.Length < left.Length)
            {
                throw new ArgumentException($"Destination too short. Required={left.Length}, Destination={destination.Length}.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateInputOutputSpanNonOverlapping<T>(
            ReadOnlySpan<T> input,
            Span<T> output)
        {
            if (input.Overlaps(output, out var elementOffset) && elementOffset != 0)
            {
                throw new ArgumentException("Input and output spans must not overlap unless they start at the same location.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateInputOutputSpanNonOverlapping<T>(
            ReadOnlySpan<T> left,
            ReadOnlySpan<T> right,
            Span<T> output)
        {
            ValidateInputOutputSpanNonOverlapping(left, output);
            ValidateInputOutputSpanNonOverlapping(right, output);
        }
    }
}