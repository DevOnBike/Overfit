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
            // 1. Fail-Fast (Odrzucamy nieciągłą pamięć)
            TensorKernelGuards.ValidateContiguous(left);
            TensorKernelGuards.ValidateContiguous(right);
            TensorKernelGuards.ValidateContiguous(destination);

            // 2. Walidacja Kształtów (Zanim wejdziemy w surowe bajty)
            TensorKernelGuards.ValidateSameShape(left, right);
            TensorKernelGuards.ValidateSameShape(left, destination);

            // 3. Zejście do najszybszej ścieżki
            Add(left.AsReadOnlySpan(), right.AsReadOnlySpan(), destination.AsSpan());
        }

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