using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// PURE DOD: Czyste funkcje statyczne. Zero alokacji, zero stanów obiektowych.
    /// Przyjmują wyłącznie Spany z pamięcią do przetworzenia.
    /// </summary>
    public static class TensorKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddInPlace(Span<float> target, ReadOnlySpan<float> source)
        {
            TensorPrimitives.Add(target, source, target);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
        {
            TensorPrimitives.Add(left, right, destination);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination)
        {
            TensorPrimitives.Multiply(left, right, destination);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Relu(ReadOnlySpan<float> source, Span<float> destination)
        {
            TensorPrimitives.Max(source, 0f, destination);
        }

        // Itd... w to miejsce w przyszłości wejdą Twoje uniwersalne "Operator Structs"
    }
}