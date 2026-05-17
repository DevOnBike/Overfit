// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    internal static class TensorKernelGuards
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateSameLength<T>(
            ReadOnlySpan<T> left,
            ReadOnlySpan<T> right,
            [CallerArgumentExpression(nameof(left))] string leftName = "",
            [CallerArgumentExpression(nameof(right))] string rightName = "")
        {
            if (left.Length != right.Length)
            {
                throw new ArgumentException($"Span lengths must match: {leftName}.Length={left.Length}, {rightName}.Length={right.Length}.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateDestinationLength<T>(
            ReadOnlySpan<T> source,
            Span<T> destination,
            [CallerArgumentExpression(nameof(source))] string sourceName = "",
            [CallerArgumentExpression(nameof(destination))] string destinationName = "")
        {
            if (destination.Length < source.Length)
            {
                throw new ArgumentException($"Destination too short: {sourceName}.Length={source.Length}, {destinationName}.Length={destination.Length}.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateSameLengthAndDestination<T>(
            ReadOnlySpan<T> left,
            ReadOnlySpan<T> right,
            Span<T> destination,
            [CallerArgumentExpression(nameof(left))] string leftName = "",
            [CallerArgumentExpression(nameof(right))] string rightName = "",
            [CallerArgumentExpression(nameof(destination))] string destinationName = "")
        {
            ValidateSameLength(left, right, leftName, rightName);

            if (destination.Length < left.Length)
            {
                throw new ArgumentException($"Destination too short: required={left.Length} (from {leftName}), {destinationName}.Length={destination.Length}.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateInputOutputSpanNonOverlapping<T>(
            ReadOnlySpan<T> input,
            Span<T> output,
            [CallerArgumentExpression(nameof(input))] string inputName = "",
            [CallerArgumentExpression(nameof(output))] string outputName = "")
        {
            if (input.Overlaps(output, out var elementOffset) && elementOffset != 0)
            {
                throw new ArgumentException($"Input and output spans must not overlap unless they start at the same location: {inputName} overlaps {outputName} by {elementOffset} elements.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateInputOutputSpanNonOverlapping<T>(
            ReadOnlySpan<T> left,
            ReadOnlySpan<T> right,
            Span<T> output,
            [CallerArgumentExpression(nameof(left))] string leftName = "",
            [CallerArgumentExpression(nameof(right))] string rightName = "",
            [CallerArgumentExpression(nameof(output))] string outputName = "")
        {
            ValidateInputOutputSpanNonOverlapping(left, output, leftName, outputName);
            ValidateInputOutputSpanNonOverlapping(right, output, rightName, outputName);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateContiguous<T>(TensorSpan<T> span, [CallerArgumentExpression("span")] string paramName = "") where T : unmanaged
        {
            if (!span.IsContiguous)
            {
                throw new InvalidOperationException($"KRYTYCZNY BŁĄD WYDAJNOŚCI: Tensor '{paramName}' nie jest ciągły w pamięci (np. po operacji Transpose). Szybkie kernele SIMD wymagają ciągłej pamięci. Zmaterializuj tensor używając TensorFactory.Materialize().");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ValidateSameShape<T>(
            TensorSpan<T> left,
            TensorSpan<float> right,
            [CallerArgumentExpression(nameof(left))] string leftName = "",
            [CallerArgumentExpression(nameof(right))] string rightName = "")
            where T : unmanaged
        {
            if (left.Size != right.Size) // In the future, full Rank and D0..D3 could be checked
            {
                throw new ArgumentException($"Niezgodność kształtów: {leftName}.Shape={left.Shape} vs {rightName}.Shape={right.Shape}.");
            }
        }
    }
}