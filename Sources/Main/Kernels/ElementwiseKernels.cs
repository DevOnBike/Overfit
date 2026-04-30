// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Kernels
{
    /// <summary>
    /// Span-only element-wise kernels for the float32 hot path.
    ///
    /// Design rules:
    /// - no allocations
    /// - no LINQ
    /// - scalar path for very small spans
    /// - TensorPrimitives path for larger spans where vectorization can pay off
    ///
    /// This class is intentionally float-only. Do not generalize to INumber&lt;T&gt;
    /// until the tensor ownership cleanup is complete.
    /// </summary>
    internal static class ElementwiseKernels
    {
        private const int TensorPrimitivesThreshold = 32;

        public static void Add(
            ReadOnlySpan<float> left,
            ReadOnlySpan<float> right,
            Span<float> destination)
        {
            ValidateBinary(
                left,
                right,
                destination,
                nameof(Add));

            if (left.Length < TensorPrimitivesThreshold)
            {
                for (var i = 0; i < left.Length; i++)
                {
                    destination[i] = left[i] + right[i];
                }

                return;
            }

            TensorPrimitives.Add(
                left,
                right,
                destination);
        }

        public static void Subtract(
            ReadOnlySpan<float> left,
            ReadOnlySpan<float> right,
            Span<float> destination)
        {
            ValidateBinary(
                left,
                right,
                destination,
                nameof(Subtract));

            if (left.Length < TensorPrimitivesThreshold)
            {
                for (var i = 0; i < left.Length; i++)
                {
                    destination[i] = left[i] - right[i];
                }

                return;
            }

            TensorPrimitives.Subtract(
                left,
                right,
                destination);
        }

        public static void Multiply(
            ReadOnlySpan<float> left,
            ReadOnlySpan<float> right,
            Span<float> destination)
        {
            ValidateBinary(
                left,
                right,
                destination,
                nameof(Multiply));

            if (left.Length < TensorPrimitivesThreshold)
            {
                for (var i = 0; i < left.Length; i++)
                {
                    destination[i] = left[i] * right[i];
                }

                return;
            }

            TensorPrimitives.Multiply(
                left,
                right,
                destination);
        }

        public static void Multiply(
            ReadOnlySpan<float> input,
            float scalar,
            Span<float> destination)
        {
            ValidateUnary(
                input,
                destination,
                nameof(Multiply));

            if (input.Length < TensorPrimitivesThreshold)
            {
                for (var i = 0; i < input.Length; i++)
                {
                    destination[i] = input[i] * scalar;
                }

                return;
            }

            TensorPrimitives.Multiply(
                input,
                scalar,
                destination);
        }

        /// <summary>
        /// Computes destination = input * scalar + addend.
        /// </summary>
        public static void MultiplyAdd(
            ReadOnlySpan<float> input,
            float scalar,
            ReadOnlySpan<float> addend,
            Span<float> destination)
        {
            ValidateBinary(
                input,
                addend,
                destination,
                nameof(MultiplyAdd));

            if (input.Length < TensorPrimitivesThreshold)
            {
                for (var i = 0; i < input.Length; i++)
                {
                    destination[i] = (input[i] * scalar) + addend[i];
                }

                return;
            }

            TensorPrimitives.MultiplyAdd(
                input,
                scalar,
                addend,
                destination);
        }

        public static void ReLU(
            ReadOnlySpan<float> input,
            Span<float> destination)
        {
            ValidateUnary(
                input,
                destination,
                nameof(ReLU));

            for (var i = 0; i < input.Length; i++)
            {
                var value = input[i];
                destination[i] = value > 0f ? value : 0f;
            }
        }

        public static void ReLUBackward(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> outputGradient,
            Span<float> inputGradient)
        {
            ValidateBinary(
                input,
                outputGradient,
                inputGradient,
                nameof(ReLUBackward));

            for (var i = 0; i < input.Length; i++)
            {
                inputGradient[i] += input[i] > 0f ? outputGradient[i] : 0f;
            }
        }

        public static void Sigmoid(
            ReadOnlySpan<float> input,
            Span<float> destination)
        {
            ValidateUnary(
                input,
                destination,
                nameof(Sigmoid));

            for (var i = 0; i < input.Length; i++)
            {
                destination[i] = 1f / (1f + MathF.Exp(-input[i]));
            }
        }

        public static void Tanh(
            ReadOnlySpan<float> input,
            Span<float> destination)
        {
            ValidateUnary(
                input,
                destination,
                nameof(Tanh));

            for (var i = 0; i < input.Length; i++)
            {
                destination[i] = MathF.Tanh(input[i]);
            }
        }

        public static float Dot(
            ReadOnlySpan<float> left,
            ReadOnlySpan<float> right)
        {
            if (left.Length != right.Length)
            {
                throw new ArgumentException(
                    $"{nameof(Dot)} length mismatch: left={left.Length}, right={right.Length}.");
            }

            if (left.Length == 0)
            {
                return 0f;
            }

            if (left.Length < TensorPrimitivesThreshold)
            {
                var sum = 0f;

                for (var i = 0; i < left.Length; i++)
                {
                    sum += left[i] * right[i];
                }

                return sum;
            }

            return TensorPrimitives.Dot(
                left,
                right);
        }

        public static float SumOfSquares(
            ReadOnlySpan<float> input)
        {
            if (input.Length == 0)
            {
                return 0f;
            }

            if (input.Length < TensorPrimitivesThreshold)
            {
                var sum = 0f;

                for (var i = 0; i < input.Length; i++)
                {
                    var value = input[i];
                    sum += value * value;
                }

                return sum;
            }

            return TensorPrimitives.SumOfSquares(input);
        }

        public static float MeanSquaredError(
            ReadOnlySpan<float> prediction,
            ReadOnlySpan<float> target)
        {
            if (prediction.Length != target.Length)
            {
                throw new ArgumentException(
                    $"{nameof(MeanSquaredError)} length mismatch: prediction={prediction.Length}, target={target.Length}.");
            }

            if (prediction.Length == 0)
            {
                throw new ArgumentException(
                    "MeanSquaredError requires at least one element.",
                    nameof(prediction));
            }

            var sum = 0f;

            for (var i = 0; i < prediction.Length; i++)
            {
                var diff = prediction[i] - target[i];
                sum += diff * diff;
            }

            return sum / prediction.Length;
        }

        public static void Fill(
            Span<float> destination,
            float value)
        {
            destination.Fill(value);
        }

        public static void Clear(
            Span<float> destination)
        {
            destination.Clear();
        }

        private static void ValidateUnary(
            ReadOnlySpan<float> input,
            Span<float> destination,
            string operation)
        {
            if (destination.Length < input.Length)
            {
                throw new ArgumentException(
                    $"{operation} destination span is too small. Expected at least {input.Length}, got {destination.Length}.",
                    nameof(destination));
            }
        }

        private static void ValidateBinary(
            ReadOnlySpan<float> left,
            ReadOnlySpan<float> right,
            Span<float> destination,
            string operation)
        {
            if (left.Length != right.Length)
            {
                throw new ArgumentException(
                    $"{operation} length mismatch: left={left.Length}, right={right.Length}.");
            }

            if (destination.Length < left.Length)
            {
                throw new ArgumentException(
                    $"{operation} destination span is too small. Expected at least {left.Length}, got {destination.Length}.",
                    nameof(destination));
            }
        }
    }
}
