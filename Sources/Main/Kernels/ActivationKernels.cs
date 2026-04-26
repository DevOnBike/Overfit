// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Kernels
{
    internal static class ActivationKernels
    {
        public static void Relu(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            if (output.Length < input.Length)
            {
                throw new ArgumentException(
                    "Output span is too small for ReLU.",
                    nameof(output));
            }

            TensorPrimitives.Max(
                input,
                0f,
                output);
        }

        public static void ReluInPlace(
            Span<float> values)
        {
            TensorPrimitives.Max(
                values,
                0f,
                values);
        }
    }
}
