// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // RESHAPE
        // ====================================================================

        public static AutogradNode Reshape(ComputationGraph? graph, AutogradNode input, params int[] newShape)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(newShape);

            if (newShape.Length is < 1 or > 4)
            {
                throw new ArgumentOutOfRangeException(nameof(newShape), "Reshape supports rank 1..4.");
            }

            var totalNewElements = 1;

            for (var i = 0; i < newShape.Length; i++)
            {
                if (newShape[i] <= 0)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(newShape),
                        $"Invalid dimension at index {i}: {newShape[i]}.");
                }

                checked
                {
                    totalNewElements *= newShape[i];
                }
            }

            if (totalNewElements != input.Shape.Size)
            {
                throw new ArgumentException(
                    $"New shape size {totalNewElements} does not match input size {input.Shape.Size}.",
                    nameof(newShape));
            }

            var outputShape = newShape.Length switch
            {
                1 => new TensorShape(newShape[0]),
                2 => new TensorShape(newShape[0], newShape[1]),
                3 => new TensorShape(newShape[0], newShape[1], newShape[2]),
                4 => new TensorShape(newShape[0], newShape[1], newShape[2], newShape[3]),
                _ => throw new UnreachableException()
            };

            // Validation: TensorSpan.Reshape already rejects non-contiguous input.
            _ = input.DataView.Reshape(outputShape);

            var output = AutogradNode.ViewOf(input, outputShape, input.RequiresGrad);

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Reshape, output, input);
            }

            return output;
        }

        public static void ReshapeBackward(AutogradNode input, AutogradNode output)
        {
            TensorPrimitives.Add(input.GradView.AsSpan(), output.GradView.AsReadOnlySpan(), input.GradView.AsSpan());
        }
    }
}