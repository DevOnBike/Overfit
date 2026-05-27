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

        
        // ====================================================================
        // TRANSPOSE (last two axes)
        // ====================================================================

        /// <summary>
        /// Swaps the last two axes: rank-2 <c>[X, Y] → [Y, X]</c> or rank-3 <c>[B, X, Y] → [B, Y, X]</c>.
        /// Unlike <see cref="Reshape"/> (a view), this reorders data, so it allocates a fresh node and
        /// copies. The "map-to-sequence" step of a CRNN: conv <c>[B, C·H, W]</c> → <c>[B, W, C·H]</c>.
        /// </summary>
        public static AutogradNode TransposeLastTwo(ComputationGraph? graph, AutogradNode input)
        {
            ArgumentNullException.ThrowIfNull(input);

            var rank = input.Shape.Rank;
            if (rank is not (2 or 3))
            {
                throw new ArgumentOutOfRangeException(nameof(input), "TransposeLastTwo supports rank 2 or 3.");
            }

            GetDims(input.Shape, out var b, out var x, out var y);
            var outShape = rank == 2 ? new TensorShape(y, x) : new TensorShape(b, y, x);

            var output = AllocateNode(graph, outShape, input.RequiresGrad, clearMemory: false);
            var src = input.DataView.AsReadOnlySpan();
            var dst = output.DataView.AsSpan();

            for (var bi = 0; bi < b; bi++)
            {
                var inBatch = bi * x * y;
                var outBatch = bi * y * x;
                for (var xi = 0; xi < x; xi++)
                {
                    for (var yi = 0; yi < y; yi++)
                    {
                        dst[outBatch + yi * x + xi] = src[inBatch + xi * y + yi];
                    }
                }
            }

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.TransposeLastTwo, output, input);
            }

            return output;
        }

        public static void TransposeLastTwoBackward(AutogradNode input, AutogradNode output)
        {
            GetDims(input.Shape, out var b, out var x, out var y);
            var ig = input.GradView.AsSpan();
            var og = output.GradView.AsReadOnlySpan();

            for (var bi = 0; bi < b; bi++)
            {
                var inBatch = bi * x * y;
                var outBatch = bi * y * x;
                for (var xi = 0; xi < x; xi++)
                {
                    for (var yi = 0; yi < y; yi++)
                    {
                        ig[inBatch + xi * y + yi] += og[outBatch + yi * x + xi];
                    }
                }
            }
        }

        private static void GetDims(TensorShape shape, out int b, out int x, out int y)
        {
            if (shape.Rank == 2)
            {
                b = 1;
                x = shape.D0;
                y = shape.D1;
            }
            else
            {
                b = shape.D0;
                x = shape.D1;
                y = shape.D2;
            }
        }
    }
}