// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        /// <summary>
        /// Swaps the first two axes of a rank-3 tensor: <c>[A, B, C] → [B, A, C]</c>, keeping the
        /// innermost axis contiguous. This is the token-major ↔ head-major reshuffle attention needs:
        /// projections produce <c>[T, H, dHead]</c> (token-major, all heads packed per token), but the
        /// per-head SDPA kernel wants each head contiguous, <c>[H, T, dHead]</c>.
        ///
        /// A pure permutation — backward transposes the upstream gradient straight back
        /// (<c>dIn[a,b,·] = dOut[b,a,·]</c>), exact, no tolerance.
        /// </summary>
        public static AutogradNode Transpose01(ComputationGraph graph, AutogradNode input)
        {
            var a = input.Shape.D0;
            var b = input.Shape.D1;
            var c = input.Shape.D2;
            if (input.Shape.D3 != 1)
            {
                throw new ArgumentException($"Transpose01 expects a rank-3 tensor, got shape {input.Shape}.");
            }

            var output = AllocateNode(graph, new TensorShape(b, a, c), input.RequiresGrad, clearMemory: false);
            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            for (var i = 0; i < a; i++)
            {
                for (var j = 0; j < b; j++)
                {
                    inS.Slice((i * b + j) * c, c).CopyTo(outS.Slice((j * a + i) * c, c));
                }
            }

            if (input.RequiresGrad)
            {
                graph?.Record(OpCode.Transpose01, output, input);
            }

            return output;
        }

        /// <summary>Transpose-0/1 backward: scatter the upstream gradient back through the same axis
        /// swap, accumulating into <c>input.Grad</c>.</summary>
        public static void Transpose01Backward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var a = input.Shape.D0;
            var b = input.Shape.D1;
            var c = input.Shape.D2;

            var dOut = output.GradView.AsReadOnlySpan();
            var dIn = input.GradView.AsSpan();

            for (var i = 0; i < a; i++)
            {
                for (var j = 0; j < b; j++)
                {
                    var dInBlock = dIn.Slice((i * b + j) * c, c);
                    TensorPrimitives.Add(dInBlock, dOut.Slice((j * a + i) * c, c), dInBlock);
                }
            }
        }
    }
}
