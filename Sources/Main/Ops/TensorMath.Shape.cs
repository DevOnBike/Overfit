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
        // ====================================================================
        // RESHAPE
        // ====================================================================

        public static AutogradNode Reshape(ComputationGraph graph, AutogradNode input, params int[] newShape)
        {
            var totalNewElements = 1;
            foreach (var dim in newShape)
            {
                totalNewElements *= dim;
            }

            var newView = input.DataView.Reshape(newShape[0], totalNewElements / newShape[0]);

            // TODO: [ARCHITECTURAL CHANGE REQUIRED] Należy podmienić na bezalokacyjne Aliasowanie, np. FastTensor.Alias(newView)
            var resD = FastTensor<float>.FromView(newView);

            var output = new AutogradNode(resD, input.RequiresGrad);
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
