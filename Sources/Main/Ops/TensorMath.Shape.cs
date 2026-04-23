// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

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

            // Otrzymujemy widok o zmienionym kształcie (tylko matematyka na int-ach)
            var newView = input.DataView.Reshape(new TensorShape(newShape[0], totalNewElements / newShape[0]));

            // ROZWIĄZANIE ZERO-ALLOC! Zamiast kopiować i robić FastTensor.FromView, 
            // materiaizujemy nową logikę do pamięci w locie z zachowaniem DOD.
            var storage = TensorFactory.Materialize(newView);
            var output = new AutogradNode(storage, newView.Shape, input.RequiresGrad);

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