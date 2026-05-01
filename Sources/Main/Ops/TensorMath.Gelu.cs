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
        // GELU constants
        private const float GELUSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/π)
        private const float GELUCoeff       = 0.044715f;

        /// <summary>
        /// GELU (Gaussian Error Linear Unit) activation.
        ///
        /// Used in GPT-1/2 FFN layers instead of ReLU.
        /// Formula (tanh approximation, as used in OpenAI's GPT):
        ///   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        ///
        /// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        /// The tanh approximation is faster and used in practice.
        /// </summary>
        public static AutogradNode Gelu(ComputationGraph graph, AutogradNode input)
        {
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);
            var inS    = input.DataView.AsReadOnlySpan();
            var outS   = output.DataView.AsSpan();

            for (var i = 0; i < inS.Length; i++)
            {
                outS[i] = GeluScalar(inS[i]);
            }

            if (input.RequiresGrad)
            {
                graph?.Record(OpCode.Gelu, output, input);
            }

            return output;
        }

        /// <summary>
        /// GELU backward.
        /// d/dx GELU(x) = 0.5 * tanh(k*(x + c*x³))
        ///              + 0.5 * x * (1 - tanh²(k*(x + c*x³))) * k * (1 + 3*c*x²)
        /// where k = sqrt(2/π), c = 0.044715
        /// </summary>
        public static void GeluBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var inS  = input.DataView.AsReadOnlySpan();
            var dOut = output.GradView.AsReadOnlySpan();
            var dIn  = input.GradView.AsSpan();

            for (var i = 0; i < inS.Length; i++)
            {
                dIn[i] += GeluGradScalar(inS[i]) * dOut[i];
            }
        }

        [System.Runtime.CompilerServices.MethodImpl(
            System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal static float GeluScalar(float x)
        {
            var inner = GELUSqrt2OverPi * (x + GELUCoeff * x * x * x);
            return 0.5f * x * (1f + MathF.Tanh(inner));
        }

        [System.Runtime.CompilerServices.MethodImpl(
            System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static float GeluGradScalar(float x)
        {
            var inner  = GELUSqrt2OverPi * (x + GELUCoeff * x * x * x);
            var tanhV  = MathF.Tanh(inner);
            var dtanh  = 1f - tanhV * tanhV; // sech²
            var dinner = GELUSqrt2OverPi * (1f + 3f * GELUCoeff * x * x);
            return 0.5f * (1f + tanhV) + 0.5f * x * dtanh * dinner;
        }
    }
}
