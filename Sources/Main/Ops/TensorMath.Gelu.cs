// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Runtime;

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
        ///
        /// Parallelized via <see cref="OverfitParallelFor"/> above
        /// <see cref="ParallelThreshold"/> elements. GELU per-element work is
        /// ~10 FLOPs + tanh, so parallel amortizes the dispatch overhead
        /// (~6 µs warm) at modest tensor sizes.
        /// </summary>
        public static AutogradNode Gelu(ComputationGraph graph, AutogradNode input)
        {
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);
            var inS    = input.DataView.AsReadOnlySpan();
            var outS   = output.DataView.AsSpan();

            if (inS.Length < ParallelThreshold)
            {
                for (var i = 0; i < inS.Length; i++)
                {
                    outS[i] = GeluScalar(inS[i]);
                }
            }
            else
            {
                unsafe
                {
                    fixed (float* inPtr = inS, outPtr = outS)
                    {
                        var ctx = new GeluForwardCtx { Input = inPtr, Output = outPtr };
                        OverfitParallelFor.For(0, inS.Length, &GeluForwardChunk, &ctx);
                    }
                }
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
        ///
        /// Parallelized via <see cref="OverfitParallelFor"/> above
        /// <see cref="ParallelThreshold"/>. Backward per-element is ~15 FLOPs
        /// + tanh — heavier than forward, amortizes parallel dispatch even
        /// more easily.
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

            if (inS.Length < ParallelThreshold)
            {
                for (var i = 0; i < inS.Length; i++)
                {
                    dIn[i] += GeluGradScalar(inS[i]) * dOut[i];
                }
            }
            else
            {
                unsafe
                {
                    fixed (float* inPtr = inS, dOutPtr = dOut, dInPtr = dIn)
                    {
                        var ctx = new GeluBackwardCtx
                        {
                            Input = inPtr,
                            GradOutput = dOutPtr,
                            GradInput = dInPtr,
                        };
                        OverfitParallelFor.For(0, inS.Length, &GeluBackwardChunk, &ctx);
                    }
                }
            }
        }

        [MethodImpl(
            MethodImplOptions.AggressiveInlining)]
        internal static float GeluScalar(float x)
        {
            var inner = GELUSqrt2OverPi * (x + GELUCoeff * x * x * x);
            return 0.5f * x * (1f + MathF.Tanh(inner));
        }

        [MethodImpl(
            MethodImplOptions.AggressiveInlining)]
        private static float GeluGradScalar(float x)
        {
            var inner  = GELUSqrt2OverPi * (x + GELUCoeff * x * x * x);
            var tanhV  = MathF.Tanh(inner);
            var dtanh  = 1f - tanhV * tanhV; // sech²
            var dinner = GELUSqrt2OverPi * (1f + 3f * GELUCoeff * x * x);
            return 0.5f * (1f + tanhV) + 0.5f * x * dtanh * dinner;
        }

        // ── OverfitParallelFor chunk bodies + contexts ────────────────────────

        private unsafe struct GeluForwardCtx
        {
            public float* Input;
            public float* Output;
        }

        private static unsafe void GeluForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<GeluForwardCtx>(contextPtr);
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                ctx.Output[i] = GeluScalar(ctx.Input[i]);
            }
        }

        private unsafe struct GeluBackwardCtx
        {
            public float* Input;
            public float* GradOutput;
            public float* GradInput;
        }

        private static unsafe void GeluBackwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<GeluBackwardCtx>(contextPtr);
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                ctx.GradInput[i] += GeluGradScalar(ctx.Input[i]) * ctx.GradOutput[i];
            }
        }
    }
}
