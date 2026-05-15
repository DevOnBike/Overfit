// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // GELU constants (tanh approximation as used in OpenAI's GPT).
        private const float GELUSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/π)
        private const float GELUCoeff       = 0.044715f;

        // Tile size for the SIMD pipeline inside chunk bodies. Two scratch
        // spans of this size live on the worker thread's stack (= 8 KB total
        // for 1024 floats × 2), well within the 1 MB default thread stack.
        // Tile chosen to fit in L1 data cache (32 KB on most modern x86)
        // so the multiple TensorPrimitives passes over the same window stay
        // hot in cache instead of streaming from L2 each time.
        private const int GeluTile = 1024;

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
        /// <para>
        /// <b>Parallelization + SIMD.</b> Two layers of speedup compose here:
        /// </para>
        /// <list type="number">
        ///   <item><b>Outer</b>: chunks dispatched via <see cref="OverfitParallelFor"/>
        ///         above <see cref="ParallelThreshold"/> elements.</item>
        ///   <item><b>Inner</b>: within each chunk, work is processed in
        ///         <see cref="GeluTile"/>-element tiles using
        ///         <see cref="TensorPrimitives"/> (SIMD-backed) for every step
        ///         of the GELU formula — including the expensive <c>tanh</c>
        ///         which <see cref="TensorPrimitives.Tanh(ReadOnlySpan{float}, Span{float})"/>
        ///         vectorizes via polynomial approximation.</item>
        /// </list>
        /// Per element this is ~8-16× faster than the scalar
        /// <c>MathF.Tanh</c> loop the kernel used to run.
        /// </summary>
        public static AutogradNode Gelu(ComputationGraph graph, AutogradNode input)
        {
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);
            var inS    = input.DataView.AsReadOnlySpan();
            var outS   = output.DataView.AsSpan();

            if (inS.Length < ParallelThreshold)
            {
                GeluForwardSimd(inS, outS);
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
        /// d/dx GELU(x) = 0.5 * (1 + tanh(k*(x + c*x³)))
        ///              + 0.5 * x * (1 - tanh²(k*(x + c*x³))) * k * (1 + 3*c*x²)
        /// where k = sqrt(2/π), c = 0.044715.
        ///
        /// Parallelized + SIMD-batched analogous to forward.
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
                GeluBackwardSimd(inS, dOut, dIn);
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

        // ── SIMD core: forward ────────────────────────────────────────────────

        /// <summary>
        /// SIMD-batched GELU forward over the given input/output spans.
        /// Loops over the spans in <see cref="GeluTile"/>-sized tiles with
        /// two stackalloc'd scratch buffers (<c>inner</c>, <c>tanhBuf</c>).
        /// </summary>
        private static void GeluForwardSimd(ReadOnlySpan<float> input, Span<float> output)
        {
            Span<float> innerBuf = stackalloc float[GeluTile];
            Span<float> tanhBuf  = stackalloc float[GeluTile];

            var len = input.Length;
            for (var offset = 0; offset < len; offset += GeluTile)
            {
                var n = Math.Min(GeluTile, len - offset);
                var x   = input.Slice(offset, n);
                var y   = output.Slice(offset, n);
                var inner = innerBuf[..n];
                var tanh  = tanhBuf[..n];

                // inner = sqrt2pi * x * (1 + c * x²)
                TensorPrimitives.Multiply(x, x, inner);                 // inner = x²
                TensorPrimitives.Multiply(inner, GELUCoeff, inner);     // inner = c·x²
                TensorPrimitives.Add(inner, 1f, inner);                 // inner = 1 + c·x²
                TensorPrimitives.Multiply(x, inner, inner);             // inner = x·(1 + c·x²) = x + c·x³
                TensorPrimitives.Multiply(inner, GELUSqrt2OverPi, inner); // inner = sqrt2pi · (x + c·x³)

                // tanh = tanh(inner)
                TensorPrimitives.Tanh(inner, tanh);

                // y = 0.5 · x · (1 + tanh)
                TensorPrimitives.Add(tanh, 1f, tanh);                   // tanh = 1 + tanh
                TensorPrimitives.Multiply(x, tanh, y);                  // y = x · (1 + tanh)
                TensorPrimitives.Multiply(y, 0.5f, y);                  // y = 0.5 · y
            }
        }

        // ── SIMD core: backward ───────────────────────────────────────────────

        /// <summary>
        /// SIMD-batched GELU backward.
        /// <c>dIn[i] += d/dx GELU(x[i]) * dOut[i]</c>, accumulating.
        /// </summary>
        private static void GeluBackwardSimd(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> gradOutput,
            Span<float> gradInput)
        {
            Span<float> innerBuf = stackalloc float[GeluTile];
            Span<float> tanhBuf  = stackalloc float[GeluTile];
            Span<float> factor   = stackalloc float[GeluTile];

            var len = input.Length;
            for (var offset = 0; offset < len; offset += GeluTile)
            {
                var n  = Math.Min(GeluTile, len - offset);
                var x  = input.Slice(offset, n);
                var dO = gradOutput.Slice(offset, n);
                var dI = gradInput.Slice(offset, n);
                var inner = innerBuf[..n];
                var tanh  = tanhBuf[..n];
                var f     = factor[..n];

                // inner = sqrt2pi · (x + c·x³)   AND   f = sqrt2pi · (1 + 3·c·x²)
                TensorPrimitives.Multiply(x, x, inner);                 // inner = x²

                // f = sqrt2pi · (1 + 3·c·x²)   (= d(inner)/dx, reused below)
                TensorPrimitives.Multiply(inner, 3f * GELUCoeff, f);    // f = 3·c·x²
                TensorPrimitives.Add(f, 1f, f);                         // f = 1 + 3·c·x²
                TensorPrimitives.Multiply(f, GELUSqrt2OverPi, f);       // f = sqrt2pi · (1 + 3·c·x²)

                // inner = x · (1 + c·x²) · sqrt2pi
                TensorPrimitives.Multiply(inner, GELUCoeff, inner);     // inner = c·x²
                TensorPrimitives.Add(inner, 1f, inner);                 // inner = 1 + c·x²
                TensorPrimitives.Multiply(x, inner, inner);             // inner = x + c·x³
                TensorPrimitives.Multiply(inner, GELUSqrt2OverPi, inner); // inner = sqrt2pi · (x + c·x³)

                // tanh = tanh(inner)
                TensorPrimitives.Tanh(inner, tanh);

                // factor = 0.5·(1 + tanh) + 0.5·x·(1 - tanh²)·f
                //
                // Compute as: factor = 0.5·(1 + tanh) + 0.5·x·f·(1 - tanh²)
                // Reuse inner as temp:  inner = 1 - tanh²
                TensorPrimitives.Multiply(tanh, tanh, inner);             // inner = tanh²
                TensorPrimitives.Subtract(1f, inner, inner);              // inner = 1 - tanh²
                TensorPrimitives.Multiply(inner, x, inner);               // inner = x · (1 - tanh²)
                TensorPrimitives.Multiply(inner, f, inner);               // inner = x · (1 - tanh²) · f
                TensorPrimitives.Multiply(inner, 0.5f, inner);            // inner = 0.5 · x · (1 - tanh²) · f

                // tanh ← 0.5·(1 + tanh) (term 1)
                TensorPrimitives.Add(tanh, 1f, tanh);
                TensorPrimitives.Multiply(tanh, 0.5f, tanh);

                // factor = term1 + term2
                TensorPrimitives.Add(tanh, inner, f);

                // dIn += f · dOut  (accumulating in-place)
                TensorPrimitives.MultiplyAdd(f, dO, dI, dI);
            }
        }

        // ── Scalar (unused in fast path, kept for AOT/SIMD-disabled fallback
        //    and for reference / numerical-parity testing) ──────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float GeluScalar(float x)
        {
            var inner = GELUSqrt2OverPi * (x + GELUCoeff * x * x * x);
            return 0.5f * x * (1f + MathF.Tanh(inner));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
            var len = chunkEnd - chunkStart;
            GeluForwardSimd(
                new ReadOnlySpan<float>(ctx.Input + chunkStart, len),
                new Span<float>(ctx.Output + chunkStart, len));
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
            var len = chunkEnd - chunkStart;
            GeluBackwardSimd(
                new ReadOnlySpan<float>(ctx.Input + chunkStart, len),
                new ReadOnlySpan<float>(ctx.GradOutput + chunkStart, len),
                new Span<float>(ctx.GradInput + chunkStart, len));
        }
    }
}
