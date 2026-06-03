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
        // Tile size for the SIMD pipeline inside chunk bodies. Two scratch spans
        // of this size live on the worker thread's stack (8 KB for 1024 floats × 2),
        // sized to stay hot in L1 across the multiple TensorPrimitives passes.
        private const int SiLUTile = 1024;

        /// <summary>
        /// SiLU / swish activation: <c>y = x · sigmoid(x)</c>.
        ///
        /// This is the gate non-linearity in SwiGLU (Llama/Qwen FFN):
        /// <c>FFN(x) = down( SiLU(gate(x)) ⊙ up(x) )</c>.
        ///
        /// <para>
        /// <b>Parallelization + SIMD</b> mirror <see cref="Gelu"/>: chunks above
        /// <see cref="ParallelThreshold"/> dispatch via <see cref="OverfitParallelFor"/>,
        /// and within each chunk the sigmoid is computed by
        /// <see cref="TensorPrimitives.Sigmoid(ReadOnlySpan{float}, Span{float})"/>
        /// (SIMD polynomial), not a scalar <c>MathF.Exp</c> loop.
        /// </para>
        /// </summary>
        public static AutogradNode SiLU(ComputationGraph graph, AutogradNode input)
        {
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);
            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            if (inS.Length < ParallelThreshold)
            {
                SiLUForwardSimd(inS, outS);
            }
            else
            {
                unsafe
                {
                    fixed (float* inPtr = inS, outPtr = outS)
                    {
                        var ctx = new SiLUForwardCtx { Input = inPtr, Output = outPtr };
                        OverfitParallelFor.For(0, inS.Length, &SiLUForwardChunk, &ctx);
                    }
                }
            }

            if (input.RequiresGrad)
            {
                graph?.Record(OpCode.SiLU, output, input);
            }

            return output;
        }

        /// <summary>
        /// SiLU backward.
        /// <c>d/dx (x·σ(x)) = σ(x)·(1 + x·(1 − σ(x)))</c>.
        /// Parallelized + SIMD-batched analogous to forward; accumulates into <c>dIn</c>.
        /// </summary>
        public static void SiLUBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var inS = input.DataView.AsReadOnlySpan();
            var dOut = output.GradView.AsReadOnlySpan();
            var dIn = input.GradView.AsSpan();

            if (inS.Length < ParallelThreshold)
            {
                SiLUBackwardSimd(inS, dOut, dIn);
            }
            else
            {
                unsafe
                {
                    fixed (float* inPtr = inS, dOutPtr = dOut, dInPtr = dIn)
                    {
                        var ctx = new SiLUBackwardCtx
                        {
                            Input = inPtr,
                            GradOutput = dOutPtr,
                            GradInput = dInPtr,
                        };
                        OverfitParallelFor.For(0, inS.Length, &SiLUBackwardChunk, &ctx);
                    }
                }
            }
        }

        // ── SIMD core: forward ────────────────────────────────────────────────

        private static void SiLUForwardSimd(ReadOnlySpan<float> input, Span<float> output)
        {
            Span<float> sBuf = stackalloc float[SiLUTile];

            var len = input.Length;
            for (var offset = 0; offset < len; offset += SiLUTile)
            {
                var n = Math.Min(SiLUTile, len - offset);
                var x = input.Slice(offset, n);
                var y = output.Slice(offset, n);
                var s = sBuf[..n];

                TensorPrimitives.Sigmoid(x, s);     // s = σ(x)
                TensorPrimitives.Multiply(x, s, y); // y = x · σ(x)
            }
        }

        // ── SIMD core: backward ───────────────────────────────────────────────

        private static void SiLUBackwardSimd(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> gradOutput,
            Span<float> gradInput)
        {
            Span<float> sBuf = stackalloc float[SiLUTile];
            Span<float> tmpBuf = stackalloc float[SiLUTile];

            var len = input.Length;
            for (var offset = 0; offset < len; offset += SiLUTile)
            {
                var n = Math.Min(SiLUTile, len - offset);
                var x = input.Slice(offset, n);
                var dO = gradOutput.Slice(offset, n);
                var dI = gradInput.Slice(offset, n);
                var s = sBuf[..n];
                var t = tmpBuf[..n];

                TensorPrimitives.Sigmoid(x, s);     // s = σ(x)

                // factor = σ·(1 + x·(1 − σ))
                TensorPrimitives.Subtract(1f, s, t);  // t = 1 − σ
                TensorPrimitives.Multiply(t, x, t);   // t = x·(1 − σ)
                TensorPrimitives.Add(t, 1f, t);       // t = 1 + x·(1 − σ)
                TensorPrimitives.Multiply(t, s, t);   // t = σ·(1 + x·(1 − σ))

                // dIn += factor · dOut  (accumulating in-place)
                TensorPrimitives.MultiplyAdd(t, dO, dI, dI);
            }
        }

        // ── Scalar reference (numerical-parity testing / fallback) ─────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float SiLUScalar(float x)
        {
            var s = 1f / (1f + MathF.Exp(-x));
            return x * s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SiLUGradScalar(float x)
        {
            var s = 1f / (1f + MathF.Exp(-x));
            return s * (1f + x * (1f - s));
        }

        // ── OverfitParallelFor chunk bodies + contexts ────────────────────────

        private unsafe struct SiLUForwardCtx
        {
            public float* Input;
            public float* Output;
        }

        private static unsafe void SiLUForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<SiLUForwardCtx>(contextPtr);
            var len = chunkEnd - chunkStart;
            SiLUForwardSimd(
                new ReadOnlySpan<float>(ctx.Input + chunkStart, len),
                new Span<float>(ctx.Output + chunkStart, len));
        }

        private unsafe struct SiLUBackwardCtx
        {
            public float* Input;
            public float* GradOutput;
            public float* GradInput;
        }

        private static unsafe void SiLUBackwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<SiLUBackwardCtx>(contextPtr);
            var len = chunkEnd - chunkStart;
            SiLUBackwardSimd(
                new ReadOnlySpan<float>(ctx.Input + chunkStart, len),
                new ReadOnlySpan<float>(ctx.GradOutput + chunkStart, len),
                new Span<float>(ctx.GradInput + chunkStart, len));
        }
    }
}
