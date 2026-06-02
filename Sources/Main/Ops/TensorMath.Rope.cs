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
        /// <summary>
        /// Rotary Position Embedding (RoPE), autograd / training version.
        ///
        /// Two pairings, matching the inference <see cref="LanguageModels.Rope.RopeKernel"/>:
        /// <list type="bullet">
        ///   <item><b>adjacent-pair</b> (default, <paramref name="splitHalf"/>=false): pairs
        ///   <c>(x[2p], x[2p+1])</c> share frequency p. Original RoPE / llama.cpp NEOX-as-wired.</item>
        ///   <item><b>split-half</b> (<paramref name="splitHalf"/>=true): pairs
        ///   <c>(x[p], x[p+halfDim])</c> share frequency p. HF rotate_half — the convention
        ///   <b>Qwen2/2.5/3 GGUFs use</b> (<c>RopeSplitHalf=true</c>).</item>
        /// </list>
        /// Each pair is a 2-D rotation by the precomputed angle:
        ///   <c>y0 = x0·cos − x1·sin,  y1 = x0·sin + x1·cos</c>.
        ///
        /// Shapes:
        ///   input    [rows, D]          flattened; D = headsPerRow · headDim
        ///   cos, sin [rows, halfDim]    per-row angles (halfDim = headDim/2), shared across the
        ///                               heads packed into a row. NOT trainable (no gradient).
        ///
        /// The rotation matrix is orthogonal, so backward is the transpose = the inverse rotation
        /// (sin negated): <c>dx0 = cos·dy0 + sin·dy1,  dx1 = −sin·dy0 + cos·dy1</c>.
        ///
        /// Parallelized over rows via <see cref="OverfitParallelFor"/> above
        /// <see cref="ParallelThreshold"/>; each row is independent.
        /// </summary>
        public static AutogradNode Rope(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode cos,
            AutogradNode sin,
            bool splitHalf = false)
        {
            var D = input.Shape[input.Shape.Rank - 1];
            var rows = input.Shape.Size / D;
            var halfDim = cos.Shape[cos.Shape.Rank - 1];
            var headDim = 2 * halfDim;

            if (D % headDim != 0)
            {
                throw new ArgumentException(
                    $"RoPE: input last dim {D} is not a multiple of headDim {headDim} (= 2·cos.lastDim).");
            }
            if (cos.Shape.Size / halfDim != rows || sin.Shape.Size / halfDim != rows)
            {
                throw new ArgumentException("RoPE: cos/sin must have one [halfDim] row per input row.");
            }

            var headsPerRow = D / headDim;
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();
            var cosS = cos.DataView.AsReadOnlySpan();
            var sinS = sin.DataView.AsReadOnlySpan();

            if ((long)rows * D < ParallelThreshold)
            {
                for (var r = 0; r < rows; r++)
                {
                    RopeForwardRow(inS, outS, cosS, sinS, r, D, headDim, halfDim, headsPerRow, splitHalf);
                }
            }
            else
            {
                unsafe
                {
                    fixed (float* inPtr = inS, outPtr = outS, cosPtr = cosS, sinPtr = sinS)
                    {
                        var ctx = new RopeCtx
                        {
                            Input = inPtr,
                            Output = outPtr,
                            Cos = cosPtr,
                            Sin = sinPtr,
                            D = D,
                            HeadDim = headDim,
                            HalfDim = halfDim,
                            HeadsPerRow = headsPerRow,
                            SplitHalf = splitHalf,
                        };
                        OverfitParallelFor.For(0, rows, &RopeForwardChunk, &ctx);
                    }
                }
            }

            if (input.RequiresGrad)
            {
                graph?.Record(OpCode.Rope, output, input, c0: cos, c1: sin, i0: splitHalf ? 1 : 0, contextCount: 2);
            }

            return output;
        }

        /// <summary>RoPE backward — inverse rotation of the upstream gradient (sin negated),
        /// accumulated into <c>input.Grad</c>. cos/sin are constants (no gradient).</summary>
        public static void RopeBackward(
            AutogradNode input, AutogradNode output, AutogradNode cos, AutogradNode sin, bool splitHalf)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var D = input.Shape[input.Shape.Rank - 1];
            var rows = input.Shape.Size / D;
            var halfDim = cos.Shape[cos.Shape.Rank - 1];
            var headDim = 2 * halfDim;
            var headsPerRow = D / headDim;

            var dOutS = output.GradView.AsReadOnlySpan();
            var dInS = input.GradView.AsSpan();
            var cosS = cos.DataView.AsReadOnlySpan();
            var sinS = sin.DataView.AsReadOnlySpan();

            if ((long)rows * D < ParallelThreshold)
            {
                for (var r = 0; r < rows; r++)
                {
                    RopeBackwardRow(dOutS, dInS, cosS, sinS, r, D, headDim, halfDim, headsPerRow, splitHalf);
                }
            }
            else
            {
                unsafe
                {
                    fixed (float* dOutPtr = dOutS, dInPtr = dInS, cosPtr = cosS, sinPtr = sinS)
                    {
                        var ctx = new RopeCtx
                        {
                            Input = dInPtr,    // reuse: Input slot carries dInput here
                            Output = dOutPtr,  // Output slot carries dOutput
                            Cos = cosPtr,
                            Sin = sinPtr,
                            D = D,
                            HeadDim = headDim,
                            HalfDim = halfDim,
                            HeadsPerRow = headsPerRow,
                            SplitHalf = splitHalf,
                        };
                        OverfitParallelFor.For(0, rows, &RopeBackwardChunk, &ctx);
                    }
                }
            }
        }

        // ── per-row cores (shared by sequential + parallel paths) ─────────────

        private static void RopeForwardRow(
            ReadOnlySpan<float> inS, Span<float> outS,
            ReadOnlySpan<float> cosS, ReadOnlySpan<float> sinS,
            int r, int D, int headDim, int halfDim, int headsPerRow, bool splitHalf)
        {
            var cosRow = cosS.Slice(r * halfDim, halfDim);
            var sinRow = sinS.Slice(r * halfDim, halfDim);
            var rowBase = r * D;

            for (var h = 0; h < headsPerRow; h++)
            {
                var b = rowBase + h * headDim;
                for (var p = 0; p < halfDim; p++)
                {
                    var (i0, i1) = splitHalf ? (p, p + halfDim) : (2 * p, 2 * p + 1);
                    var x0 = inS[b + i0];
                    var x1 = inS[b + i1];
                    var c = cosRow[p];
                    var s = sinRow[p];
                    outS[b + i0] = x0 * c - x1 * s;
                    outS[b + i1] = x0 * s + x1 * c;
                }
            }
        }

        private static void RopeBackwardRow(
            ReadOnlySpan<float> dOutS, Span<float> dInS,
            ReadOnlySpan<float> cosS, ReadOnlySpan<float> sinS,
            int r, int D, int headDim, int halfDim, int headsPerRow, bool splitHalf)
        {
            var cosRow = cosS.Slice(r * halfDim, halfDim);
            var sinRow = sinS.Slice(r * halfDim, halfDim);
            var rowBase = r * D;

            for (var h = 0; h < headsPerRow; h++)
            {
                var b = rowBase + h * headDim;
                for (var p = 0; p < halfDim; p++)
                {
                    var (i0, i1) = splitHalf ? (p, p + halfDim) : (2 * p, 2 * p + 1);
                    var dy0 = dOutS[b + i0];
                    var dy1 = dOutS[b + i1];
                    var c = cosRow[p];
                    var s = sinRow[p];
                    dInS[b + i0] += c * dy0 + s * dy1;
                    dInS[b + i1] += -s * dy0 + c * dy1;
                }
            }
        }

        // ── OverfitParallelFor chunk bodies + context ─────────────────────────

        private unsafe struct RopeCtx
        {
            public float* Input;
            public float* Output;
            public float* Cos;
            public float* Sin;
            public int D;
            public int HeadDim;
            public int HalfDim;
            public int HeadsPerRow;
            public bool SplitHalf;
        }

        private static unsafe void RopeForwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<RopeCtx>(contextPtr);
            var inAll = new ReadOnlySpan<float>(ctx.Input, chunkEnd * ctx.D);
            var outAll = new Span<float>(ctx.Output, chunkEnd * ctx.D);
            var cosAll = new ReadOnlySpan<float>(ctx.Cos, chunkEnd * ctx.HalfDim);
            var sinAll = new ReadOnlySpan<float>(ctx.Sin, chunkEnd * ctx.HalfDim);
            for (var r = chunkStart; r < chunkEnd; r++)
            {
                RopeForwardRow(inAll, outAll, cosAll, sinAll, r, ctx.D, ctx.HeadDim, ctx.HalfDim, ctx.HeadsPerRow, ctx.SplitHalf);
            }
        }

        private static unsafe void RopeBackwardChunk(int chunkStart, int chunkEnd, void* contextPtr)
        {
            ref var ctx = ref Unsafe.AsRef<RopeCtx>(contextPtr);
            // ctx.Input carries dInput, ctx.Output carries dOutput (see RopeBackward).
            var dInAll = new Span<float>(ctx.Input, chunkEnd * ctx.D);
            var dOutAll = new ReadOnlySpan<float>(ctx.Output, chunkEnd * ctx.D);
            var cosAll = new ReadOnlySpan<float>(ctx.Cos, chunkEnd * ctx.HalfDim);
            var sinAll = new ReadOnlySpan<float>(ctx.Sin, chunkEnd * ctx.HalfDim);
            for (var r = chunkStart; r < chunkEnd; r++)
            {
                RopeBackwardRow(dOutAll, dInAll, cosAll, sinAll, r, ctx.D, ctx.HeadDim, ctx.HalfDim, ctx.HeadsPerRow, ctx.SplitHalf);
            }
        }
    }
}
