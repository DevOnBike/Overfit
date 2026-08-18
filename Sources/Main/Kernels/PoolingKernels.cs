// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Kernels
{
    internal static class PoolingKernels
    {
        // ─────────────────────────────────────────────────────────────────────
        // MaxPool2D forward — inference only (no index tracking)
        // ─────────────────────────────────────────────────────────────────────

        public static void MaxPool2DForwardNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW,
            int pool)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pool);

            if (inputH % pool != 0 || inputW % pool != 0)
            {
                throw new ArgumentException("MaxPool2D requires inputH and inputW divisible by pool.");
            }

            var outH = inputH / pool;
            var outW = inputW / pool;
            var inputSize = channels * inputH * inputW;
            var outputSize = channels * outH * outW;

            if (input.Length % inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by MaxPool2D input size.",
                    nameof(input));
            }

            var batchSize = input.Length / inputSize;

            if (output.Length < batchSize * outputSize)
            {
                throw new ArgumentException(
                    "Output span is too small for MaxPool2D.",
                    nameof(output));
            }

            for (var n = 0; n < batchSize; n++)
            {
                MaxPool2DForwardSingleBatchNchw(
                    input.Slice(n * inputSize, inputSize),
                    output.Slice(n * outputSize, outputSize),
                    channels,
                    inputH,
                    inputW,
                    pool,
                    outH,
                    outW);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // MaxPool2D forward — general windowed inference (overlapping stride + padding)
        //
        // The non-overlapping overload above is the fast path (TensorPrimitives pool=2). This one is the
        // ONNX-general case: stride may be < kernel (overlapping, e.g. ResNet's 3x3 stride-2) and the window
        // may extend into zero-padding. ONNX pads MaxPool with -inf, so out-of-bounds positions are simply
        // skipped — they never win the max (every valid output window has at least one in-bounds element).
        //
        //   outH = (inputH + 2*padding - kernelSize) / stride + 1
        //   outW = (inputW + 2*padding - kernelSize) / stride + 1
        // ─────────────────────────────────────────────────────────────────────

        public static void MaxPool2DForwardNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int batchSize,
            int channels,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stride);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kernelSize);

            var outH = (inputH + 2 * padding - kernelSize) / stride + 1;
            var outW = (inputW + 2 * padding - kernelSize) / stride + 1;
            var inputPlane = channels * inputH * inputW;
            var outputPlane = channels * outH * outW;

            for (var n = 0; n < batchSize; n++)
            {
                var inB = input.Slice(n * inputPlane, inputPlane);
                var outB = output.Slice(n * outputPlane, outputPlane);

                for (var c = 0; c < channels; c++)
                {
                    var inChan = c * inputH * inputW;
                    var outChan = c * outH * outW;

                    for (var oy = 0; oy < outH; oy++)
                    {
                        var iyBase = oy * stride - padding;

                        for (var ox = 0; ox < outW; ox++)
                        {
                            var ixBase = ox * stride - padding;
                            var max = float.MinValue;

                            for (var ky = 0; ky < kernelSize; ky++)
                            {
                                var iy = iyBase + ky;
                                if ((uint)iy >= (uint)inputH)
                                {
                                    continue;
                                }

                                var rowBase = inChan + iy * inputW;
                                for (var kx = 0; kx < kernelSize; kx++)
                                {
                                    var ix = ixBase + kx;
                                    if ((uint)ix >= (uint)inputW)
                                    {
                                        continue;
                                    }

                                    var v = inB[rowBase + ix];
                                    if (v > max)
                                    {
                                        max = v;
                                    }
                                }
                            }

                            outB[outChan + oy * outW + ox] = max;
                        }
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // MaxPool2D forward — training path (with index tracking)
        // Fills output values + flat maxIndices for backward scatter.
        // batchOffset is the flat offset of this batch's input start in the
        // full [B, C, H, W] tensor, so indices are globally addressable.
        // ─────────────────────────────────────────────────────────────────────

        public static void MaxPool2DForwardWithIndicesNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            Span<float> maxIndices,
            int channels,
            int inputH,
            int inputW,
            int pool,
            int batchOffset)
        {
            var outH = inputH / pool;
            var outW = inputW / pool;

            if (pool == 2 && inputW % 2 == 0)
            {
                MaxPool2DForwardWithIndicesPool2(
                    input, output, maxIndices,
                    channels, inputH, inputW, outH, outW, batchOffset);

                return;
            }

            MaxPool2DForwardWithIndicesGeneric(
                input, output, maxIndices,
                channels, inputH, inputW, pool, outH, outW, batchOffset);
        }

        // ─────────────────────────────────────────────────────────────────────
        // GlobalAveragePool2D forward
        // ─────────────────────────────────────────────────────────────────────

        public static void GlobalAveragePool2DForwardNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);

            var spatialSize = inputH * inputW;
            var inputSize = channels * spatialSize;
            var outputSize = channels;

            if (input.Length % inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by GlobalAveragePool2D input size.",
                    nameof(input));
            }

            var batchSize = input.Length / inputSize;

            if (output.Length < batchSize * outputSize)
            {
                throw new ArgumentException(
                    "Output span is too small for GlobalAveragePool2D.",
                    nameof(output));
            }

            var scale = 1f / spatialSize;

            for (var n = 0; n < batchSize; n++)
            {
                GlobalAveragePool2DForwardSingleBatchNchw(
                    input.Slice(n * inputSize, inputSize),
                    output.Slice(n * outputSize, outputSize),
                    channels,
                    spatialSize,
                    scale);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Private: inference single batch
        // ─────────────────────────────────────────────────────────────────────

        private static void MaxPool2DForwardSingleBatchNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW,
            int pool,
            int outH,
            int outW)
        {
            if (pool == 2 && inputW % 2 == 0)
            {
                MaxPool2DForwardSingleBatchPool2NoIndex(
                    input, output, channels, inputH, inputW, outH, outW);
                return;
            }

            for (var c = 0; c < channels; c++)
            {
                var inputChannelBase = c * inputH * inputW;
                var outputChannelBase = c * outH * outW;

                for (var oh = 0; oh < outH; oh++)
                {
                    for (var ow = 0; ow < outW; ow++)
                    {
                        var max = float.MinValue;

                        for (var ph = 0; ph < pool; ph++)
                        {
                            var rowBase = inputChannelBase + (oh * pool + ph) * inputW + ow * pool;
                            for (var pw = 0; pw < pool; pw++)
                            {
                                var value = input[rowBase + pw];
                                if (value > max)
                                {
                                    max = value;
                                }
                            }
                        }

                        output[outputChannelBase + oh * outW + ow] = max;
                    }
                }
            }
        }

        /// <summary>
        /// Pool=2, stride=2 fast path without index recording (inference).
        ///
        /// Two-step approach per output row:
        ///   1) TensorPrimitives.Max(row0, row1, pairMax) — SIMD element-wise
        ///      vertical max across the two input rows (hardware-vectorised by the
        ///      JIT via AVX2/AVX-512 on supported CPUs).
        ///   2) Scalar loop over outW — collapses adjacent horizontal pairs.
        ///      Step 2 is only outW iterations (13 for MNIST), negligible cost.
        ///
        /// Benchmark context: for [64, 8, 26, 26] → [64, 8, 13, 13]:
        ///   - 64 batches × 8 channels × 13 rows = 6656 TensorPrimitives.Max calls
        ///     each on inputW=26 floats → fully vectorised
        ///   - eliminates all branching in the hot path
        ///   - pairMax is stackalloc'd (104 bytes for inputW=26) — zero heap alloc
        /// </summary>
        /// <summary>
        /// Elements in the input tensor below which pooling stays on one thread.
        ///
        /// <para>Fixed dispatch cost for <see cref="OverfitParallel"/> is ~0.22 ms measured, and pooling is
        /// pure streaming, so a small tensor pays the dispatch and gains nothing. 262,144 floats is 1 MB, at
        /// which the serial pass costs roughly a millisecond here — several times the dispatch. The MNIST
        /// CNN sits far below this and keeps the old path exactly; VGG-16's pooling sits far above it.</para>
        /// </summary>
        private static readonly int ParallelPoolElements =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ParallelPool) == "0"
                ? int.MaxValue
                : 262_144;

        /// <summary>
        /// Pool=2 stride=2 inference, split across workers by channel.
        ///
        /// <para><b>Why this is parallel now (`XC-78`, 2026-08-18).</b> On VGG-16 the five pooling nodes cost
        /// <b>5.40 ms of a 79.48 ms inference — 6.8%</b> — and the first of them moved 16.06 MB in 2.86 ms,
        /// which is <b>5.6 GB/s where the ReLU node beside it on the same tensor reached 62.7 GB/s</b>.
        /// Pooling ran on one core while the rest of the model used the machine. Channels are independent,
        /// so this was a decomposition that was missing, not a kernel that was slow.</para>
        ///
        /// <para>Each worker owns its row scratch. One shared buffer across workers is a data race that
        /// produces plausible wrong pixels rather than a crash, which is the worse of the two failures.</para>
        /// </summary>
        private static unsafe void MaxPool2DForwardSingleBatchPool2NoIndex(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW,
            int outH,
            int outW)
        {
            var elements = (long)channels * inputH * inputW;

            fixed (float* pin = input, pout = output)
            {
                var ctx = new Pool2Ctx(pin, pout, channels, inputH, inputW, outH, outW);

                if (elements < ParallelPoolElements)
                {
                    Pool2ChannelRange(0, channels, &ctx);
                    return;
                }

                OverfitParallel.For(0, channels, 1, &Pool2ChannelWorker, &ctx);
            }
        }

        private static unsafe void Pool2ChannelWorker(int channelStart, int channelEnd, void* ctxPtr)
        {
            Pool2ChannelRange(channelStart, channelEnd, (Pool2Ctx*)ctxPtr);
        }

        private static unsafe void Pool2ChannelRange(int channelStart, int channelEnd, Pool2Ctx* ctxPtr)
        {
            ref readonly var ctx = ref *ctxPtr;

            var inputW = ctx.InputW;
            var outH = ctx.OutH;
            var outW = ctx.OutW;
            var inputPlane = ctx.InputH * inputW;
            var outputPlane = outH * outW;

            using var pooledPairMax = inputW <= 128 ? default : new PooledBuffer<float>(inputW, clearMemory: false);
#pragma warning disable OVERFIT026 // BOUND: guarded at inputW <= 128 floats = 512 B, exactly the OVERFIT025 budget; wider inputs take the pooled branch on the line above.
            var pairMax = inputW <= 128
                ? stackalloc float[inputW]
                : pooledPairMax.Span;
#pragma warning restore OVERFIT026

            for (var c = channelStart; c < channelEnd; c++)
            {
                var inputChannelBase = c * inputPlane;
                var outputChannelBase = c * outputPlane;

                for (var oh = 0; oh < outH; oh++)
                {
                    var row0 = new ReadOnlySpan<float>(ctx.Input + inputChannelBase + (oh * 2 * inputW), inputW);
                    var row1 = new ReadOnlySpan<float>(ctx.Input + inputChannelBase + ((oh * 2 + 1) * inputW), inputW);

                    // Vertical max: for each column, keep the larger of the two rows.
                    TensorPrimitives.Max(row0, row1, pairMax);

                    // Horizontal max: collapse adjacent pairs → output pixels.
                    var outRow = ctx.Output + outputChannelBase + (oh * outW);

                    for (var ow = 0; ow < outW; ow++)
                    {
                        var a = pairMax[ow * 2];
                        var b = pairMax[(ow * 2) + 1];

                        outRow[ow] = a > b ? a : b;
                    }
                }
            }
        }

        /// <summary>Pointers and shape for one pool=2 tensor, passed to workers by address.</summary>
        private readonly unsafe struct Pool2Ctx
        {
            public readonly float* Input;
            public readonly float* Output;
            public readonly int Channels;
            public readonly int InputH;
            public readonly int InputW;
            public readonly int OutH;
            public readonly int OutW;

            public Pool2Ctx(
                float* input,
                float* output,
                int channels,
                int inputH,
                int inputW,
                int outH,
                int outW)
            {
                Input = input;
                Output = output;
                Channels = channels;
                InputH = inputH;
                InputW = inputW;
                OutH = outH;
                OutW = outW;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Private: training paths (with index tracking)
        // ─────────────────────────────────────────────────────────────────────

        private static void MaxPool2DForwardWithIndicesPool2(
            ReadOnlySpan<float> input,
            Span<float> output,
            Span<float> maxIndices,
            int channels,
            int inputH,
            int inputW,
            int outH,
            int outW,
            int batchOffset)
        {
            if (CpuFeatures.HasAvx2 && outW >= 8)
            {
                MaxPool2DForwardWithIndicesPool2Avx2(
                    input, output, maxIndices, channels, inputH, inputW, outH, outW, batchOffset);
                return;
            }

            MaxPool2DForwardWithIndicesPool2Scalar(
                input, output, maxIndices, channels, inputH, inputW, outH, outW, batchOffset);
        }

        /// <summary>Scalar pool=2 reference (the pre-AVX2 production path) — kept callable as the
        /// bit-identity oracle for <see cref="MaxPool2DForwardWithIndicesPool2Avx2"/>.</summary>
        internal static void MaxPool2DForwardWithIndicesPool2Scalar(
            ReadOnlySpan<float> input,
            Span<float> output,
            Span<float> maxIndices,
            int channels,
            int inputH,
            int inputW,
            int outH,
            int outW,
            int batchOffset)
        {
            using var pooledPairMax = inputW <= 128 ? default : new PooledBuffer<float>(inputW, clearMemory: false);
#pragma warning disable OVERFIT026 // BOUND: guarded at inputW <= 128 floats = 512 B, exactly the OVERFIT025 budget; wider inputs take the pooled branch on the line above.
            var pairMax = inputW <= 128
                ? stackalloc float[inputW]
                : pooledPairMax.Span;
#pragma warning restore OVERFIT026

            for (var c = 0; c < channels; c++)
            {
                var inputChannelBase = c * inputH * inputW;
                var outputChannelBase = c * outH * outW;

                for (var oh = 0; oh < outH; oh++)
                {
                    var row0Start = inputChannelBase + oh * 2 * inputW;
                    var row1Start = inputChannelBase + (oh * 2 + 1) * inputW;

                    var row0 = input.Slice(row0Start, inputW);
                    var row1 = input.Slice(row1Start, inputW);

                    TensorPrimitives.Max(row0, row1, pairMax);

                    var outRowBase = outputChannelBase + oh * outW;

                    for (var ow = 0; ow < outW; ow++)
                    {
                        var a = pairMax[ow * 2];
                        var b = pairMax[ow * 2 + 1];

                        // Horizontal winner picks the column; vertical winner is whichever row held the
                        // larger value. Ternaries, not two ifs: both outputs are assigned on every path, and
                        // split ifs would not prove definite assignment to the compiler.
                        var takeLeft = a >= b;
                        var maxVal = takeLeft ? a : b;
                        var col = takeLeft ? ow * 2 : (ow * 2) + 1;
                        var idxInRow0 = row0Start + col;
                        var idxInRow1 = row1Start + col;
                        var maxIdx = input[idxInRow0] >= input[idxInRow1] ? idxInRow0 : idxInRow1;

                        output[outRowBase + ow] = maxVal;
                        maxIndices[outRowBase + ow] = batchOffset + maxIdx;
                    }
                }
            }
        }

        /// <summary>
        /// AVX2 pool=2 path — 8 outputs per iteration. Bit-identical to the scalar Pool2 path,
        /// including the tie rules the backward scatter depends on: horizontal tie (a == b) picks the
        /// LEFT column; vertical tie picks ROW 0. Values: vertical <c>Max(row0,row1)</c> then horizontal
        /// <c>Max(even,odd)</c> — same numbers as the scalar compare chain (no NaNs in this path).
        /// Index math is exact INT32, converted to float at the end (indices &lt; 2^24 by construction:
        /// batch-tensor-relative). Trailing outputs (outW % 8) fall through to the scalar inner loop.
        /// </summary>
        internal static void MaxPool2DForwardWithIndicesPool2Avx2(
            ReadOnlySpan<float> input,
            Span<float> output,
            Span<float> maxIndices,
            int channels,
            int inputH,
            int inputW,
            int outH,
            int outW,
            int batchOffset)
        {
            if (outW < 8)
            {
                MaxPool2DForwardWithIndicesPool2Scalar(
                    input, output, maxIndices, channels, inputH, inputW, outH, outW, batchOffset);
                return;
            }

            // Deinterleave reorder: per-128-lane Shuffle yields [a0,a2,b0,b2 | a4,a6,b4,b6];
            // this permute restores ascending order [a0,a2,a4,a6,b0,b2,b4,b6].
            var fix = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
            var iota2 = Vector256.Create(0, 2, 4, 6, 8, 10, 12, 14);
            var onesI = Vector256.Create(1);

            for (var c = 0; c < channels; c++)
            {
                var inputChannelBase = c * inputH * inputW;
                var outputChannelBase = c * outH * outW;

                for (var oh = 0; oh < outH; oh++)
                {
                    var row0Start = inputChannelBase + oh * 2 * inputW;
                    var row1Start = inputChannelBase + (oh * 2 + 1) * inputW;
                    var row0 = input.Slice(row0Start, inputW);
                    var row1 = input.Slice(row1Start, inputW);
                    var outRowBase = outputChannelBase + oh * outW;

                    var row0StartV = Vector256.Create(row0Start + batchOffset);
                    var row1StartV = Vector256.Create(row1Start + batchOffset);

                    // Overlapping-last-window: when outW % 8 != 0 the final iteration re-runs at
                    // ow = outW-8, overwriting up to 7 already-computed lanes with identical values
                    // (pure function of the input) — no scalar tail, still bit-identical.
                    var ow = 0;
                    for (; ow + 8 <= outW; ow += 8)
                    {
                        var col = ow * 2;
                        var r0Lo = Vector256.Create(row0.Slice(col, 8));
                        var r0Hi = Vector256.Create(row0.Slice(col + 8, 8));
                        var r1Lo = Vector256.Create(row1.Slice(col, 8));
                        var r1Hi = Vector256.Create(row1.Slice(col + 8, 8));

                        // Deinterleave both rows into even/odd column lanes.
                        var r0E = Avx2.PermuteVar8x32(Avx.Shuffle(r0Lo, r0Hi, 0b10_00_10_00), fix);
                        var r0O = Avx2.PermuteVar8x32(Avx.Shuffle(r0Lo, r0Hi, 0b11_01_11_01), fix);
                        var r1E = Avx2.PermuteVar8x32(Avx.Shuffle(r1Lo, r1Hi, 0b10_00_10_00), fix);
                        var r1O = Avx2.PermuteVar8x32(Avx.Shuffle(r1Lo, r1Hi, 0b11_01_11_01), fix);

                        // Vertical max per column, then horizontal winner (a = even col, b = odd col).
                        var a = Avx.Max(r0E, r1E);
                        var b = Avx.Max(r0O, r1O);
                        var hMask = Avx.CompareGreaterThanOrEqual(a, b);   // true → LEFT column wins
                        var maxVal = Avx.Max(a, b);

                        // Vertical winner at the winning column: row0 wins on >= (tie → row0).
                        var r0Win = Avx.BlendVariable(r0O, r0E, hMask);
                        var r1Win = Avx.BlendVariable(r1O, r1E, hMask);
                        var vMask = Avx.CompareGreaterThanOrEqual(r0Win, r1Win);

                        // index = (winning row start + batchOffset) + 2·ow + (left ? 0 : 1) — exact int32.
                        var colV = Avx2.Add(Vector256.Create(col), iota2);
                        colV = Avx2.Add(colV, Avx2.AndNot(hMask.AsInt32(), onesI));
                        var baseV = Avx2.BlendVariable(row1StartV, row0StartV, vMask.AsInt32());
                        var idxF = Avx.ConvertToVector256Single(Avx2.Add(baseV, colV));

                        maxVal.CopyTo(output.Slice(outRowBase + ow, 8));
                        idxF.CopyTo(maxIndices.Slice(outRowBase + ow, 8));
                    }

                    if (ow < outW)
                    {
                        // Overlapping last window: redo one 8-wide pass at outW-8 — overwrites up to 7
                        // already-written lanes with identical values (pure function) → bit-identical,
                        // no scalar tail.
                        ow = outW - 8;
                        var col = ow * 2;
                        var r0Lo = Vector256.Create(row0.Slice(col, 8));
                        var r0Hi = Vector256.Create(row0.Slice(col + 8, 8));
                        var r1Lo = Vector256.Create(row1.Slice(col, 8));
                        var r1Hi = Vector256.Create(row1.Slice(col + 8, 8));
                        var r0E = Avx2.PermuteVar8x32(Avx.Shuffle(r0Lo, r0Hi, 0b10_00_10_00), fix);
                        var r0O = Avx2.PermuteVar8x32(Avx.Shuffle(r0Lo, r0Hi, 0b11_01_11_01), fix);
                        var r1E = Avx2.PermuteVar8x32(Avx.Shuffle(r1Lo, r1Hi, 0b10_00_10_00), fix);
                        var r1O = Avx2.PermuteVar8x32(Avx.Shuffle(r1Lo, r1Hi, 0b11_01_11_01), fix);
                        var a = Avx.Max(r0E, r1E);
                        var b = Avx.Max(r0O, r1O);
                        var hMask = Avx.CompareGreaterThanOrEqual(a, b);
                        var maxVal = Avx.Max(a, b);
                        var r0Win = Avx.BlendVariable(r0O, r0E, hMask);
                        var r1Win = Avx.BlendVariable(r1O, r1E, hMask);
                        var vMask = Avx.CompareGreaterThanOrEqual(r0Win, r1Win);
                        var colV = Avx2.Add(Vector256.Create(col), iota2);
                        colV = Avx2.Add(colV, Avx2.AndNot(hMask.AsInt32(), onesI));
                        var baseV = Avx2.BlendVariable(row1StartV, row0StartV, vMask.AsInt32());
                        var idxF = Avx.ConvertToVector256Single(Avx2.Add(baseV, colV));
                        maxVal.CopyTo(output.Slice(outRowBase + ow, 8));
                        idxF.CopyTo(maxIndices.Slice(outRowBase + ow, 8));
                    }

                }
            }
        }

        internal static void MaxPool2DForwardWithIndicesGeneric(
            ReadOnlySpan<float> input,
            Span<float> output,
            Span<float> maxIndices,
            int channels,
            int inputH,
            int inputW,
            int pool,
            int outH,
            int outW,
            int batchOffset)
        {
            for (var c = 0; c < channels; c++)
            {
                var inputChannelBase = c * inputH * inputW;
                var outputChannelBase = c * outH * outW;

                for (var oh = 0; oh < outH; oh++)
                {
                    for (var ow = 0; ow < outW; ow++)
                    {
                        var maxVal = float.MinValue;
                        var maxIdx = 0;

                        for (var ph = 0; ph < pool; ph++)
                        {
                            var rowBase = inputChannelBase + (oh * pool + ph) * inputW + ow * pool;
                            for (var pw = 0; pw < pool; pw++)
                            {
                                var idx = rowBase + pw;
                                var val = input[idx];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdx = idx;
                                }
                            }
                        }

                        var outIdx = outputChannelBase + oh * outW + ow;
                        output[outIdx] = maxVal;
                        maxIndices[outIdx] = batchOffset + maxIdx;
                    }
                }
            }
        }

        private static void GlobalAveragePool2DForwardSingleBatchNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int spatialSize,
            float scale)
        {
            for (var c = 0; c < channels; c++)
            {
                output[c] = TensorPrimitives.Sum(input.Slice(c * spatialSize, spatialSize)) * scale;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // AveragePool2D forward (ONNX-3b)
        //
        // Standard windowed average pooling with:
        //   - Square kernel (kernelSize × kernelSize)
        //   - Symmetric zero-padding
        //   - Configurable stride
        //   - count_include_pad support (ONNX default = 0)
        //
        // Output dims:
        //   outH = (inputH + 2*padding - kernelSize) / stride + 1
        //   outW = (inputW + 2*padding - kernelSize) / stride + 1
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Windowed average pooling (NCHW layout).
        /// </summary>
        public static void AveragePool2DForwardNchw(
            ReadOnlySpan<float> input,    // [batch, C, inputH, inputW]
            Span<float> output,           // [batch, C, outH, outW]
            int batchSize,
            int channels,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride,
            bool countIncludePad = false)
        {
            var outH = (inputH + 2 * padding - kernelSize) / stride + 1;
            var outW = (inputW + 2 * padding - kernelSize) / stride + 1;
            var inputPlane = channels * inputH * inputW;
            var outputPlane = channels * outH * outW;

            output.Clear();

            for (var n = 0; n < batchSize; n++)
            {
                AveragePool2DForwardSingleBatchNchw(
                    input.Slice(n * inputPlane, inputPlane),
                    output.Slice(n * outputPlane, outputPlane),
                    channels,
                    inputH, inputW,
                    kernelSize,
                    outH, outW,
                    padding, stride,
                    countIncludePad);
            }
        }

        private static void AveragePool2DForwardSingleBatchNchw(
            ReadOnlySpan<float> input,
            Span<float> output,
            int channels,
            int inputH,
            int inputW,
            int kernelSize,
            int outH,
            int outW,
            int padding,
            int stride,
            bool countIncludePad)
        {
            for (var c = 0; c < channels; c++)
            {
                var inputChanBase = c * inputH * inputW;
                var outputChanBase = c * outH * outW;

                for (var oy = 0; oy < outH; oy++)
                {
                    var inputYBase = oy * stride - padding;

                    for (var ox = 0; ox < outW; ox++)
                    {
                        var inputXBase = ox * stride - padding;
                        var sum = 0f;
                        var count = 0;

                        for (var ky = 0; ky < kernelSize; ky++)
                        {
                            var iy = inputYBase + ky;
                            var inBoundsY = (uint)iy < (uint)inputH;

                            for (var kx = 0; kx < kernelSize; kx++)
                            {
                                var ix = inputXBase + kx;
                                var inBoundsX = (uint)ix < (uint)inputW;

                                var inBounds = inBoundsY && inBoundsX;

                                if (inBounds)
                                {
                                    sum += input[inputChanBase + iy * inputW + ix];
                                    count++;
                                }

                                if (!inBounds && countIncludePad)
                                {
                                    // Zero-pad contributes 0 to sum but 1 to count.
                                    count++;
                                }
                            }
                        }

                        var divisor = countIncludePad ? kernelSize * kernelSize : count;
                        output[outputChanBase + oy * outW + ox] =
                            divisor > 0 ? sum / divisor : 0f;
                    }
                }
            }
        }

    }
}