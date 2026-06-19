// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Kernels
{
    internal static class Conv2DKernels
    {
        public static void ForwardValidNchw(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputH);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputW);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kernelSize);

            if (kernelSize > inputH || kernelSize > inputW)
            {
                throw new ArgumentException("Kernel size cannot be larger than input spatial dimensions.");
            }

            var outH = inputH - kernelSize + 1;
            var outW = inputW - kernelSize + 1;

            var inputSize = inChannels * inputH * inputW;
            var outputSize = outChannels * outH * outW;
            var kernelSizePerOutput = inChannels * kernelSize * kernelSize;

            if (input.Length % inputSize != 0)
            {
                throw new ArgumentException(
                    "Input length is not divisible by Conv2D input size.",
                    nameof(input));
            }

            if (kernels.Length < outChannels * kernelSizePerOutput)
            {
                throw new ArgumentException(
                    "Kernel span is too small.",
                    nameof(kernels));
            }

            var batchSize = input.Length / inputSize;
            var expectedOutputLength = batchSize * outputSize;

            if (output.Length < expectedOutputLength)
            {
                throw new ArgumentException(
                    "Output span is too small for Conv2D inference.",
                    nameof(output));
            }

            // Generic valid conv (incl. ResNet's 1x1 bottlenecks) → im2col + GEMM. The single-channel 3x3 case
            // keeps its dedicated vectorized fast path below.
            if (Conv2DGemmKernels.IsSupported && !(inChannels == 1 && kernelSize == 3))
            {
                Conv2DGemmKernels.Forward(
                    input, kernels, output, batchSize, inChannels, outChannels, inputH, inputW, kernelSize, 0, 1);
                return;
            }

            for (var n = 0; n < batchSize; n++)
            {
                var inputBatch = input.Slice(n * inputSize, inputSize);
                var outputBatch = output.Slice(n * outputSize, outputSize);

                if (inChannels == 1 && kernelSize == 3)
                {
                    ForwardValidSingleChannel3x3(
                        inputBatch,
                        kernels,
                        outputBatch,
                        outChannels,
                        inputH,
                        inputW,
                        outH,
                        outW);
                }
                else
                {
                    // Valid conv (no padding, unit stride) is the padded path with padding=0, stride=1 — reuse the
                    // parallel ForwardNchwSingleBatch so 1x1 / generic convs (e.g. ResNet's bottleneck 1x1 layers,
                    // which take this branch) also fan out over output channels instead of running single-threaded.
                    ForwardNchwSingleBatch(
                        inputBatch,
                        kernels,
                        outputBatch,
                        inChannels,
                        outChannels,
                        inputH,
                        inputW,
                        kernelSize,
                        outH,
                        outW,
                        kernelSizePerOutput,
                        padding: 0,
                        stride: 1);
                }
            }
        }

        private static void ForwardValidSingleChannel3x3(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int outChannels,
            int inputH,
            int inputW,
            int outH,
            int outW)
        {
            if (Vector.IsHardwareAccelerated && outW >= Vector<float>.Count)
            {
                ForwardValidSingleChannel3x3Vectorized(
                    input,
                    kernels,
                    output,
                    outChannels,
                    inputW,
                    outH,
                    outW);

                return;
            }

            ForwardValidSingleChannel3x3Scalar(
                input,
                kernels,
                output,
                outChannels,
                inputW,
                outH,
                outW);
        }

        private static void ForwardValidSingleChannel3x3Vectorized(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int outChannels,
            int inputW,
            int outH,
            int outW)
        {
            var vectorWidth = Vector<float>.Count;

            for (var oc = 0; oc < outChannels; oc++)
            {
                var kernelBase = oc * 9;
                var outputChannelBase = oc * outH * outW;

                var k00 = new Vector<float>(kernels[kernelBase + 0]);
                var k01 = new Vector<float>(kernels[kernelBase + 1]);
                var k02 = new Vector<float>(kernels[kernelBase + 2]);
                var k10 = new Vector<float>(kernels[kernelBase + 3]);
                var k11 = new Vector<float>(kernels[kernelBase + 4]);
                var k12 = new Vector<float>(kernels[kernelBase + 5]);
                var k20 = new Vector<float>(kernels[kernelBase + 6]);
                var k21 = new Vector<float>(kernels[kernelBase + 7]);
                var k22 = new Vector<float>(kernels[kernelBase + 8]);

                for (var oy = 0; oy < outH; oy++)
                {
                    var inputRow0 = oy * inputW;
                    var inputRow1 = (oy + 1) * inputW;
                    var inputRow2 = (oy + 2) * inputW;
                    var outputRow = outputChannelBase + oy * outW;

                    var ox = 0;

                    for (; ox <= outW - vectorWidth; ox += vectorWidth)
                    {
                        var acc =
                            new Vector<float>(input.Slice(inputRow0 + ox, vectorWidth)) * k00 +
                            new Vector<float>(input.Slice(inputRow0 + ox + 1, vectorWidth)) * k01 +
                            new Vector<float>(input.Slice(inputRow0 + ox + 2, vectorWidth)) * k02 +
                            new Vector<float>(input.Slice(inputRow1 + ox, vectorWidth)) * k10 +
                            new Vector<float>(input.Slice(inputRow1 + ox + 1, vectorWidth)) * k11 +
                            new Vector<float>(input.Slice(inputRow1 + ox + 2, vectorWidth)) * k12 +
                            new Vector<float>(input.Slice(inputRow2 + ox, vectorWidth)) * k20 +
                            new Vector<float>(input.Slice(inputRow2 + ox + 1, vectorWidth)) * k21 +
                            new Vector<float>(input.Slice(inputRow2 + ox + 2, vectorWidth)) * k22;

                        acc.CopyTo(output.Slice(outputRow + ox, vectorWidth));
                    }

                    for (; ox < outW; ox++)
                    {
                        output[outputRow + ox] =
                            input[inputRow0 + ox] * kernels[kernelBase + 0] +
                            input[inputRow0 + ox + 1] * kernels[kernelBase + 1] +
                            input[inputRow0 + ox + 2] * kernels[kernelBase + 2] +
                            input[inputRow1 + ox] * kernels[kernelBase + 3] +
                            input[inputRow1 + ox + 1] * kernels[kernelBase + 4] +
                            input[inputRow1 + ox + 2] * kernels[kernelBase + 5] +
                            input[inputRow2 + ox] * kernels[kernelBase + 6] +
                            input[inputRow2 + ox + 1] * kernels[kernelBase + 7] +
                            input[inputRow2 + ox + 2] * kernels[kernelBase + 8];
                    }
                }
            }
        }

        private static void ForwardValidSingleChannel3x3Scalar(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int outChannels,
            int inputW,
            int outH,
            int outW)
        {
            for (var oc = 0; oc < outChannels; oc++)
            {
                var kernelBase = oc * 9;
                var outputChannelBase = oc * outH * outW;

                var k00 = kernels[kernelBase + 0];
                var k01 = kernels[kernelBase + 1];
                var k02 = kernels[kernelBase + 2];
                var k10 = kernels[kernelBase + 3];
                var k11 = kernels[kernelBase + 4];
                var k12 = kernels[kernelBase + 5];
                var k20 = kernels[kernelBase + 6];
                var k21 = kernels[kernelBase + 7];
                var k22 = kernels[kernelBase + 8];

                for (var oy = 0; oy < outH; oy++)
                {
                    var inputRow0 = oy * inputW;
                    var inputRow1 = (oy + 1) * inputW;
                    var inputRow2 = (oy + 2) * inputW;
                    var outputRow = outputChannelBase + oy * outW;

                    for (var ox = 0; ox < outW; ox++)
                    {
                        output[outputRow + ox] =
                            input[inputRow0 + ox] * k00 +
                            input[inputRow0 + ox + 1] * k01 +
                            input[inputRow0 + ox + 2] * k02 +
                            input[inputRow1 + ox] * k10 +
                            input[inputRow1 + ox + 1] * k11 +
                            input[inputRow1 + ox + 2] * k12 +
                            input[inputRow2 + ox] * k20 +
                            input[inputRow2 + ox + 1] * k21 +
                            input[inputRow2 + ox + 2] * k22;
                    }
                }
            }
        }


        // ─────────────────────────────────────────────────────────────────────
        // ForwardNchw — general Conv2D with padding and stride (ONNX-4)
        //
        // Replaces ForwardValidNchw for ONNX-imported models that use padding/stride.
        // ForwardValidNchw (padding=0, stride=1) is kept as-is for training path
        // backward compatibility.
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// General 2-D convolution forward pass (NCHW layout).
        /// Supports symmetric zero-padding and stride ≥ 1.
        /// Square kernel assumed (kH == kW == kernelSize).
        ///
        /// Output dimensions:
        ///   outH = (inputH + 2*padding - kernelSize) / stride + 1
        ///   outW = (inputW + 2*padding - kernelSize) / stride + 1
        /// </summary>
        public static void ForwardNchw(
            ReadOnlySpan<float> input,   // [batch, inC, inputH, inputW]
            ReadOnlySpan<float> kernels, // [outC, inC, kSize, kSize]
            Span<float> output,          // [batch, outC, outH, outW]
            int batchSize,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize,
            int padding,
            int stride)
        {
            // 3x3 stride-1 (VGG/ResNet's bulk) → Winograd F(2,3): ~2.25x fewer multiplies than im2col.
            if (kernelSize == 3 && stride == 1 && Conv2DWinogradKernels.IsSupported)
            {
                Conv2DWinogradKernels.Forward(
                    input, kernels, output, batchSize, inChannels, outChannels, inputH, inputW, padding);
                return;
            }

            // im2col + register-blocked GEMM closes most of the gap to a native conv on real-sized models;
            // it subsumes every stride/padding via the patch gather. Falls back to the direct-conv SIMD workers
            // when AVX2+FMA is unavailable (or for tiny convs where the im2col overhead would not pay).
            if (Conv2DGemmKernels.IsSupported)
            {
                Conv2DGemmKernels.Forward(
                    input, kernels, output, batchSize, inChannels, outChannels, inputH, inputW, kernelSize, padding, stride);
                return;
            }

            var outH = (inputH + 2 * padding - kernelSize) / stride + 1;
            var outW = (inputW + 2 * padding - kernelSize) / stride + 1;
            var inputPlaneSize = inChannels * inputH * inputW;
            var outputPlaneSize = outChannels * outH * outW;
            var kernelSizePerOutput = inChannels * kernelSize * kernelSize;

            output.Clear();

            for (var n = 0; n < batchSize; n++)
            {
                var inputBatch = input.Slice(n * inputPlaneSize, inputPlaneSize);
                var outputBatch = output.Slice(n * outputPlaneSize, outputPlaneSize);

                ForwardNchwSingleBatch(
                    inputBatch, kernels, outputBatch,
                    inChannels, outChannels,
                    inputH, inputW,
                    kernelSize, outH, outW,
                    kernelSizePerOutput, padding, stride);
            }
        }

        // Output channels are independent, so the conv parallelises trivially over them — a big win for the
        // (single-threaded, scalar) ONNX CNN path. Zero managed allocations: dispatch via the fn-pointer
        // OverfitParallel.For with a pointer context (OverfitParallel runs the body inline when parallelism is
        // suppressed — e.g. inside a data-parallel replica — so nesting stays safe).
        private static unsafe void ForwardNchwSingleBatch(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output,
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize,
            int outH,
            int outW,
            int kernelSizePerOutput,
            int padding,
            int stride)
        {
            fixed (float* pIn = input, pK = kernels, pOut = output)
            {
                var ctx = new ConvNchwCtx(
                    pIn, pK, pOut, inChannels, inputH, inputW, kernelSize, outH, outW, kernelSizePerOutput, padding, stride);

                // Unit stride → the per-(ky,ic,kx) contribution to an output ROW is a contiguous scalar·vector
                // FMA (acc[ox] += w·in[ix0+ox]), so we SIMD-accumulate over the long output-width dimension.
                // Strided convs gather non-contiguously, so they keep the scalar worker.
                delegate*<int, int, void*, void> worker =
                    stride == 1 ? &ConvNchwStride1SimdWorker
                    : CpuFeatures.HasAvx2 ? &ConvNchwStridedSimdWorker
                    : &ConvNchwOutputChannelWorker;
                OverfitParallel.For(0, outChannels, 1, worker, &ctx);
            }
        }

        // SIMD output-channel worker for unit-stride conv: accumulates each output row in a scratch buffer via
        // contiguous scalar·vector FMA (TensorPrimitives.MultiplyAdd), then writes the row. The reduction order
        // differs from the scalar worker (per-row vs per-pixel), so results match within float tolerance, not bit-
        // identically — fine for inference (parity tests use 1e-4).
        private static unsafe void ConvNchwStride1SimdWorker(int ocStart, int ocEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<ConvNchwCtx>(ctxPtr);
            var outW = c.OutW;

            using var pooled = outW <= 2048 ? default : new PooledBuffer<float>(outW, clearMemory: false);
            var acc = outW <= 2048 ? stackalloc float[outW] : pooled.Span;

            for (var oc = ocStart; oc < ocEnd; oc++)
            {
                var kernelBase = oc * c.KPerOut;
                var outChanBase = oc * c.OutH * outW;

                for (var oy = 0; oy < c.OutH; oy++)
                {
                    acc.Clear();
                    var iyBase = oy - c.Pad; // stride == 1

                    for (var ic = 0; ic < c.InC; ic++)
                    {
                        var inChanBase = ic * c.InputH * c.InputW;
                        var kChanBase = kernelBase + ic * c.KSize * c.KSize;

                        for (var ky = 0; ky < c.KSize; ky++)
                        {
                            var iy = iyBase + ky;
                            if ((uint)iy >= (uint)c.InputH)
                            {
                                continue; // padded row
                            }

                            var inRowBase = inChanBase + iy * c.InputW;
                            var kRowBase = kChanBase + ky * c.KSize;

                            for (var kx = 0; kx < c.KSize; kx++)
                            {
                                var w = c.K[kRowBase + kx];
                                var off = kx - c.Pad; // input x for output ox is ox + off

                                // Valid ox range: ox + off in [0, inputW).
                                var oxLo = off < 0 ? -off : 0;
                                var oxHi = c.InputW - off;
                                if (oxHi > outW)
                                {
                                    oxHi = outW;
                                }
                                if (oxLo >= oxHi)
                                {
                                    continue;
                                }

                                var len = oxHi - oxLo;
                                var inSpan = new ReadOnlySpan<float>(c.In + inRowBase + oxLo + off, len);
                                var accSpan = acc.Slice(oxLo, len);
                                TensorPrimitives.MultiplyAdd(inSpan, w, accSpan, accSpan);
                            }
                        }
                    }

                    acc.CopyTo(new Span<float>(c.Out + outChanBase + oy * outW, outW));
                }
            }
        }

        private readonly unsafe struct ConvNchwCtx
        {
            public readonly float* In;
            public readonly float* K;
            public readonly float* Out;
            public readonly int InC;
            public readonly int InputH;
            public readonly int InputW;
            public readonly int KSize;
            public readonly int OutH;
            public readonly int OutW;
            public readonly int KPerOut;
            public readonly int Pad;
            public readonly int Stride;

            public ConvNchwCtx(
                float* input, float* kernels, float* output, int inC, int inputH, int inputW, int kSize,
                int outH, int outW, int kPerOut, int pad, int stride)
            {
                In = input; K = kernels; Out = output;
                InC = inC; InputH = inputH; InputW = inputW; KSize = kSize;
                OutH = outH; OutW = outW; KPerOut = kPerOut; Pad = pad; Stride = stride;
            }
        }

        // SIMD output-channel worker for STRIDED conv (stride > 1, AVX2). The output-row input reads are strided
        // (in[ox·stride + off]), so we use a vector gather to load 8 strided inputs at once, then FMA-accumulate
        // into the output-row scratch. Border lanes are handled by restricting the vectorized span to fully
        // in-bounds [oxLo, oxHi) blocks with a scalar tail. Same per-row reduction order as the unit-stride SIMD
        // worker (tolerance-equal, not bit-identical to scalar).
        private static unsafe void ConvNchwStridedSimdWorker(int ocStart, int ocEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<ConvNchwCtx>(ctxPtr);
            var outW = c.OutW;
            var stride = c.Stride;

            using var pooled = outW <= 2048 ? default : new PooledBuffer<float>(outW, clearMemory: false);
            var acc = outW <= 2048 ? stackalloc float[outW] : pooled.Span;

            // Gather index lanes {0, stride, 2·stride, …, 7·stride} (element offsets; gather scale = 4 bytes).
            var idx = Vector256.Create(0, stride, 2 * stride, 3 * stride, 4 * stride, 5 * stride, 6 * stride, 7 * stride);

            for (var oc = ocStart; oc < ocEnd; oc++)
            {
                var kernelBase = oc * c.KPerOut;
                var outChanBase = oc * c.OutH * outW;

                for (var oy = 0; oy < c.OutH; oy++)
                {
                    acc.Clear();
                    var iyBase = oy * stride - c.Pad;

                    for (var ic = 0; ic < c.InC; ic++)
                    {
                        var inChanBase = ic * c.InputH * c.InputW;
                        var kChanBase = kernelBase + ic * c.KSize * c.KSize;

                        for (var ky = 0; ky < c.KSize; ky++)
                        {
                            var iy = iyBase + ky;
                            if ((uint)iy >= (uint)c.InputH)
                            {
                                continue;
                            }

                            var inRowBase = inChanBase + iy * c.InputW;
                            var kRowBase = kChanBase + ky * c.KSize;

                            for (var kx = 0; kx < c.KSize; kx++)
                            {
                                var w = c.K[kRowBase + kx];
                                var off = kx - c.Pad; // input x for output ox is ox·stride + off

                                // In-bounds output range: ox·stride + off in [0, inputW). A few-iteration clamp
                                // avoids signed ceil/floor-division pitfalls.
                                var oxLo = 0;
                                while (oxLo * stride + off < 0)
                                {
                                    oxLo++;
                                }
                                var oxHi = outW;
                                while (oxHi > oxLo && (oxHi - 1) * stride + off >= c.InputW)
                                {
                                    oxHi--;
                                }
                                if (oxLo >= oxHi)
                                {
                                    continue;
                                }

                                var wVec = Vector256.Create(w);
                                var ox = oxLo;
                                for (; ox + 8 <= oxHi; ox += 8)
                                {
                                    var gathered = Avx2.GatherVector256(c.In + inRowBase + ox * stride + off, idx, 4);
                                    var accSlice = acc.Slice(ox, 8);
                                    var accVec = Avx.Add(Vector256.Create((ReadOnlySpan<float>)accSlice), Avx.Multiply(wVec, gathered));
                                    accVec.CopyTo(accSlice);
                                }
                                for (; ox < oxHi; ox++)
                                {
                                    acc[ox] += w * c.In[inRowBase + ox * stride + off];
                                }
                            }
                        }
                    }

                    acc.CopyTo(new Span<float>(c.Out + outChanBase + oy * outW, outW));
                }
            }
        }

        private static unsafe void ConvNchwOutputChannelWorker(int ocStart, int ocEnd, void* ctxPtr)
        {
            ref readonly var c = ref Unsafe.AsRef<ConvNchwCtx>(ctxPtr);

            for (var oc = ocStart; oc < ocEnd; oc++)
            {
                var kernelBase = oc * c.KPerOut;
                var outputChanBase = oc * c.OutH * c.OutW;

                for (var oy = 0; oy < c.OutH; oy++)
                {
                    var inputYBase = oy * c.Stride - c.Pad;

                    for (var ox = 0; ox < c.OutW; ox++)
                    {
                        var inputXBase = ox * c.Stride - c.Pad;
                        var sum = 0f;

                        for (var ic = 0; ic < c.InC; ic++)
                        {
                            var inputChanBase = ic * c.InputH * c.InputW;
                            var kernelChanBase = kernelBase + ic * c.KSize * c.KSize;

                            for (var ky = 0; ky < c.KSize; ky++)
                            {
                                var iy = inputYBase + ky;
                                if ((uint)iy >= (uint)c.InputH)
                                {
                                    continue; // zero-pad: skip out-of-bounds rows
                                }

                                var kernelRowBase = kernelChanBase + ky * c.KSize;
                                var inputRowBase = inputChanBase + iy * c.InputW;

                                for (var kx = 0; kx < c.KSize; kx++)
                                {
                                    var ix = inputXBase + kx;
                                    if ((uint)ix >= (uint)c.InputW)
                                    {
                                        continue; // zero-pad: skip out-of-bounds cols
                                    }

                                    sum += c.In[inputRowBase + ix] * c.K[kernelRowBase + kx];
                                }
                            }
                        }

                        c.Out[outputChanBase + oy * c.OutW + ox] = sum;
                    }
                }
            }
        }

    }
}