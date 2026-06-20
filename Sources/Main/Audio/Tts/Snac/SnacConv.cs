// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// The new primitive the SNAC codec decoder (TTS, S2) is built on: <b>1-D transposed convolution</b>
    /// (PyTorch <c>ConvTranspose1d</c>) — the learned-upsampling op that turns a low-rate latent sequence into a
    /// higher-rate waveform stage. Plain-span, no autograd; parallel over output channels above a work threshold
    /// (matching <c>WhisperKernels</c>). Model-free and exactly testable, so it can be validated before any SNAC
    /// weights are on the box.
    /// </summary>
    internal static unsafe class SnacConv
    {
        // Parallelize only when total work (outC·tOut·inC·kSize) clears this; below it, dispatch overhead wins.
        private const long ParallelThreshold = 1 << 18; // 262144

        /// <summary>Output length of a transposed conv: <c>(tIn−1)·stride − 2·pad + dilation·(kSize−1) +
        /// outputPadding + 1</c> (PyTorch <c>ConvTranspose1d</c>).</summary>
        public static int OutputLength(int tIn, int kSize, int stride, int pad, int dilation, int outputPadding)
            => ((tIn - 1) * stride) - (2 * pad) + (dilation * (kSize - 1)) + outputPadding + 1;

        /// <summary>Output length of a plain conv: <c>(tIn + 2·pad − dilation·(kSize−1) − 1)/stride + 1</c>
        /// (PyTorch <c>Conv1d</c>).</summary>
        public static int ConvOutputLength(int tIn, int kSize, int stride, int pad, int dilation)
            => ((tIn + (2 * pad) - (dilation * (kSize - 1)) - 1) / stride) + 1;

        /// <summary>
        /// 1-D convolution over time with <paramref name="groups"/> support (SNAC's residual units and the
        /// decoder stem use <b>depthwise</b> convs, i.e. <c>groups == channels</c>). <paramref name="input"/> is
        /// channel-major <c>[inC × tIn]</c>; <paramref name="weight"/> is <c>[outC × (inC/groups) × kSize]</c>
        /// (PyTorch <c>Conv1d</c> layout); output channel-major <c>[outC × tOut]</c>. Each output channel reads
        /// only its group's input channels. Parallel over output channels above a work threshold.
        /// </summary>
        public static void Conv1d(
            ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> dst, int inC, int tIn, int outC, int kSize, int stride, int pad, int dilation, int groups, int tOut)
        {
            var icPerGroup = inC / groups;
            var ocPerGroup = outC / groups;
            if ((long)outC * tOut * icPerGroup * kSize < ParallelThreshold)
            {
                for (var oc = 0; oc < outC; oc++)
                {
                    Conv1dChannel(oc, input, weight, bias, dst, tIn, kSize, stride, pad, dilation, icPerGroup, ocPerGroup, tOut);
                }
                return;
            }

            fixed (float* ip = input, wp = weight, bp = bias, dp = dst)
            {
                var ctx = new Conv1dCtx(ip, wp, bias.IsEmpty ? null : bp, dp,
                    inC, tIn, outC, kSize, stride, pad, dilation, icPerGroup, ocPerGroup, tOut);
                OverfitParallel.For(0, outC, &Conv1dWorker, &ctx);
            }
        }

        private static void Conv1dChannel(
            int oc, ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> dst, int tIn, int kSize, int stride, int pad, int dilation, int icPerGroup, int ocPerGroup, int tOut)
        {
            var b0 = bias.IsEmpty ? 0f : bias[oc];
            var group = oc / ocPerGroup;
            var icStart = group * icPerGroup;
            for (var to = 0; to < tOut; to++)
            {
                var acc = b0;
                var start = (to * stride) - pad;
                for (var icl = 0; icl < icPerGroup; icl++)
                {
                    var wBase = ((oc * icPerGroup) + icl) * kSize;
                    var inBase = (icStart + icl) * tIn;
                    for (var k = 0; k < kSize; k++)
                    {
                        var ti = start + (k * dilation);
                        if (ti >= 0 && ti < tIn)
                        {
                            acc += weight[wBase + k] * input[inBase + ti];
                        }
                    }
                }
                dst[(oc * tOut) + to] = acc;
            }
        }

        private readonly struct Conv1dCtx
        {
            public readonly float* In, W, B, D;
            public readonly int InC, TIn, OutC, KSize, Stride, Pad, Dilation, IcPerGroup, OcPerGroup, TOut;

            public Conv1dCtx(float* inp, float* w, float* b, float* d, int inC, int tIn, int outC,
                int kSize, int stride, int pad, int dilation, int icPerGroup, int ocPerGroup, int tOut)
            {
                In = inp;
                W = w;
                B = b;
                D = d;
                InC = inC;
                TIn = tIn;
                OutC = outC;
                KSize = kSize;
                Stride = stride;
                Pad = pad;
                Dilation = dilation;
                IcPerGroup = icPerGroup;
                OcPerGroup = ocPerGroup;
                TOut = tOut;
            }
        }

        private static void Conv1dWorker(int start, int end, void* ctxPtr)
        {
            ref var c = ref Unsafe.AsRef<Conv1dCtx>(ctxPtr);
            var input = new ReadOnlySpan<float>(c.In, c.InC * c.TIn);
            var weight = new ReadOnlySpan<float>(c.W, c.OutC * c.IcPerGroup * c.KSize);
            var bias = c.B == null ? ReadOnlySpan<float>.Empty : new ReadOnlySpan<float>(c.B, c.OutC);
            var dst = new Span<float>(c.D, c.OutC * c.TOut);
            for (var oc = start; oc < end; oc++)
            {
                Conv1dChannel(oc, input, weight, bias, dst, c.TIn, c.KSize, c.Stride, c.Pad, c.Dilation, c.IcPerGroup, c.OcPerGroup, c.TOut);
            }
        }

        /// <summary>
        /// 1-D transposed convolution over time. <paramref name="input"/> is channel-major <c>[inC × tIn]</c>;
        /// <paramref name="weight"/> is <c>[inC × outC × kSize]</c> (PyTorch transposed-conv weight layout —
        /// in-channels outermost); output is channel-major <c>[outC × tOut]</c>. Gather formulation: each output
        /// sample sums every input/kernel pair that maps onto it (<c>to = ti·stride + k·dilation − pad</c>), so it
        /// is race-free to parallelize over output channels.
        /// </summary>
        public static void ConvTranspose1d(
            ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> dst, int inC, int tIn, int outC, int kSize, int stride, int pad, int dilation, int tOut)
        {
            if ((long)outC * tOut * inC * kSize < ParallelThreshold)
            {
                for (var oc = 0; oc < outC; oc++)
                {
                    ConvTranspose1dChannel(oc, input, weight, bias, dst, inC, tIn, outC, kSize, stride, pad, dilation, tOut);
                }
                return;
            }

            fixed (float* ip = input, wp = weight, bp = bias, dp = dst)
            {
                var ctx = new ConvTCtx(ip, wp, bias.IsEmpty ? null : bp, dp, inC, tIn, outC, kSize, stride, pad, dilation, tOut);
                OverfitParallel.For(0, outC, &ConvTWorker, &ctx);
            }
        }

        private static void ConvTranspose1dChannel(
            int oc, ReadOnlySpan<float> input, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
            Span<float> dst, int inC, int tIn, int outC, int kSize, int stride, int pad, int dilation, int tOut)
        {
            var b = bias.IsEmpty ? 0f : bias[oc];
            for (var to = 0; to < tOut; to++)
            {
                var acc = b;
                for (var k = 0; k < kSize; k++)
                {
                    var num = to + pad - (k * dilation);
                    if (num < 0 || (num % stride) != 0)
                    {
                        continue;
                    }
                    var ti = num / stride;
                    if (ti >= tIn)
                    {
                        continue;
                    }
                    for (var ic = 0; ic < inC; ic++)
                    {
                        acc += weight[((ic * outC) + oc) * kSize + k] * input[(ic * tIn) + ti];
                    }
                }
                dst[(oc * tOut) + to] = acc;
            }
        }

        private readonly struct ConvTCtx
        {
            public readonly float* In, W, B, D;
            public readonly int InC, TIn, OutC, KSize, Stride, Pad, Dilation, TOut;

            public ConvTCtx(float* inp, float* w, float* b, float* d,
                int inC, int tIn, int outC, int kSize, int stride, int pad, int dilation, int tOut)
            {
                In = inp;
                W = w;
                B = b;
                D = d;
                InC = inC;
                TIn = tIn;
                OutC = outC;
                KSize = kSize;
                Stride = stride;
                Pad = pad;
                Dilation = dilation;
                TOut = tOut;
            }
        }

        private static void ConvTWorker(int start, int end, void* ctxPtr)
        {
            ref var c = ref Unsafe.AsRef<ConvTCtx>(ctxPtr);
            var input = new ReadOnlySpan<float>(c.In, c.InC * c.TIn);
            var weight = new ReadOnlySpan<float>(c.W, c.InC * c.OutC * c.KSize);
            var bias = c.B == null ? ReadOnlySpan<float>.Empty : new ReadOnlySpan<float>(c.B, c.OutC);
            var dst = new Span<float>(c.D, c.OutC * c.TOut);
            for (var oc = start; oc < end; oc++)
            {
                ConvTranspose1dChannel(oc, input, weight, bias, dst, c.InC, c.TIn, c.OutC, c.KSize, c.Stride, c.Pad, c.Dilation, c.TOut);
            }
        }
    }
}
