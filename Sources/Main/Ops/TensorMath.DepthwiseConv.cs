// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // DEPTHWISE CONV2D (groups == channels — the MobileNet building block)
        // ====================================================================

        /// <summary>
        /// Depthwise convolution: each of the <paramref name="channels"/> input channels is convolved
        /// with its <b>own</b> k×k filter (no cross-channel mixing), so cost is <c>C·k²·outHW</c> vs a
        /// full conv's <c>C²·k²·outHW</c> — the cheap spatial half of a depthwise-separable (MobileNet)
        /// block. Input <c>[N, C, H, W]</c>, kernel <c>[C, k·k]</c>, output <c>[N, C, outH, outW]</c>.
        /// The hot path is a SIMD AXPY (<see cref="Simd.MulAdd"/>) over each output row (stride 1);
        /// strided convs fall back to a scalar inner loop.
        /// </summary>
        public static AutogradNode DepthwiseConv2D(
            ComputationGraph graph, AutogradNode input, AutogradNode kernel,
            int channels, int h, int w, int k, int padding, int stride, AutogradNode bias)
        {
            var n = input.Shape.D0;
            var outH = (h + 2 * padding - k) / stride + 1;
            var outW = (w + 2 * padding - k) / stride + 1;
            var kk = k * k;
            var hw = h * w;
            var outHW = outH * outW;

            var requiresGrad = input.RequiresGrad || kernel.RequiresGrad || (bias is not null && bias.RequiresGrad);
            var output = AllocateNode(graph, new TensorShape(n, channels, outH, outW), requiresGrad, clearMemory: true);

            var inS = input.DataView.AsReadOnlySpan();
            var kS = kernel.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();
            var hasBias = bias is not null;
            var bS = hasBias ? bias.DataView.AsReadOnlySpan() : default;

            for (var ni = 0; ni < n; ni++)
            {
                for (var c = 0; c < channels; c++)
                {
                    var outPlane = outS.Slice((ni * channels + c) * outHW, outHW);
                    if (hasBias)
                    {
                        outPlane.Fill(bS[c]);
                    }

                    var inPlane = inS.Slice((ni * channels + c) * hw, hw);
                    var kBase = c * kk;

                    for (var ky = 0; ky < k; ky++)
                    {
                        for (var kx = 0; kx < k; kx++)
                        {
                            var kVal = kS[kBase + ky * k + kx];
                            for (var oy = 0; oy < outH; oy++)
                            {
                                var inY = oy * stride - padding + ky;
                                if (inY < 0 || inY >= h)
                                {
                                    continue;
                                }

                                var outRow = outPlane.Slice(oy * outW, outW);
                                var inRow = inPlane.Slice(inY * w, w);

                                if (stride == 1)
                                {
                                    var lo = Math.Max(0, padding - kx);
                                    var hi = Math.Min(outW, w + padding - kx);
                                    if (hi > lo)
                                    {
                                        Simd.MulAdd(inRow.Slice(lo - padding + kx, hi - lo), kVal, outRow.Slice(lo, hi - lo));
                                    }
                                }
                                else
                                {
                                    for (var ox = 0; ox < outW; ox++)
                                    {
                                        var inX = ox * stride - padding + kx;
                                        if (inX >= 0 && inX < w)
                                        {
                                            outRow[ox] += kVal * inRow[inX];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (requiresGrad)
            {
                graph?.Record(OpCode.DepthwiseConv2D, output, input, kernel, PackConvParams(k, padding, stride), c0: bias);
            }

            return output;
        }

        public static void DepthwiseConv2DBackward(
            ComputationGraph graph, AutogradNode input, AutogradNode kernel, AutogradNode output, AutogradNode bias,
            int channels, int h, int w, int k, int padding, int stride)
        {
            var biasNeedsGrad = bias is not null && bias.RequiresGrad;
            if (!input.RequiresGrad && !kernel.RequiresGrad && !biasNeedsGrad)
            {
                return;
            }

            var n = input.Shape.D0;
            var outH = output.Shape.D2;
            var outW = output.Shape.D3;
            var kk = k * k;
            var hw = h * w;
            var outHW = outH * outW;

            var inS = input.DataView.AsReadOnlySpan();
            var kS = kernel.DataView.AsReadOnlySpan();
            var outGradS = output.GradView.AsReadOnlySpan();

            if (biasNeedsGrad)
            {
                var biasGrad = bias.GradView.AsSpan();
                for (var ni = 0; ni < n; ni++)
                {
                    for (var c = 0; c < channels; c++)
                    {
                        biasGrad[c] += TensorPrimitives.Sum(outGradS.Slice((ni * channels + c) * outHW, outHW));
                    }
                }
            }

            var wantInput = input.RequiresGrad;
            var wantKernel = kernel.RequiresGrad;
            if (!wantInput && !wantKernel)
            {
                return;
            }

            var inGradS = wantInput ? input.GradView.AsSpan() : default;
            var kGradS = wantKernel ? kernel.GradView.AsSpan() : default;

            for (var ni = 0; ni < n; ni++)
            {
                for (var c = 0; c < channels; c++)
                {
                    var outGradPlane = outGradS.Slice((ni * channels + c) * outHW, outHW);
                    var inPlane = inS.Slice((ni * channels + c) * hw, hw);
                    var inGradPlane = wantInput ? inGradS.Slice((ni * channels + c) * hw, hw) : default;
                    var kBase = c * kk;

                    for (var ky = 0; ky < k; ky++)
                    {
                        for (var kx = 0; kx < k; kx++)
                        {
                            var kVal = kS[kBase + ky * k + kx];
                            var kIdx = kBase + ky * k + kx;

                            for (var oy = 0; oy < outH; oy++)
                            {
                                var inY = oy * stride - padding + ky;
                                if (inY < 0 || inY >= h)
                                {
                                    continue;
                                }

                                var ogRow = outGradPlane.Slice(oy * outW, outW);
                                var inRow = inPlane.Slice(inY * w, w);

                                if (stride == 1)
                                {
                                    var lo = Math.Max(0, padding - kx);
                                    var hi = Math.Min(outW, w + padding - kx);
                                    if (hi <= lo)
                                    {
                                        continue;
                                    }
                                    var len = hi - lo;
                                    var inOff = inY * w + (lo - padding + kx);
                                    if (wantKernel)
                                    {
                                        kGradS[kIdx] += Simd.Dot(ogRow.Slice(lo, len), inRow.Slice(lo - padding + kx, len));
                                    }
                                    if (wantInput)
                                    {
                                        Simd.MulAdd(ogRow.Slice(lo, len), kVal, inGradPlane.Slice(inOff, len));
                                    }
                                }
                                else
                                {
                                    for (var ox = 0; ox < outW; ox++)
                                    {
                                        var inX = ox * stride - padding + kx;
                                        if (inX < 0 || inX >= w)
                                        {
                                            continue;
                                        }
                                        if (wantKernel)
                                        {
                                            kGradS[kIdx] += ogRow[ox] * inRow[inX];
                                        }
                                        if (wantInput)
                                        {
                                            inGradPlane[inY * w + inX] += kVal * ogRow[ox];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
