// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // CONV2D
        // ====================================================================

        public static AutogradNode Conv2D(ComputationGraph graph, AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Shape.D0, kSqInC = k * k * inC, colS = kSqInC * outH * outW;

            var outputNode = AllocateNode(graph, new TensorShape(batchSize, outC, outH, outW), input.RequiresGrad || weights.RequiresGrad, clearMemory: true);

            Parallel.For(0, batchSize,
                () => new TensorStorage<float>(colS, false),
                (n, loopState, localWorkspace) =>
                {
                    var w2D = weights.DataView.Reshape(new TensorShape(outC, kSqInC));
                    var colS_n = localWorkspace.AsSpan();

                    Im2Col(input.DataView.AsReadOnlySpan().Slice(n * inC * h * w, inC * h * w), inC, h, w, k, 1, 0, colS_n);
                    MatMulRawSeq(w2D.AsReadOnlySpan(), colS_n, outC, kSqInC, outH * outW, outputNode.DataView.AsSpan().Slice(n * outC * outH * outW, outC * outH * outW));

                    return localWorkspace;
                },
                localWorkspace => localWorkspace.Dispose()
            );

            if (outputNode.RequiresGrad)
            {
                graph?.Record(OpCode.Conv2D, outputNode, input, weights, inC, outC, h, w, k);
            }

            return outputNode;
        }

        public static void Conv2DBackward(AutogradNode input, AutogradNode weights, AutogradNode output, int inC, int outC, int h, int w, int k)
        {
            if (!input.RequiresGrad && !weights.RequiresGrad)
            {
                return;
            }

            int batchSize = input.Shape.D0, outH = h - k + 1, outW = w - k + 1, K = outH * outW, kSqInC = k * k * inC;

            var partialWeightGrads = weights.RequiresGrad ? new ConcurrentBag<TensorStorage<float>>() : null;

            Parallel.For(0, batchSize,
                () => weights.RequiresGrad ? new TensorStorage<float>(outC * kSqInC, clearMemory: true) : null,
                (n, loopState, localDw) =>
                {
                    using var colS = new PooledBuffer<float>(kSqInC * K);

                    Im2Col(input.DataView.AsReadOnlySpan().Slice(n * inC * h * w, inC * h * w), inC, h, w, k, 1, 0, colS.Span);

                    var outGS = output.GradView.AsReadOnlySpan().Slice(n * outC * K, outC * K);

                    if (weights.RequiresGrad && localDw != null)
                    {
                        MatMulAdd_A_BT_Seq(outGS, colS.Span, localDw.AsSpan(), outC, K, kSqInC);
                    }

                    if (input.RequiresGrad)
                    {
                        using var dColS = new PooledBuffer<float>(kSqInC * K);
                        var w2D = weights.DataView.Reshape(new TensorShape(outC, kSqInC));
                        MatMulAdd_AT_B_Seq(w2D.AsReadOnlySpan(), outGS, dColS.Span, outC, kSqInC, K);
                        Col2Im(dColS.Span, inC, h, w, k, 1, 0, input.GradView.AsSpan().Slice(n * inC * h * w, inC * h * w));
                    }

                    return localDw;
                },
                localDw =>
                {
                    if (localDw != null)
                    {
                        partialWeightGrads!.Add(localDw);
                    }
                });

            if (weights.RequiresGrad && partialWeightGrads != null)
            {
                var weightsGrad = weights.GradView.AsSpan();
                foreach (var partial in partialWeightGrads)
                {
                    try
                    {
                        Simd.Add(weightsGrad, partial.AsReadOnlySpan(), weightsGrad);
                    }
                    finally
                    {
                        partial.Dispose();
                    }
                }
            }
        }

        // ====================================================================
        // IM2COL / COL2IM (BEZ ZMIAN)
        // ====================================================================

        public static void Im2Col(ReadOnlySpan<float> input, int channels, int h, int w, int k, int s, int p, Span<float> output)
        {
            int oH = (h + 2 * p - k) / s + 1, oW = (w + 2 * p - k) / s + 1;
            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < k; kh++)
                {
                    for (var kw = 0; kw < k; kw++)
                    {
                        var rO = (c * k * k + kh * k + kw) * oH * oW;
                        for (var y = 0; y < oH; y++)
                        {
                            var i = y * s - p + kh;
                            if (i >= 0 && i < h)
                            {
                                var iO = c * h * w + i * w;
                                for (var x = 0; x < oW; x++)
                                {
                                    var j = x * s - p + kw;
                                    output[rO + y * oW + x] = j >= 0 && j < w ? input[iO + j] : 0f;
                                }
                            }
                            else
                            {
                                output.Slice(rO + y * oW, oW).Clear();
                            }
                        }
                    }
                }
            }
        }

        public static void Col2Im(ReadOnlySpan<float> col, int channels, int h, int w, int k, int s, int p, Span<float> gI)
        {
            int oH = (h + 2 * p - k) / s + 1, oW = (w + 2 * p - k) / s + 1;
            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < k; kh++)
                {
                    for (var kw = 0; kw < k; kw++)
                    {
                        var rO = (c * k * k + kh * k + kw) * oH * oW;
                        for (var y = 0; y < oH; y++)
                        {
                            var i = y * s - p + kh;
                            if (i >= 0 && i < h)
                            {
                                var iO = c * h * w + i * w;
                                for (var x = 0; x < oW; x++)
                                {
                                    var j = x * s - p + kw;
                                    if (j >= 0 && j < w)
                                    {
                                        gI[iO + j] += col[rO + y * oW + x];
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