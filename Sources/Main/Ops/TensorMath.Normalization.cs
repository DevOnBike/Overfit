// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // BATCH NORM 1D
        // ====================================================================

        public static AutogradNode BatchNorm1D(ComputationGraph graph, AutogradNode input, AutogradNode gamma, AutogradNode beta, TensorStorage<float> runningMean, TensorStorage<float> runningVar, float momentum, float eps, bool isTraining)
        {
            int N = input.Shape.D0, C = input.Shape.D1;

            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);

            // CRITICAL FIX: 'mean' MUST be cleared, because we are about to accumulate (Add) values into it!
            var mean = AllocateNode(graph, new TensorShape(C), false, clearMemory: true);
            var invStd = AllocateNode(graph, new TensorShape(C), false, clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();
            var meanS = mean.DataView.AsSpan();
            var invStdS = invStd.DataView.AsSpan();

            if (isTraining)
            {
                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Add(meanS, inS.Slice(i * C, C), meanS);
                }
                TensorPrimitives.Multiply(meanS, 1f / N, meanS);

                using var vB = new PooledBuffer<float>(C);
                using var tB = new PooledBuffer<float>(C, false);

                for (var i = 0; i < N; i++)
                {
                    TensorPrimitives.Subtract(inS.Slice(i * C, C), meanS, tB.Span);
                    TensorPrimitives.MultiplyAdd(tB.Span, tB.Span, vB.Span, vB.Span);
                }

                TensorPrimitives.Multiply(vB.Span, 1f / N, vB.Span);

                var rmS = runningMean.AsSpan();
                var rvS = runningVar.AsSpan();

                TensorPrimitives.Multiply(rmS, 1f - momentum, rmS);
                TensorPrimitives.MultiplyAdd(meanS, momentum, rmS, rmS);
                TensorPrimitives.Multiply(rvS, 1f - momentum, rvS);
                TensorPrimitives.MultiplyAdd(vB.Span, momentum, rvS, rvS);
                TensorPrimitives.Add(vB.Span, eps, invStdS);
                TensorPrimitives.ReciprocalSqrt(invStdS, invStdS);
            }
            else
            {
                runningMean.AsReadOnlySpan().CopyTo(meanS);
                TensorPrimitives.Add(runningVar.AsReadOnlySpan(), eps, invStdS);
                TensorPrimitives.ReciprocalSqrt(invStdS, invStdS);
            }

            var gS = gamma.DataView.AsReadOnlySpan();
            var bS = beta.DataView.AsReadOnlySpan();
            var roMeanS = mean.DataView.AsReadOnlySpan();
            var roInvS = invStd.DataView.AsReadOnlySpan();

            for (var i = 0; i < N; i++)
            {
                var oR = outS.Slice(i * C, C);
                TensorPrimitives.Subtract(inS.Slice(i * C, C), roMeanS, oR);
                TensorPrimitives.Multiply(oR, roInvS, oR);
                TensorPrimitives.MultiplyAdd(oR, gS, bS, oR);
            }

            if (output.RequiresGrad && isTraining)
            {
                graph?.Record(OpCode.BatchNorm1D, output, input, c0: gamma, c1: beta, c2: mean, c3: invStd, contextCount: 4);
            }
            else if (!isTraining)
            {
                mean.Dispose();
                invStd.Dispose();
            }

            return output;
        }

        public static void BatchNorm1DBackward(AutogradNode input, AutogradNode output, AutogradNode gamma, AutogradNode beta, AutogradNode mean, AutogradNode invStd)
        {
            if (!input.RequiresGrad && !gamma.RequiresGrad && !beta.RequiresGrad)
            {
                return;
            }

            int N = input.Shape.D0, C = input.Shape.D1;

            using var coeffBuf = new PooledBuffer<float>(C, false);
            var coeff = coeffBuf.Span;
            using var termBuf = new PooledBuffer<float>(C, false);
            var term = termBuf.Span;
            using var sDyBuf = new PooledBuffer<float>(C);
            var sDy = sDyBuf.Span;
            using var sDyXBuf = new PooledBuffer<float>(C);
            var sDyX = sDyXBuf.Span;
            using var xHRBuf = new PooledBuffer<float>(C, false);
            var xHR = xHRBuf.Span;

            var gammaS = gamma.DataView.AsReadOnlySpan();
            var invStdS = invStd.DataView.AsReadOnlySpan();
            var meanS = mean.DataView.AsReadOnlySpan();
            var outGradS = output.GradView.AsReadOnlySpan();
            var inDataS = input.DataView.AsReadOnlySpan();

            TensorPrimitives.Multiply(gammaS, invStdS, coeff);
            TensorPrimitives.Multiply(coeff, 1f / N, coeff);

            for (var i = 0; i < N; i++)
            {
                var gR = outGradS.Slice(i * C, C);
                var iR = inDataS.Slice(i * C, C);
                TensorPrimitives.Subtract(iR, meanS, xHR);
                TensorPrimitives.Multiply(xHR, invStdS, xHR);
                TensorPrimitives.Add(sDy, gR, sDy);
                TensorPrimitives.MultiplyAdd(gR, xHR, sDyX, sDyX);

                if (beta.RequiresGrad)
                {
                    TensorPrimitives.Add(beta.GradView.AsSpan(), gR, beta.GradView.AsSpan());
                }
                if (gamma.RequiresGrad)
                {
                    TensorPrimitives.MultiplyAdd(gR, xHR, gamma.GradView.AsSpan(), gamma.GradView.AsSpan());
                }
            }

            if (input.RequiresGrad)
            {
                var iGS = input.GradView.AsSpan();
                using var tempXHatBuf = new PooledBuffer<float>(C, false);
                var tempXHat = tempXHatBuf.Span;

                for (var i = 0; i < N; i++)
                {
                    var gR = outGradS.Slice(i * C, C);
                    var iGR = iGS.Slice(i * C, C);
                    var iR = inDataS.Slice(i * C, C);
                    TensorPrimitives.Subtract(iR, meanS, xHR);
                    TensorPrimitives.Multiply(xHR, invStdS, xHR);
                    TensorPrimitives.Multiply(gR, N, term);
                    TensorPrimitives.Subtract(term, sDy, term);

                    xHR.CopyTo(tempXHat);
                    TensorPrimitives.Multiply(tempXHat, sDyX, tempXHat);
                    TensorPrimitives.Subtract(term, tempXHat, term);
                    TensorPrimitives.MultiplyAdd(coeff, term, iGR, iGR);
                }
            }
        }

        // ====================================================================
        // BATCH NORM 2D (spatial — per-channel over N, H, W)
        // ====================================================================

        /// <summary>
        /// Spatial batch normalisation for conv feature maps <c>[N, C, H, W]</c>: statistics are pooled
        /// per channel over the N·H·W elements (vs <see cref="BatchNorm1D"/>'s per-feature over N). Same
        /// affine (γ, β per channel) + running-mean/var (EMA) used at inference.
        /// </summary>
        public static AutogradNode BatchNorm2D(
            ComputationGraph graph, AutogradNode input, AutogradNode gamma, AutogradNode beta,
            TensorStorage<float> runningMean, TensorStorage<float> runningVar, float momentum, float eps, bool isTraining)
        {
            int N = input.Shape.D0, C = input.Shape.D1, H = input.Shape.D2, W = input.Shape.D3;
            var hw = H * W;
            var m = (long)N * hw;

            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);
            var mean = AllocateNode(graph, new TensorShape(C), false, clearMemory: true);
            var invStd = AllocateNode(graph, new TensorShape(C), false, clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();
            var meanS = mean.DataView.AsSpan();
            var invStdS = invStd.DataView.AsSpan();

            if (isTraining)
            {
                for (var n = 0; n < N; n++)
                {
                    for (var c = 0; c < C; c++)
                    {
                        meanS[c] += TensorPrimitives.Sum(inS.Slice((n * C + c) * hw, hw));
                    }
                }
                for (var c = 0; c < C; c++)
                {
                    meanS[c] /= m;
                }

                using var varBuf = new PooledBuffer<float>(C);          // cleared
                using var tmpBuf = new PooledBuffer<float>(hw, false);
                var varS = varBuf.Span;

                for (var n = 0; n < N; n++)
                {
                    for (var c = 0; c < C; c++)
                    {
                        TensorPrimitives.Subtract(inS.Slice((n * C + c) * hw, hw), meanS[c], tmpBuf.Span);
                        varS[c] += TensorPrimitives.SumOfSquares(tmpBuf.Span);
                    }
                }

                var rmS = runningMean.AsSpan();
                var rvS = runningVar.AsSpan();

                for (var c = 0; c < C; c++)
                {
                    varS[c] /= m;
                    rmS[c] = rmS[c] * (1f - momentum) + meanS[c] * momentum;
                    rvS[c] = rvS[c] * (1f - momentum) + varS[c] * momentum;
                    invStdS[c] = 1f / MathF.Sqrt(varS[c] + eps);
                }
            }
            else
            {
                runningMean.AsReadOnlySpan().CopyTo(meanS);
                var rv = runningVar.AsReadOnlySpan();
                for (var c = 0; c < C; c++)
                {
                    invStdS[c] = 1f / MathF.Sqrt(rv[c] + eps);
                }
            }

            var gS = gamma.DataView.AsReadOnlySpan();
            var bS = beta.DataView.AsReadOnlySpan();



            for (var n = 0; n < N; n++)
            {
                for (var c = 0; c < C; c++)
                {
                    var scale = invStdS[c] * gS[c];
                    var shift = bS[c] - meanS[c] * scale;
                    var outB = outS.Slice((n * C + c) * hw, hw);
                    TensorPrimitives.Multiply(inS.Slice((n * C + c) * hw, hw), scale, outB);
                    TensorPrimitives.Add(outB, shift, outB);
                }
            }

            if (output.RequiresGrad && isTraining)
            {
                graph?.Record(OpCode.BatchNorm2D, output, input, c0: gamma, c1: beta, c2: mean, c3: invStd, contextCount: 4);
            }
            else if (!isTraining)
            {
                mean.Dispose();
                invStd.Dispose();
            }

            return output;
        }

        public static void BatchNorm2DBackward(
            AutogradNode input, AutogradNode output, AutogradNode gamma, AutogradNode beta, AutogradNode mean, AutogradNode invStd)
        {
            if (!input.RequiresGrad && !gamma.RequiresGrad && !beta.RequiresGrad)
            {
                return;
            }

            int N = input.Shape.D0, C = input.Shape.D1, H = input.Shape.D2, W = input.Shape.D3;
            var hw = H * W;
            var m = N * hw;

            var gammaS = gamma.DataView.AsReadOnlySpan();
            var invStdS = invStd.DataView.AsReadOnlySpan();
            var meanS = mean.DataView.AsReadOnlySpan();
            var outGradS = output.GradView.AsReadOnlySpan();
            var inS = input.DataView.AsReadOnlySpan();

            // Empirical size-gate (CIFAR vs MNIST profile, 2026-05-29): below ~200K total element-ops
            // per call, `Parallel.For(0, C)` spawn overhead ≥ wins (MNIST: N=32, C=8/16, hw=196/784
            // → 50-200K, no gain at all in measurements). At/above that threshold, channels are large
            // enough (CIFAR C=32+, hw=256+: >500K) that per-thread work amortizes spawn. Sequential
            // path preserved verbatim below for the small-shape case.
            var totalWork = (long)N * C * hw;
            const long ParallelThreshold = 200_000;

            if (totalWork < ParallelThreshold)
            {
                BatchNorm2DBackwardSequential(input, output, gamma, beta, mean, invStd, N, C, hw, m);
                return;
            }

            BatchNorm2DBackwardParallel(input, output, gamma, beta, mean, invStd, N, C, hw, m);
        }

        private static void BatchNorm2DBackwardSequential(
            AutogradNode input, AutogradNode output, AutogradNode gamma, AutogradNode beta,
            AutogradNode mean, AutogradNode invStd, int N, int C, int hw, int m)
        {
            var gammaS = gamma.DataView.AsReadOnlySpan();
            var invStdS = invStd.DataView.AsReadOnlySpan();
            var meanS = mean.DataView.AsReadOnlySpan();
            var outGradS = output.GradView.AsReadOnlySpan();
            var inS = input.DataView.AsReadOnlySpan();

            using var sumDyBuf = new PooledBuffer<float>(C);
            using var sumDyXBuf = new PooledBuffer<float>(C);
            using var xhatBuf = new PooledBuffer<float>(hw, false);
            var sumDy = sumDyBuf.Span;
            var sumDyX = sumDyXBuf.Span;
            var xhat = xhatBuf.Span;

            for (var n = 0; n < N; n++)
            {
                for (var c = 0; c < C; c++)
                {
                    var gB = outGradS.Slice((n * C + c) * hw, hw);
                    TensorPrimitives.Subtract(inS.Slice((n * C + c) * hw, hw), meanS[c], xhat);
                    TensorPrimitives.Multiply(xhat, invStdS[c], xhat);
                    sumDy[c] += TensorPrimitives.Sum(gB);
                    sumDyX[c] += TensorPrimitives.Dot(gB, xhat);
                }
            }

            if (beta.RequiresGrad)
            {
                var betaGrad = beta.GradView.AsSpan();
                for (var c = 0; c < C; c++)
                {
                    betaGrad[c] += sumDy[c];
                }
            }
            if (gamma.RequiresGrad)
            {
                var gammaGrad = gamma.GradView.AsSpan();
                for (var c = 0; c < C; c++)
                {
                    gammaGrad[c] += sumDyX[c];
                }
            }

            if (input.RequiresGrad)
            {
                var iGS = input.GradView.AsSpan();
                using var termBuf = new PooledBuffer<float>(hw, false);
                var term = termBuf.Span;

                for (var n = 0; n < N; n++)
                {
                    for (var c = 0; c < C; c++)
                    {
                        var gB = outGradS.Slice((n * C + c) * hw, hw);
                        var iGB = iGS.Slice((n * C + c) * hw, hw);
                        TensorPrimitives.Subtract(inS.Slice((n * C + c) * hw, hw), meanS[c], xhat);
                        TensorPrimitives.Multiply(xhat, invStdS[c], xhat);
                        TensorPrimitives.Multiply(gB, (float)m, term);
                        TensorPrimitives.Subtract(term, sumDy[c], term);
                        TensorPrimitives.Multiply(xhat, sumDyX[c], xhat);
                        TensorPrimitives.Subtract(term, xhat, term);
                        TensorPrimitives.MultiplyAdd(term, gammaS[c] * invStdS[c] / m, iGB, iGB);
                    }
                }
            }
        }

        private static void BatchNorm2DBackwardParallel(
            AutogradNode input, AutogradNode output, AutogradNode gamma, AutogradNode beta,
            AutogradNode mean, AutogradNode invStd, int N, int C, int hw, int m)
        {
            // Closure captures: AutogradNode references (classes) — Spans rebuilt inside the lambda
            // (cannot cross a delegate boundary). Per-thread xhat/term buffers rented via PooledBuffer
            // (using-scoped, ArrayPool<T>.Shared-backed — fastest per 2026-05-29 PoolComparisonBenchmark).
            // Rented (>= C) — every slot [0, C) is ASSIGNED by the first parallel pass before any read,
            // so no Clear() is needed and the excess tail is never touched.
            var sumDy = PooledBuffer<float>.RentArray(C);
            var sumDyX = PooledBuffer<float>.RentArray(C);
            try
            {
                var inputNode = input;
                var outputNode = output;
                var meanNode = mean;
                var invStdNode = invStd;

                OverfitParallel.For(0, C, c =>
                {
                    var inLocal = inputNode.DataView.AsReadOnlySpan();
                    var ogLocal = outputNode.GradView.AsReadOnlySpan();
                    var meanLocal = meanNode.DataView.AsReadOnlySpan();
                    var invStdLocal = invStdNode.DataView.AsReadOnlySpan();

                    using var xhatBuf = new PooledBuffer<float>(hw, clearMemory: false);
                    var xhatLocal = xhatBuf.Span;
                    var meanC = meanLocal[c];
                    var invStdC = invStdLocal[c];
                    var sDy = 0f;
                    var sDyX = 0f;
                    for (var n = 0; n < N; n++)
                    {
                        var off = (n * C + c) * hw;
                        var gB = ogLocal.Slice(off, hw);
                        TensorPrimitives.Subtract(inLocal.Slice(off, hw), meanC, xhatLocal);
                        TensorPrimitives.Multiply(xhatLocal, invStdC, xhatLocal);
                        sDy += TensorPrimitives.Sum(gB);
                        sDyX += TensorPrimitives.Dot(gB, xhatLocal);
                    }

                    sumDy[c] = sDy;
                    sumDyX[c] = sDyX;
                });

                if (beta.RequiresGrad)
                {
                    var betaGrad = beta.GradView.AsSpan();
                    for (var c = 0; c < C; c++)
                    {
                        betaGrad[c] += sumDy[c];
                    }
                }
                if (gamma.RequiresGrad)
                {
                    var gammaGrad = gamma.GradView.AsSpan();
                    for (var c = 0; c < C; c++)
                    {
                        gammaGrad[c] += sumDyX[c];
                    }
                }

                if (input.RequiresGrad)
                {
                    var gammaNode = gamma;
                    OverfitParallel.For(0, C, c =>
                    {
                        var inLocal = inputNode.DataView.AsReadOnlySpan();
                        var ogLocal = outputNode.GradView.AsReadOnlySpan();
                        var iGLocal = inputNode.GradView.AsSpan();
                        var meanLocal = meanNode.DataView.AsReadOnlySpan();
                        var invStdLocal = invStdNode.DataView.AsReadOnlySpan();
                        var gammaLocal = gammaNode.DataView.AsReadOnlySpan();

                        using var xhatBuf = new PooledBuffer<float>(hw, clearMemory: false);
                        using var termBuf = new PooledBuffer<float>(hw, clearMemory: false);
                        var xhatLocal = xhatBuf.Span;
                        var termLocal = termBuf.Span;
                        var meanC = meanLocal[c];
                        var invStdC = invStdLocal[c];
                        var coeff = gammaLocal[c] * invStdC / m;
                        var sumDyC = sumDy[c];
                        var sumDyXC = sumDyX[c];

                        for (var n = 0; n < N; n++)
                        {
                            var off = (n * C + c) * hw;
                            var gB = ogLocal.Slice(off, hw);
                            var iGB = iGLocal.Slice(off, hw);
                            TensorPrimitives.Subtract(inLocal.Slice(off, hw), meanC, xhatLocal);
                            TensorPrimitives.Multiply(xhatLocal, invStdC, xhatLocal);
                            TensorPrimitives.Multiply(gB, (float)m, termLocal);
                            TensorPrimitives.Subtract(termLocal, sumDyC, termLocal);
                            TensorPrimitives.Multiply(xhatLocal, sumDyXC, xhatLocal);
                            TensorPrimitives.Subtract(termLocal, xhatLocal, termLocal);
                            TensorPrimitives.MultiplyAdd(termLocal, coeff, iGB, iGB);
                        }
                    });
                }
            }
            finally
            {
                PooledBuffer<float>.ReturnArray(sumDyX);
                PooledBuffer<float>.ReturnArray(sumDy);
            }
        }
    }
}