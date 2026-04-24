// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
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

            // KRYTYCZNA POPRAWKA: 'mean' MUSI być czyszczone, ponieważ za chwilę akumulujemy (Add) do niego wartości!
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
                graph?.Record(OpCode.BatchNorm1D, output, input, null, 0, 0, 0, 0, 0, [gamma, beta, mean, invStd]);
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
    }
}