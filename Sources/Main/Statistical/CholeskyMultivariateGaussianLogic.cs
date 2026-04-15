// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    public static class CholeskyMultivariateGaussianLogic
    {
        private const int StackAllocThreshold = 256;

        public static double GetLogProbability(
            ReadOnlySpan<float> observation,
            ReadOnlySpan<float> mean,
            TensorView<float> L,
            double logNormConst)
        {
            return LogProbabilityDensity(observation, mean, L, logNormConst);
        }

        public static double ProbabilityDensity(
            ReadOnlySpan<float> observation,
            ReadOnlySpan<float> mean,
            TensorView<float> L,
            double logNormConst)
        {
            return Math.Exp(LogProbabilityDensity(observation, mean, L, logNormConst));
        }

        public static double LogProbabilityDensity(
            ReadOnlySpan<float> observation,
            ReadOnlySpan<float> mean,
            TensorView<float> L,
            double logNormConst)
        {
            var dimensions = observation.Length;
            var mahalanobisDistanceSq = 0.0;

            if (dimensions <= StackAllocThreshold)
            {
                Span<float> diff = stackalloc float[dimensions];
                Span<float> y = stackalloc float[dimensions];
                mahalanobisDistanceSq = SolveAndGetDistance(observation, mean, L, dimensions, diff, y);
            }
            else
            {
                using var diffBuf = new PooledBuffer<float>(dimensions);
                using var yBuf = new PooledBuffer<float>(dimensions);
                mahalanobisDistanceSq = SolveAndGetDistance(observation, mean, L, dimensions, diffBuf.Span, yBuf.Span);
            }

            return logNormConst - 0.5 * mahalanobisDistanceSq;
        }

        private static double SolveAndGetDistance(
            ReadOnlySpan<float> observation,
            ReadOnlySpan<float> mean,
            TensorView<float> L,
            int dimensions,
            Span<float> diff,
            Span<float> y)
        {
            TensorPrimitives.Subtract(observation, mean, diff);

            var lSpan = L.AsReadOnlySpan();
            for (var i = 0; i < dimensions; i++)
            {
                var sum = diff[i];
                var lRow = lSpan.Slice(i * dimensions, dimensions);

                for (var j = 0; j < i; j++)
                {
                    sum -= lRow[j] * y[j];
                }

                y[i] = sum / lRow[i];
            }

            var mahalanobisDistanceSq = 0.0;
            for (var i = 0; i < dimensions; i++)
            {
                mahalanobisDistanceSq += y[i] * y[i];
            }

            return mahalanobisDistanceSq;
        }

        public static void ValidateInputs(ReadOnlySpan<float> mean, TensorView<float> covariance)
        {
            var dimensions = mean.Length;

            if (covariance.GetDim(0) != dimensions || covariance.GetDim(1) != dimensions)
            {
                throw new ArgumentException($"Covariance matrix must be {dimensions}x{dimensions}.", nameof(covariance));
            }
        }

        public static FastTensor<float> DecomposeCholesky(TensorView<float> matrix)
        {
            var n = matrix.GetDim(0);
            var L = new FastTensor<float>(n, n, clearMemory: true);
            var lSpan = L.GetView().AsSpan();
            var mSpan = matrix.AsReadOnlySpan();

            for (var i = 0; i < n; i++)
            {
                var lRowI = lSpan.Slice(i * n, n);
                for (var j = 0; j <= i; j++)
                {
                    var lRowJ = lSpan.Slice(j * n, n);
                    var dot = TensorPrimitives.Dot(lRowI.Slice(0, j), lRowJ.Slice(0, j));

                    if (i == j)
                    {
                        var pivot = mSpan[i * n + i] - dot;

                        if (pivot <= 0.0)
                        {
                            L.Dispose();
                            throw new ArgumentException($"Covariance matrix is not positive-definite! Negative pivot [{i},{i}] = {pivot:G6}.", nameof(matrix));
                        }

                        lRowI[i] = MathF.Sqrt(pivot);
                    }
                    else
                    {
                        lRowI[j] = (mSpan[i * n + j] - dot) / lRowJ[j];
                    }
                }
            }

            return L;
        }

        public static double CalculateLogNormConstant(int dimensions, TensorView<float> L)
        {
            if (L.GetDim(0) != dimensions || L.GetDim(1) != dimensions)
            {
                throw new ArgumentException($"Matrix L must be {dimensions}x{dimensions}.", nameof(L));
            }

            var logNormConst = -0.5 * dimensions * Math.Log(2.0 * Math.PI);
            var lSpan = L.AsReadOnlySpan();

            for (var i = 0; i < dimensions; i++)
            {
                logNormConst -= Math.Log(lSpan[i * dimensions + i]);
            }

            return logNormConst;
        }
    }
}