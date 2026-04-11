// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    ///     Stateless, static class grouping the logic of a Multivariate Gaussian Distribution
    ///     using Cholesky decomposition.
    ///     Optimized for memory efficiency (stackalloc / ArrayPool) and SIMD acceleration.
    /// </summary>
    public static class CholeskyMultivariateGaussianLogic
    {
        private const int StackAllocThreshold = 256;

        #region 1. Probability Density Calculations

        public static double GetLogProbability(
            ReadOnlySpan<float> observation,
            ReadOnlySpan<float> mean,
            FastMatrix<float> L,
            double logNormConst)
        {
            return LogProbabilityDensity(observation, mean, L, logNormConst);
        }

        public static double ProbabilityDensity(
            ReadOnlySpan<float> observation,
            ReadOnlySpan<float> mean,
            FastMatrix<float> L,
            double logNormConst)
        {
            return Math.Exp(LogProbabilityDensity(observation, mean, L, logNormConst));
        }

        /// <summary>
        ///     Logarithm of the probability density of a Multivariate Gaussian using Cholesky matrix.
        ///     Zero-allocation path using stackalloc for small dimensions (n <= 256).
        ///     Complexity: O(n^2) through forward-solve; SIMD-accelerated via TensorPrimitives.
        /// </summary>
        public static double LogProbabilityDensity(
            ReadOnlySpan<float> observation,
            ReadOnlySpan<float> mean,
            FastMatrix<float> L,
            double logNormConst)
        {
            var n = mean.Length;

            if (observation.Length != n)
                throw new ArgumentException($"Observation must have length {n}.", nameof(observation));

            if (L.Rows != n || L.Cols != n)
                throw new ArgumentException($"Matrix L must be {n}x{n}.", nameof(L));

            float[] diffArr = null;
            float[] zArr = null;

            try
            {
                // Magia Zero-Allocation: Dla metryk K8s (np. n=8), to bierze pamięć bezpośrednio z cache L1 procesora
                var diff = n <= StackAllocThreshold ? stackalloc float[n] : (diffArr = ArrayPool<float>.Shared.Rent(n)).AsSpan(0, n);
                var z = n <= StackAllocThreshold ? stackalloc float[n] : (zArr = ArrayPool<float>.Shared.Rent(n)).AsSpan(0, n);

                TensorPrimitives.Subtract(observation, mean, diff);

                // Forward-solve: L * z = diff 
                for (var i = 0; i < n; i++)
                {
                    // SIMD dot product: L[i, 0..i) * z[0..i)
                    var dot = TensorPrimitives.Dot(L.ReadOnlyRow(i)[..i], z[..i]);
                    z[i] = (diff[i] - dot) / L[i, i];
                }

                // SIMD sum of squares (Mahalanobis distance squared)
                var mahalSq = TensorPrimitives.Dot(z, z);

                return logNormConst - 0.5 * mahalSq;
            }
            finally
            {
                if (diffArr != null) ArrayPool<float>.Shared.Return(diffArr);
                if (zArr != null) ArrayPool<float>.Shared.Return(zArr);
            }
        }

        #endregion

        #region 2. Linear Algebra and Initialization

        public static void ValidateInputs(ReadOnlySpan<float> mean, FastMatrix<float> covariance)
        {
            var n = mean.Length;

            if (n == 0) throw new ArgumentException("Mean vector cannot be empty.", nameof(mean));

            for (var i = 0; i < n; i++)
            {
                if (!float.IsFinite(mean[i]))
                    throw new ArgumentException($"mean[{i}] = {mean[i]} is not finite.", nameof(mean));
            }

            if (covariance.Rows != n || covariance.Cols != n)
            {
                throw new ArgumentException($"Covariance matrix must be {n}x{n}, got {covariance.Rows}x{covariance.Cols}.", nameof(covariance));
            }
        }

        /// <summary>
        ///     Cholesky Decomposition: returns a lower triangular matrix L such that A = L * L^T.
        ///     The caller takes ownership of the returned FastMatrix and is responsible for calling Dispose().
        /// </summary>
        public static FastMatrix<float> DecomposeCholesky(FastMatrix<float> matrix)
        {
            if (matrix.Rows != matrix.Cols)
                throw new ArgumentException("Covariance matrix must be square.", nameof(matrix));

            var n = matrix.Rows;
            var L = new FastMatrix<float>(n, n);

            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    var dot = TensorPrimitives.Dot(L.ReadOnlyRow(i)[..j], L.ReadOnlyRow(j)[..j]);

                    if (i == j)
                    {
                        var pivot = matrix[i, i] - dot;

                        if (pivot <= 0.0)
                        {
                            L.Dispose();
                            throw new ArgumentException($"Covariance matrix is not positive-definite! Negative pivot [{i},{i}] = {pivot:G6}.", nameof(matrix));
                        }

                        L[i, i] = MathF.Sqrt(pivot);
                    }
                    else
                    {
                        L[i, j] = (matrix[i, j] - dot) / L[j, j];
                    }
                }
            }

            return L;
        }

        public static double CalculateLogNormConstant(int dimensions, FastMatrix<float> L)
        {
            if (L.Rows != dimensions || L.Cols != dimensions)
                throw new ArgumentException($"Matrix L must be {dimensions}x{dimensions}.", nameof(L));

            var logNormConst = -0.5 * dimensions * Math.Log(2.0 * Math.PI);

            for (var i = 0; i < dimensions; i++)
            {
                logNormConst -= Math.Log(L[i, i]);
            }

            return logNormConst;
        }

        #endregion
    }
}