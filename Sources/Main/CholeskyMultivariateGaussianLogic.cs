using System.Numerics.Tensors;

namespace DevOnBike.Overfit
{
    /// <summary>
    /// Stateless, static class grouping the logic of a Multivariate Gaussian Distribution 
    /// using Cholesky decomposition.
    /// Optimized for memory efficiency using FastMatrix and FastBuffer (ArrayPool integration).
    /// </summary>
    public static class CholeskyMultivariateGaussianLogic
    {
        #region 1. Probability Density Calculations

        /// <summary>
        /// Logarithm of the probability density. Wrapper with a readable argument order.
        /// </summary>
        public static double GetLogProbability(
            ReadOnlySpan<double> observation,
            ReadOnlySpan<double> mean,
            FastMatrix<double> L,
            double logNormConst)
        {
            return LogProbabilityDensity(observation, mean, L, logNormConst);
        }

        /// <summary>
        /// Probability density (linear space).
        /// </summary>
        public static double ProbabilityDensity(
            ReadOnlySpan<double> observation,
            ReadOnlySpan<double> mean,
            FastMatrix<double> L,
            double logNormConst)
        {
            return Math.Exp(LogProbabilityDensity(observation, mean, L, logNormConst));
        }

        /// <summary>
        /// Logarithm of the probability density of a Multivariate Gaussian using Cholesky matrix.
        /// Elegantly uses FastBuffer to avoid heavy array allocations, relying on ArrayPool.
        /// Complexity: O(n^2) through forward-solve; diff and mahalSq are SIMD-accelerated via TensorPrimitives.
        /// </summary>
        public static double LogProbabilityDensity(
            ReadOnlySpan<double> observation,
            ReadOnlySpan<double> mean,
            FastMatrix<double> L,
            double logNormConst)
        {
            var n = mean.Length;

            if (observation.Length != n)
            {
                throw new ArgumentException($"Observation must have length {n}.", nameof(observation));
            }

            if (L.Rows != n || L.Cols != n)
            {
                throw new ArgumentException($"Matrix L must be {n}x{n}.", nameof(L));
            }

            // Clean code perfection: 'using' takes care of returning the array to the pool automatically.
            using var diffBuffer = new FastBuffer<double>(n);
            var diff = diffBuffer.AsSpan();

            TensorPrimitives.Subtract(observation, mean, diff);

            // Forward-solve: L * z = diff 
            using var zBuffer = new FastBuffer<double>(n);
            var z = zBuffer.AsSpan();

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

        #endregion

        #region 2. Linear Algebra and Initialization

        /// <summary>
        /// Validates the mean vector and covariance matrix dimensions.
        /// </summary>
        public static void ValidateInputs(ReadOnlySpan<double> mean, FastMatrix<double> covariance)
        {
            var n = mean.Length;

            if (n == 0)
            {
                throw new ArgumentException("Mean vector cannot be empty.", nameof(mean));
            }
            
            for (var i = 0; i < n; i++)
            {
                if (!double.IsFinite(mean[i]))
                {
                    throw new ArgumentException($"mean[{i}] = {mean[i]} is not finite.", nameof(mean));
                }
            }

            if (covariance.Rows != n || covariance.Cols != n)
            {
                throw new ArgumentException($"Covariance matrix must be {n}x{n}, got {covariance.Rows}x{covariance.Cols}.", nameof(covariance));
            }
        }

        /// <summary>
        /// Cholesky Decomposition: returns a lower triangular matrix L such that A = L * L^T.
        /// Uses TensorPrimitives.Dot for the inner dot product — SIMD-accelerated.
        /// The caller takes ownership of the returned FastMatrix and is responsible for calling Dispose().
        /// </summary>
        public static FastMatrix<double> DecomposeCholesky(FastMatrix<double> matrix)
        {
            if (matrix.Rows != matrix.Cols)
            {
                throw new ArgumentException("Covariance matrix must be square.", nameof(matrix));
            }

            var n = matrix.Rows;
            var L = new FastMatrix<double>(n, n);

            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j <= i; j++)
                {
                    // SIMD dot product L[i, 0..j) * L[j, 0..j) — replaces the inner scalar loop for k
                    var dot = TensorPrimitives.Dot(L.ReadOnlyRow(i)[..j], L.ReadOnlyRow(j)[..j]);

                    if (i == j)
                    {
                        var pivot = matrix[i, i] - dot;

                        if (pivot <= 0.0)
                        {
                            L.Dispose();
                            throw new ArgumentException($"Covariance matrix is not positive-definite! Negative pivot [{i},{i}] = {pivot:G6}.", nameof(matrix));
                        }

                        L[i, i] = Math.Sqrt(pivot);
                    }
                    else
                    {
                        L[i, j] = (matrix[i, j] - dot) / L[j, j];
                    }
                }
            }

            return L;
        }

        /// <summary>
        /// Calculates the normalizing constant for the log-density: -0.5 * n * ln(2π) - Σ ln(L_ii).
        /// </summary>
        public static double CalculateLogNormConstant(int dimensions, FastMatrix<double> L)
        {
            if (L.Rows != dimensions || L.Cols != dimensions)
            {
                throw new ArgumentException($"Matrix L must be {dimensions}x{dimensions}.", nameof(L));
            }

            var logNormConst = -0.5 * dimensions * Math.Log(2.0 * Math.PI);

            // The diagonal is not contiguous in memory, so a scalar loop is correct and fast enough here
            for (var i = 0; i < dimensions; i++)
            {
                logNormConst -= Math.Log(L[i, i]);
            }

            return logNormConst;
        }

        #endregion
    }
}