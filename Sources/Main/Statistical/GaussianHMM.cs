// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// Multivariate Gaussian Hidden Markov Model (HMM) with Full Covariance.
    /// Integrates Cholesky decomposition for extreme SIMD inference speed.
    /// Operates entirely in Log-Space to prevent numerical underflow.
    /// </summary>
    public sealed class GaussianHMM : IDisposable
    {
        public int StateCount { get; }
        public int FeatureCount { get; }

        // Model Parameters (Log-Space)
        private readonly FastTensor<float> _logPi;          // [StateCount]
        private readonly FastTensor<float> _logA;           // [StateCount, StateCount]

        // Emission Parameters
        private readonly FastTensor<float> _means;          // [StateCount, FeatureCount]

        // Full Covariance (Cholesky Decomposed L matrices)
        private readonly FastMatrix<float>[] _choleskyL;    // Array of size [StateCount]
        private readonly double[] _logNormConstants;        // Array of size [StateCount]

        private bool _disposed;
        private const float LogZero = float.NegativeInfinity;

        public GaussianHMM(int stateCount, int featureCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stateCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(featureCount);

            StateCount = stateCount;
            FeatureCount = featureCount;

            _logPi = new FastTensor<float>(stateCount);
            _logA = new FastTensor<float>(stateCount, stateCount);
            _means = new FastTensor<float>(stateCount, featureCount);

            _choleskyL = new FastMatrix<float>[stateCount];
            _logNormConstants = new double[stateCount];
        }

        /// <summary>
        /// Loads model parameters and automatically decomposes full covariance matrices using Cholesky.
        /// Probabilities must be provided in normal space (0.0 to 1.0).
        /// </summary>
        public void SetModel(
            ReadOnlySpan<float> pi,
            ReadOnlySpan<float> transitionMatrix,
            ReadOnlySpan<float> means,
            FastMatrix<float>[] fullCovariances)
        {
            if (fullCovariances.Length != StateCount)
                throw new ArgumentException("You must provide one covariance matrix per state.");

            for (int i = 0; i < StateCount; i++)
            {
                _logPi[i] = pi[i] > 0 ? MathF.Log(pi[i]) : LogZero;

                for (int j = 0; j < StateCount; j++)
                {
                    float a = transitionMatrix[i * StateCount + j];
                    _logA[i, j] = a > 0 ? MathF.Log(a) : LogZero;
                }
            }

            means.CopyTo(_means.AsSpan());

            for (int i = 0; i < StateCount; i++)
            {
                // Dispose old matrix if we are overwriting an existing model
                _choleskyL[i]?.Dispose();

                var covMatrix = fullCovariances[i];
                var meanVector = _means.AsReadOnlySpan().Slice(i * FeatureCount, FeatureCount);

                CholeskyMultivariateGaussianLogic.ValidateInputs(meanVector, covMatrix);

                // Pre-decompose to L matrix and calculate PDF constants
                _choleskyL[i] = CholeskyMultivariateGaussianLogic.DecomposeCholesky(covMatrix);
                _logNormConstants[i] = CholeskyMultivariateGaussianLogic.CalculateLogNormConstant(FeatureCount, _choleskyL[i]);
            }
        }

        /// <summary>
        /// Highly optimized, SIMD-accelerated emission calculation using your custom logic.
        /// </summary>
        private float ComputeLogEmission(int state, ReadOnlySpan<float> x)
        {
            var mean = _means.AsReadOnlySpan().Slice(state * FeatureCount, FeatureCount);
            var L = _choleskyL[state];
            var logNormConst = _logNormConstants[state];

            return (float)CholeskyMultivariateGaussianLogic.LogProbabilityDensity(x, mean, L, logNormConst);
        }

        // -------------------------------------------------------------------------
        // 1. VITERBI ALGORITHM (Decoding hidden states)
        // -------------------------------------------------------------------------

        public void DecodeViterbi(int timeSteps, ReadOnlySpan<float> sequence, Span<int> outputStates)
        {
            if (outputStates.Length < timeSteps)
                throw new ArgumentException("Output buffer too small.");

            int N = StateCount;
            int vLen = timeSteps * N;

            var vArr = ArrayPool<float>.Shared.Rent(vLen);
            var ptrArr = ArrayPool<int>.Shared.Rent(vLen);

            try
            {
                var V = vArr.AsSpan(0, vLen);
                var ptr = ptrArr.AsSpan(0, vLen);

                var x0 = sequence.Slice(0, FeatureCount);
                for (int i = 0; i < N; i++)
                {
                    V[i] = _logPi[i] + ComputeLogEmission(i, x0);
                    ptr[i] = 0;
                }

                var logA = _logA.AsReadOnlySpan();
                for (int t = 1; t < timeSteps; t++)
                {
                    var xt = sequence.Slice(t * FeatureCount, FeatureCount);

                    for (int j = 0; j < N; j++)
                    {
                        float maxLogProb = LogZero;
                        int bestPrevState = 0;

                        float emission = ComputeLogEmission(j, xt);

                        for (int i = 0; i < N; i++)
                        {
                            float prob = V[(t - 1) * N + i] + logA[i * N + j];
                            if (prob > maxLogProb)
                            {
                                maxLogProb = prob;
                                bestPrevState = i;
                            }
                        }

                        V[t * N + j] = maxLogProb + emission;
                        ptr[t * N + j] = bestPrevState;
                    }
                }

                float bestFinalProb = LogZero;
                int bestFinalState = 0;
                for (int i = 0; i < N; i++)
                {
                    if (V[(timeSteps - 1) * N + i] > bestFinalProb)
                    {
                        bestFinalProb = V[(timeSteps - 1) * N + i];
                        bestFinalState = i;
                    }
                }

                outputStates[timeSteps - 1] = bestFinalState;
                for (int t = timeSteps - 1; t > 0; t--)
                {
                    outputStates[t - 1] = ptr[t * N + outputStates[t]];
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(vArr);
                ArrayPool<int>.Shared.Return(ptrArr);
            }
        }

        // -------------------------------------------------------------------------
        // 2. FORWARD ALGORITHM (Anomaly Detection / Likelihood scoring)
        // -------------------------------------------------------------------------

        public float ScoreSequence(int timeSteps, ReadOnlySpan<float> sequence)
        {
            int N = StateCount;
            var alphaArr = ArrayPool<float>.Shared.Rent(N);
            var nextAlphaArr = ArrayPool<float>.Shared.Rent(N);

            try
            {
                var alpha = alphaArr.AsSpan(0, N);
                var nextAlpha = nextAlphaArr.AsSpan(0, N);

                var x0 = sequence.Slice(0, FeatureCount);
                for (int i = 0; i < N; i++)
                {
                    alpha[i] = _logPi[i] + ComputeLogEmission(i, x0);
                }

                var logA = _logA.AsReadOnlySpan();
                for (int t = 1; t < timeSteps; t++)
                {
                    var xt = sequence.Slice(t * FeatureCount, FeatureCount);

                    for (int j = 0; j < N; j++)
                    {
                        float emission = ComputeLogEmission(j, xt);
                        float sumLogExp = LogZero;

                        for (int i = 0; i < N; i++)
                        {
                            float p = alpha[i] + logA[i * N + j];
                            sumLogExp = LogSumExp(sumLogExp, p);
                        }

                        nextAlpha[j] = sumLogExp + emission;
                    }

                    nextAlpha.CopyTo(alpha);
                }

                float totalLogLikelihood = LogZero;
                for (int i = 0; i < N; i++)
                {
                    totalLogLikelihood = LogSumExp(totalLogLikelihood, alpha[i]);
                }

                return totalLogLikelihood;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(alphaArr);
                ArrayPool<float>.Shared.Return(nextAlphaArr);
            }
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static float LogSumExp(float a, float b)
        {
            if (float.IsNegativeInfinity(a)) return b;
            if (float.IsNegativeInfinity(b)) return a;

            float max = MathF.Max(a, b);
            return max + MathF.Log(1f + MathF.Exp(-MathF.Abs(a - b)));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _logPi?.Dispose();
            _logA?.Dispose();
            _means?.Dispose();

            if (_choleskyL != null)
            {
                for (int i = 0; i < _choleskyL.Length; i++)
                {
                    _choleskyL[i]?.Dispose();
                }
            }
        }
    }
}