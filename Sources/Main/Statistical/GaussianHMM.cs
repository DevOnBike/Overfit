// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Buffers;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// Generic Gaussian Hidden Markov Model (HMM) with Diagonal Covariance.
    /// Operates entirely in Log-Space to prevent numerical underflow.
    /// Zero-allocation inference paths using ArrayPool.
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
        private readonly FastTensor<float> _variances;      // [StateCount, FeatureCount]

        // Precalculated constant for Gaussian PDF: -0.5 * (F * log(2pi) + sum(log(var)))
        private readonly FastTensor<float> _pdfConstants;   // [StateCount]

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
            _variances = new FastTensor<float>(stateCount, featureCount);
            _pdfConstants = new FastTensor<float>(stateCount);
        }

        /// <summary>
        /// Loads model parameters and precalculates Gaussian constants.
        /// Probabilities must be provided in normal space (0.0 to 1.0), they are converted to Log-Space internally.
        /// </summary>
        public void SetModel(
            ReadOnlySpan<float> pi,
            ReadOnlySpan<float> transitionMatrix,
            ReadOnlySpan<float> means,
            ReadOnlySpan<float> variances)
        {
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
            variances.CopyTo(_variances.AsSpan());

            PrecalculateConstants();
        }

        private void PrecalculateConstants()
        {
            float log2Pi = MathF.Log(2f * MathF.PI);
            var varSpan = _variances.AsReadOnlySpan();

            for (int i = 0; i < StateCount; i++)
            {
                float sumLogVar = 0f;
                for (int f = 0; f < FeatureCount; f++)
                {
                    // Adding small epsilon to variance to prevent Log(0)
                    float v = MathF.Max(varSpan[i * FeatureCount + f], 1e-6f);
                    sumLogVar += MathF.Log(v);
                }

                _pdfConstants[i] = -0.5f * (FeatureCount * log2Pi + sumLogVar);
            }
        }

        /// <summary>
        /// Fast calculation of Log( P(x | State) ).
        /// </summary>
        private float ComputeLogEmission(int state, ReadOnlySpan<float> x)
        {
            var meanS = _means.AsReadOnlySpan().Slice(state * FeatureCount, FeatureCount);
            var varS = _variances.AsReadOnlySpan().Slice(state * FeatureCount, FeatureCount);

            float sumMahalanobis = 0f;

            for (int f = 0; f < FeatureCount; f++)
            {
                float diff = x[f] - meanS[f];
                // Math.Max prevents division by zero
                sumMahalanobis += (diff * diff) / MathF.Max(varS[f], 1e-6f);
            }

            return _pdfConstants[state] - 0.5f * sumMahalanobis;
        }

        // -------------------------------------------------------------------------
        // 1. VITERBI ALGORITHM (Decoding hidden states)
        // -------------------------------------------------------------------------

        /// <summary>
        /// Predicts the most likely sequence of hidden states.
        /// Zero allocation path.
        /// </summary>
        /// <param name="timeSteps">T - number of observations.</param>
        /// <param name="sequence">Flat array [T * FeatureCount].</param>
        /// <param name="outputStates">Buffer for output states [T].</param>
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

                // Initialization (t = 0)
                var x0 = sequence.Slice(0, FeatureCount);
                for (int i = 0; i < N; i++)
                {
                    V[i] = _logPi[i] + ComputeLogEmission(i, x0);
                    ptr[i] = 0;
                }

                // Recursion (t > 0)
                var logA = _logA.AsReadOnlySpan();
                for (int t = 1; t < timeSteps; t++)
                {
                    var xt = sequence.Slice(t * FeatureCount, FeatureCount);

                    for (int j = 0; j < N; j++) // current state
                    {
                        float maxLogProb = LogZero;
                        int bestPrevState = 0;

                        float emission = ComputeLogEmission(j, xt);

                        for (int i = 0; i < N; i++) // previous state
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

                // Termination
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

                // Backtracking
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

        /// <summary>
        /// Calculates the Log-Likelihood of the sequence.
        /// If this value drops significantly, the sequence is an Anomaly.
        /// Zero allocation path.
        /// </summary>
        public float ScoreSequence(int timeSteps, ReadOnlySpan<float> sequence)
        {
            int N = StateCount;
            var alphaArr = ArrayPool<float>.Shared.Rent(N);
            var nextAlphaArr = ArrayPool<float>.Shared.Rent(N);

            try
            {
                var alpha = alphaArr.AsSpan(0, N);
                var nextAlpha = nextAlphaArr.AsSpan(0, N);

                // Initialization (t = 0)
                var x0 = sequence.Slice(0, FeatureCount);
                for (int i = 0; i < N; i++)
                {
                    alpha[i] = _logPi[i] + ComputeLogEmission(i, x0);
                }

                // Recursion (t > 0)
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

                // Termination
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
            _variances?.Dispose();
            _pdfConstants?.Dispose();
        }
    }
}