// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Statistical
{
    public sealed class GaussianHMM : IDisposable
    {
        public int StateCount { get; }
        public int FeatureCount { get; }

        private readonly FastTensor<float> _logPi;
        private readonly FastTensor<float> _logA;
        private readonly FastTensor<float> _means;
        private readonly FastTensor<float>[] _choleskyL;
        private readonly double[] _logNormConstants;

        private bool _disposed;
        private const float LogZero = float.NegativeInfinity;

        public GaussianHMM(int stateCount, int featureCount)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stateCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(featureCount);

            StateCount = stateCount;
            FeatureCount = featureCount;

            _logPi = new FastTensor<float>(stateCount, clearMemory: false);
            _logA = new FastTensor<float>(stateCount, stateCount, clearMemory: false);
            _means = new FastTensor<float>(stateCount, featureCount, clearMemory: false);

            _choleskyL = new FastTensor<float>[stateCount];
            _logNormConstants = new double[stateCount];
        }

        public void SetModel(ReadOnlySpan<float> pi, ReadOnlySpan<float> transitionMatrix, ReadOnlySpan<float> means, FastTensor<float>[] fullCovariances)
        {
            if (fullCovariances.Length != StateCount)
            {
                throw new ArgumentException("You must provide one covariance matrix per state.");
            }

            for (var i = 0; i < StateCount; i++)
            {
                _logPi.GetView().AsSpan()[i] = pi[i] > 0 ? MathF.Log(pi[i]) : LogZero;

                for (var j = 0; j < StateCount; j++)
                {
                    var a = transitionMatrix[i * StateCount + j];
                    _logA.GetView().AsSpan()[i * StateCount + j] = a > 0 ? MathF.Log(a) : LogZero;
                }
            }

            means.CopyTo(_means.GetView().AsSpan());

            for (var i = 0; i < StateCount; i++)
            {
                _choleskyL[i]?.Dispose();

                var covMatrixView = fullCovariances[i].GetView();
                var meanVector = _means.GetView().AsReadOnlySpan().Slice(i * FeatureCount, FeatureCount);

                CholeskyMultivariateGaussianLogic.ValidateInputs(meanVector, covMatrixView);

                _choleskyL[i] = CholeskyMultivariateGaussianLogic.DecomposeCholesky(covMatrixView);
                _logNormConstants[i] = CholeskyMultivariateGaussianLogic.CalculateLogNormConstant(FeatureCount, _choleskyL[i].GetView());
            }
        }

        private float ComputeLogEmission(int state, ReadOnlySpan<float> x)
        {
            var mean = _means.GetView().AsReadOnlySpan().Slice(state * FeatureCount, FeatureCount);
            var L = _choleskyL[state].GetView();
            var logNormConst = _logNormConstants[state];

            return (float)CholeskyMultivariateGaussianLogic.LogProbabilityDensity(x, mean, L, logNormConst);
        }

        public void DecodeViterbi(int timeSteps, ReadOnlySpan<float> sequence, Span<int> outputStates)
        {
            if (outputStates.Length < timeSteps)
            {
                throw new ArgumentException("Output buffer too small.");
            }

            var N = StateCount;
            var vLen = timeSteps * N;

            using var vArr = new PooledBuffer<float>(vLen);
            using var ptrArr = new PooledBuffer<int>(vLen);

            var V = vArr.Span;
            var ptr = ptrArr.Span;

            var x0 = sequence.Slice(0, FeatureCount);
            for (var i = 0; i < N; i++)
            {
                V[i] = _logPi.GetView().AsReadOnlySpan()[i] + ComputeLogEmission(i, x0);
                ptr[i] = 0;
            }

            var logA = _logA.GetView().AsReadOnlySpan();
            for (var t = 1; t < timeSteps; t++)
            {
                var xt = sequence.Slice(t * FeatureCount, FeatureCount);

                for (var j = 0; j < N; j++)
                {
                    var maxLogProb = LogZero;
                    var bestPrevState = 0;
                    var emission = ComputeLogEmission(j, xt);

                    for (var i = 0; i < N; i++)
                    {
                        var prob = V[(t - 1) * N + i] + logA[i * N + j];
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

            var bestFinalProb = LogZero;
            var bestFinalState = 0;
            for (var i = 0; i < N; i++)
            {
                if (V[(timeSteps - 1) * N + i] > bestFinalProb)
                {
                    bestFinalProb = V[(timeSteps - 1) * N + i];
                    bestFinalState = i;
                }
            }

            outputStates[timeSteps - 1] = bestFinalState;
            for (var t = timeSteps - 1; t > 0; t--)
            {
                outputStates[t - 1] = ptr[t * N + outputStates[t]];
            }
        }

        public float ScoreSequence(int timeSteps, ReadOnlySpan<float> sequence)
        {
            var N = StateCount;

            using var alphaArr = new PooledBuffer<float>(N);
            using var nextAlphaArr = new PooledBuffer<float>(N);

            var alpha = alphaArr.Span;
            var nextAlpha = nextAlphaArr.Span;

            var x0 = sequence.Slice(0, FeatureCount);
            for (var i = 0; i < N; i++)
            {
                alpha[i] = _logPi.GetView().AsReadOnlySpan()[i] + ComputeLogEmission(i, x0);
            }

            var logA = _logA.GetView().AsReadOnlySpan();
            for (var t = 1; t < timeSteps; t++)
            {
                var xt = sequence.Slice(t * FeatureCount, FeatureCount);

                for (var j = 0; j < N; j++)
                {
                    var emission = ComputeLogEmission(j, xt);
                    var sumLogExp = LogZero;

                    for (var i = 0; i < N; i++)
                    {
                        var p = alpha[i] + logA[i * N + j];
                        sumLogExp = LogSumExp(sumLogExp, p);
                    }

                    nextAlpha[j] = sumLogExp + emission;
                }

                nextAlpha.CopyTo(alpha);
            }

            var totalLogLikelihood = LogZero;
            for (var i = 0; i < N; i++)
            {
                totalLogLikelihood = LogSumExp(totalLogLikelihood, alpha[i]);
            }

            return totalLogLikelihood;
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static float LogSumExp(float a, float b)
        {
            if (float.IsNegativeInfinity(a))
            {
                return b;
            }
            if (float.IsNegativeInfinity(b))
            {
                return a;
            }

            var max = MathF.Max(a, b);
            return max + MathF.Log(1f + MathF.Exp(-MathF.Abs(a - b)));
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            _logPi?.Dispose();
            _logA?.Dispose();
            _means?.Dispose();

            if (_choleskyL != null)
            {
                for (var i = 0; i < _choleskyL.Length; i++)
                {
                    _choleskyL[i]?.Dispose();
                }
            }
        }
    }
}