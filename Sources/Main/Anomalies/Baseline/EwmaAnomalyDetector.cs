// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Baseline
{
    /// <summary>
    /// Classical statistical anomaly baseline — an exponentially-weighted moving average
    /// (EWMA) z-score detector — used to benchmark <see cref="GptAnomalyDetector"/> with
    /// rigor ("the transformer beats a tuned classical baseline by X", or it doesn't).
    /// See <c>docs/gp-anomaly-baseline.md</c>: the cheap O(n) floor to clear before any GP.
    ///
    /// Per metric it tracks an EWMA mean μ and EWMA variance σ² (RiskMetrics form). A
    /// snapshot's surprise is the mean over metrics of the Gaussian neg-log-density's
    /// data term <c>½·z²</c> with <c>z = (x−μ)/σ</c> — the direct continuous analogue of
    /// the GPT detector's mean per-token negative log-probability, so the two are
    /// comparable on separation and worst-metric attribution. Returns the same
    /// <see cref="AnomalyScore"/> shape (incl. <see cref="AnomalyScore.WorstMetric"/> named
    /// via <see cref="MetricTokenizer.MetricNameOf"/>) so a single harness can score both.
    ///
    /// Pure stats — no model, zero allocations per score. Not thread-safe; one per pod.
    /// </summary>
    public sealed class EwmaAnomalyDetector
    {
        private const int FeatureCount = MetricSnapshot.FeatureCount;   // 12

        private readonly int _warmup;
        private readonly float _decay;
        private readonly float _zCap;
        private readonly float[] _mean = new float[FeatureCount];
        private readonly float[] _var = new float[FeatureCount];

        /// <summary>
        /// Finite observations folded into each feature. Zero means the feature has no baseline yet — either
        /// because the detector is new, or because that metric has never once arrived.
        /// </summary>
        private readonly int[] _observed = new int[FeatureCount];
        private int _seen;

        /// <param name="warmupSnapshots">Samples used to seed the EWMA before scoring (default 21, matching the GPT detector's window).</param>
        /// <param name="decay">EWMA decay α in (0,1]; larger = faster adaptation, shorter memory (default 0.2 ≈ 10-sample memory).</param>
        /// <param name="zCap">Per-metric |z| clamp so one saturated metric can't produce an infinite score (default 8).</param>
        public EwmaAnomalyDetector(int warmupSnapshots = 21, float decay = 0.2f, float zCap = 8f)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(warmupSnapshots);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(decay);
            if (decay > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(decay), "Decay must be in (0, 1].");
            }
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(zCap);

            _warmup = warmupSnapshots;
            _decay = decay;
            _zCap = zCap;
        }

        /// <summary>True once enough samples have seeded the EWMA — scores are reliable.</summary>
        public bool WindowFilled => _seen >= _warmup;

        /// <summary>Feeds a snapshot and returns its anomaly score (IsWarmup until the EWMA is seeded).</summary>
        public AnomalyScore Score(MetricSnapshot snapshot)
        {
            Span<float> x = stackalloc float[FeatureCount];
            snapshot.WriteFeatureVector(x);

            if (_seen < _warmup)
            {
                Observe(x);
                _seen++;
                return Warmup(snapshot);
            }

            var total = 0f;
            var scored = 0;
            var worst = -1;
            var worstAbsZ = -1f;
            float worstExpected = 0f, worstActual = 0f;

            for (var m = 0; m < FeatureCount; m++)
            {
                // A feature that has never been observed has no baseline to deviate from, and one that is
                // missing right now has nothing to compare. Either way it contributes no evidence — scoring
                // it against a zero mean would invent a deviation out of an absence.
                if (_observed[m] == 0 || !float.IsFinite(x[m]))
                {
                    continue;
                }

                scored++;
                var sigma = MathF.Sqrt(_var[m]);
                // Scale-relative floor: keeps a near-constant metric (e.g. oom_events_rate)
                // from yielding an infinite z, without flattening a genuinely quiet metric.
                var floor = 1e-3f * MathF.Abs(_mean[m]) + 1e-9f;
                var sigmaEff = MathF.Max(sigma, floor);

                var absZ = MathF.Abs((x[m] - _mean[m]) / sigmaEff);
                if (absZ > _zCap)
                {
                    absZ = _zCap;
                }

                total += 0.5f * absZ * absZ;
                if (absZ > worstAbsZ)
                {
                    worstAbsZ = absZ;
                    worst = m;
                    worstExpected = _mean[m];
                    worstActual = x[m];
                }
            }

            // Update AFTER scoring so a snapshot is never compared against itself.
            Observe(x);
            _seen++;

            // Every feature was missing or unseeded: there is nothing to score, and reporting 0 would say
            // "healthy" about a snapshot we could not read at all.
            if (scored == 0)
            {
                return Warmup(snapshot);
            }

            return new AnomalyScore
            {
                IsWarmup = false,

                // Averaged over the features that actually contributed, not over all twelve — dividing by
                // the full count would dilute a real deviation in proportion to how much data is missing,
                // making a half-blind detector look calmer than a fully-sighted one.
                Score = total / scored,
                PodName = snapshot.PodName,
                Timestamp = snapshot.Timestamp,
                WorstMetric = MetricTokenizer.MetricNameOf(worst * MetricTokenizer.BinsPerMetric),
                ExpectedValue = worstExpected,
                ActualValue = worstActual,
            };
        }

        /// <summary>Clears the EWMA state. Call after a pod restart.</summary>
        public void Reset()
        {
            Array.Clear(_mean);
            Array.Clear(_var);
            Array.Clear(_observed);
            _seen = 0;
        }

        /// <summary>
        /// Folds one snapshot into the baseline, per feature. A feature is seeded by its <b>first finite
        /// value</b> rather than by the first snapshot: a metric that only starts reporting on the tenth
        /// scrape would otherwise be seeded from a NaN and stay poisoned forever, because every arithmetic
        /// operation on NaN yields NaN and nothing downstream ever recovers.
        /// </summary>
        private void Observe(ReadOnlySpan<float> x)
        {
            for (var m = 0; m < FeatureCount; m++)
            {
                if (!float.IsFinite(x[m]))
                {
                    continue;
                }

                if (_observed[m] == 0)
                {
                    _mean[m] = x[m];
                    _var[m] = 0f;
                    _observed[m] = 1;
                    continue;
                }

                var diff = x[m] - _mean[m];
                _mean[m] += _decay * diff;
                // RiskMetrics EWMA variance: σ²ₜ = (1−α)(σ²ₜ₋₁ + α·diff²), diff vs the prior mean.
                _var[m] = (1f - _decay) * (_var[m] + _decay * diff * diff);
                _observed[m]++;
            }
        }

        private AnomalyScore Warmup(MetricSnapshot snapshot) => new()
        {
            IsWarmup = true,
            Score = 0f,
            PodName = snapshot.PodName,
            Timestamp = snapshot.Timestamp,
        };
    }
}
