// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Anomalies.Gpt
{
    /// <summary>
    /// GPT-based anomaly detector for K8s pod metrics.
    ///
    /// Replaces the HMM-based K8sAnomalyDetector.
    ///
    /// How it works:
    ///   1. Maintains a rolling window of recent MetricSnapshots.
    ///   2. On each new snapshot, tokenizes the window → context tokens.
    ///   3. Asks GPT: "given this history, what is the probability of each next metric?"
    ///   4. Anomaly score = mean negative log-probability of actual metric tokens.
    ///      Low probability = model didn't expect this → anomaly.
    ///
    /// Score interpretation:
    ///   score ≈ 0.0 : predicted well → normal
    ///   score ≈ 2.0 : moderately unexpected
    ///   score ≈ 5.0+: highly unexpected → strong anomaly
    ///
    /// Thread safety: NOT thread-safe. One instance per pod.
    /// </summary>
    public sealed class GptAnomalyDetector : IDisposable
    {
        private readonly MetricTokenizer _tokenizer;
        private readonly SlmRuntimeHandle _handle;
        private readonly ISlmSession _session;
        private readonly int _contextSnapshots;
        private readonly Queue<MetricSnapshot> _window;
        private bool _disposed;

        /// <param name="handle">
        ///   Runtime handle from SlmRuntimeFactory.CreateGpt1(model, SlmRuntimeMode.Cached).
        ///   The detector takes ownership — dispose the detector, not the handle separately.
        /// </param>
        /// <param name="contextSnapshots">
        ///   Rolling window size in snapshots (default: 21 = ~5 min at 15s scrape).
        ///   Must not exceed model.ContextLength / MetricTokenizer.TokensPerSnapshot.
        /// </param>
        public GptAnomalyDetector(SlmRuntimeHandle handle, int contextSnapshots = 21)
        {
            ArgumentNullException.ThrowIfNull(handle);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextSnapshots);

            _handle = handle;
            _session = handle.Session;
            _contextSnapshots = contextSnapshots;
            _tokenizer = new MetricTokenizer();
            _window = new Queue<MetricSnapshot>(contextSnapshots + 1);
        }

        /// <summary>True once the window is full — scores are reliable.</summary>
        public bool WindowFilled => _window.Count >= _contextSnapshots;

        /// <summary>
        /// Feeds a new snapshot and returns an anomaly score.
        /// Returns IsWarmup=true until the window is filled.
        /// </summary>
        public AnomalyScore Score(MetricSnapshot snapshot)
        {
            if (_window.Count >= _contextSnapshots)
            {
                _window.Dequeue();
            }
            _window.Enqueue(snapshot);

            if (!WindowFilled)
            {
                return new AnomalyScore
                {
                    IsWarmup = true,
                    Score = 0f,
                    PodName = snapshot.PodName,
                    Timestamp = snapshot.Timestamp,
                };
            }

            // Tokenize full window as context
            var history = _window.ToArray();
            var contextCount = history.Length - 1;
            var contextTokens = new int[contextCount * MetricTokenizer.TokensPerSnapshot];
            for (var i = 0; i < contextCount; i++)
            {
                _tokenizer.EncodeSnapshot(history[i], contextTokens, i * MetricTokenizer.TokensPerSnapshot);
            }

            // Tokenize the new snapshot (target to predict)
            var actualTokens = new int[MetricTokenizer.TokensPerSnapshot];
            _tokenizer.EncodeSnapshot(snapshot, actualTokens);

            // Score each metric token using GetLastLogits after priming the session
            _session.Reset(contextTokens);

            var logitBuf = new float[MetricTokenizer.VocabSize];
            var totalNegLog = 0f;
            var worstMetric = 0;
            var worstScore = float.MinValue;
            var worstExpected = 0;
            var worstActual = 0;

            for (var m = 0; m < MetricTokenizer.TokensPerSnapshot; m++)
            {
                // Get logits at current position
                _session.GetLastLogits(logitBuf);

                var actual = actualTokens[m];
                var score = ComputeNegLogProb(logitBuf, actual);
                totalNegLog += score;

                if (score > worstScore)
                {
                    worstScore = score;
                    worstMetric = m;
                    worstActual = actual;
                    worstExpected = ArgMax(logitBuf);
                }

                // Feed actual token to advance position (teacher forcing during scoring)
                if (m < MetricTokenizer.TokensPerSnapshot - 1)
                {
                    _session.GenerateNextToken(SamplingOptions.Greedy);
                }
            }

            return new AnomalyScore
            {
                IsWarmup = false,
                Score = totalNegLog / MetricTokenizer.TokensPerSnapshot,
                PodName = snapshot.PodName,
                Timestamp = snapshot.Timestamp,
                WorstMetric = MetricTokenizer.MetricNameOf(worstMetric * MetricTokenizer.BinsPerMetric),
                ExpectedValue = MetricTokenizer.Decode(worstExpected),
                ActualValue = MetricTokenizer.Decode(worstActual),
            };
        }

        /// <summary>Resets the context window. Call after pod restart.</summary>
        public void Reset() => _window.Clear();

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _handle.Dispose();
        }

        // ── Private ──────────────────────────────────────────────────────────

        private static float ComputeNegLogProb(float[] logits, int target)
        {
            var maxVal = logits[0];
            for (var i = 1; i < logits.Length; i++)
            {
                if (logits[i] > maxVal)
                {
                    maxVal = logits[i];
                }
            }

            var sumExp = 0f;
            for (var i = 0; i < logits.Length; i++)
            {
                sumExp += MathF.Exp(logits[i] - maxVal);
            }

            return -(logits[target] - maxVal - MathF.Log(sumExp));
        }

        private static int ArgMax(float[] logits)
        {
            var best = 0;
            for (var i = 1; i < logits.Length; i++)
            {
                if (logits[i] > logits[best])
                {
                    best = i;
                }
            }
            return best;
        }
    }

    /// <summary>Result of scoring one MetricSnapshot.</summary>
    public sealed class AnomalyScore
    {
        /// <summary>True during warmup — window not yet filled.</summary>
        public bool IsWarmup { get; init; }

        /// <summary>
        /// Mean negative log-probability. ~0 = normal, ~3+ = anomaly.
        /// Tune threshold on validation data.
        /// </summary>
        public float Score { get; init; }

        public string PodName { get; init; } = string.Empty;
        public DateTime Timestamp { get; init; }
        public string WorstMetric { get; init; } = string.Empty;
        public float ExpectedValue { get; init; }
        public float ActualValue { get; init; }

        public override string ToString() =>
            IsWarmup
                ? $"[{PodName}] warmup"
                : $"[{PodName}] score={Score:F2} worst={WorstMetric} " +
                  $"expected={ExpectedValue:F1} actual={ActualValue:F1}";
    }
}
