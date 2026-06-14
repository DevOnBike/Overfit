// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Serving
{
    /// <summary>
    /// Aggregated result of a serving load test: latency percentiles (TTFT / ITL / end-to-end),
    /// throughput, goodput, error rate, and a single holistic <see cref="Score"/> that rewards low
    /// latency, high throughput and high concurrency while dividing out the serving cost. The metric
    /// shape mirrors the production-serving methodology used in the GPU world (vLLM / Modal style
    /// leaderboards), so a pure-.NET CPU server can be measured and compared apples-to-apples.
    ///
    /// The score is intentionally the same form as a GPU inference leaderboard:
    /// <c>score = (goodput × concurrency) / (TTFT_p95_s × ITL_p95_s × cost_units)</c>, where
    /// <c>goodput = throughput × (1 − error_rate)</c>. Adding capacity (cost units) without a
    /// proportional latency win does not raise the score — efficiency, not brute force, is rewarded.
    /// </summary>
    public sealed class ServingLoadReport
    {
        private ServingLoadReport()
        {
        }

        /// <summary>Number of concurrent virtual users that drove the test.</summary>
        public int Concurrency { get; private init; }

        /// <summary>Wall-clock duration of the measured window, seconds.</summary>
        public double WallClockSeconds { get; private init; }

        /// <summary>Serving cost units in the score denominator — e.g. CPU cores, server count, or 1.
        /// The GPU-leaderboard analogue is <c>total_GPUs</c>; it stops "throw more hardware at it"
        /// from inflating the score.</summary>
        public double CostUnits { get; private init; }

        public int TotalRequests { get; private init; }
        public int SuccessfulRequests { get; private init; }
        public int FailedRequests => TotalRequests - SuccessfulRequests;

        /// <summary>Fraction of requests that failed (HTTP error, timeout, transport drop), 0.0–1.0.</summary>
        public double ErrorRate => TotalRequests == 0 ? 0.0 : (double)FailedRequests / TotalRequests;

        public double TimeToFirstTokenP50Ms { get; private init; }
        public double TimeToFirstTokenP95Ms { get; private init; }
        public double TimeToFirstTokenP99Ms { get; private init; }

        public double InterTokenLatencyP50Ms { get; private init; }
        public double InterTokenLatencyP95Ms { get; private init; }
        public double InterTokenLatencyP99Ms { get; private init; }

        public double EndToEndP50Ms { get; private init; }
        public double EndToEndP95Ms { get; private init; }
        public double EndToEndP99Ms { get; private init; }

        /// <summary>Aggregate output tokens/chunks per second across all successful requests over the
        /// measured wall-clock window — the throughput term.</summary>
        public double ThroughputTokensPerSecond { get; private init; }

        /// <summary>Effective throughput after penalising failures: <c>throughput × (1 − error_rate)</c>.</summary>
        public double Goodput => ThroughputTokensPerSecond * (1.0 - ErrorRate);

        /// <summary>The holistic leaderboard score (higher is better). Zero when there is no successful
        /// traffic or a degenerate latency reading (avoids a divide-by-zero blow-up).</summary>
        public double Score
        {
            get
            {
                var ttftSeconds = TimeToFirstTokenP95Ms / 1000.0;
                var itlSeconds = InterTokenLatencyP95Ms / 1000.0;
                var denominator = ttftSeconds * itlSeconds * CostUnits;

                if (SuccessfulRequests == 0 || denominator <= 0.0)
                {
                    return 0.0;
                }

                return Goodput * Concurrency / denominator;
            }
        }

        /// <summary>
        /// Build a report from the per-request <paramref name="samples"/>, the <paramref name="concurrency"/>
        /// that produced them, the measured <paramref name="wallClockSeconds"/>, and the
        /// <paramref name="costUnits"/> for the score denominator (CPU cores / servers / 1).
        /// </summary>
        public static ServingLoadReport From(
            IReadOnlyList<ServingRequestSample> samples,
            int concurrency,
            double wallClockSeconds,
            double costUnits = 1.0)
        {
            ArgumentNullException.ThrowIfNull(samples);

            var successes = 0;
            long totalTokens = 0;

            for (var i = 0; i < samples.Count; i++)
            {
                if (samples[i].Succeeded)
                {
                    successes++;
                    totalTokens += samples[i].OutputTokens;
                }
            }

            var ttft = new double[successes];
            var itl = new double[successes];
            var e2e = new double[successes];

            var k = 0;
            for (var i = 0; i < samples.Count; i++)
            {
                var s = samples[i];
                if (!s.Succeeded)
                {
                    continue;
                }

                ttft[k] = s.TimeToFirstTokenMs;
                itl[k] = s.InterTokenLatencyMs;
                e2e[k] = s.EndToEndMs;
                k++;
            }

            Array.Sort(ttft);
            Array.Sort(itl);
            Array.Sort(e2e);

            var throughput = wallClockSeconds > 0.0 ? totalTokens / wallClockSeconds : 0.0;

            return new ServingLoadReport
            {
                Concurrency = concurrency,
                WallClockSeconds = wallClockSeconds,
                CostUnits = costUnits <= 0.0 ? 1.0 : costUnits,
                TotalRequests = samples.Count,
                SuccessfulRequests = successes,
                ThroughputTokensPerSecond = throughput,
                TimeToFirstTokenP50Ms = Percentile(ttft, 0.50),
                TimeToFirstTokenP95Ms = Percentile(ttft, 0.95),
                TimeToFirstTokenP99Ms = Percentile(ttft, 0.99),
                InterTokenLatencyP50Ms = Percentile(itl, 0.50),
                InterTokenLatencyP95Ms = Percentile(itl, 0.95),
                InterTokenLatencyP99Ms = Percentile(itl, 0.99),
                EndToEndP50Ms = Percentile(e2e, 0.50),
                EndToEndP95Ms = Percentile(e2e, 0.95),
                EndToEndP99Ms = Percentile(e2e, 0.99),
            };
        }

        /// <summary>Linearly-interpolated percentile of an ascending-sorted array (<paramref name="q"/>
        /// in 0.0–1.0). Empty → 0; single element → that element.</summary>
        internal static double Percentile(double[] sortedAscending, double q)
        {
            if (sortedAscending.Length == 0)
            {
                return 0.0;
            }

            if (sortedAscending.Length == 1)
            {
                return sortedAscending[0];
            }

            var rank = q * (sortedAscending.Length - 1);
            var lo = (int)Math.Floor(rank);
            var hi = (int)Math.Ceiling(rank);
            var fraction = rank - lo;

            return sortedAscending[lo] + (sortedAscending[hi] - sortedAscending[lo]) * fraction;
        }
    }
}
