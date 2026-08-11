// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Demo.LocalAgent.Observability
{
    /// <summary>
    /// A fixed-bucket histogram in the Prometheus exposition shape: cumulative <c>_bucket</c> counters, a
    /// <c>_sum</c> and a <c>_count</c>.
    ///
    /// <para><b>Why the demo owns this rather than taking it from a library.</b> The only package that
    /// exported a Prometheus scrape endpoint here was
    /// <c>OpenTelemetry.Exporter.Prometheus.AspNetCore</c>, which has been in prerelease since 2022 — 33
    /// versions, not one stable, while the rest of its suite ships stable. The product itself does not use
    /// it: both <c>Sources/Server.AspNet/Endpoints/MetricsEndpoints.cs</c> and
    /// <c>Sources/Anomalies/Monitoring/GuardTelemetry.cs</c> write the text by hand. A demo that
    /// demonstrates the opposite of what the product does is worse than no demo, so this follows the
    /// product.</para>
    ///
    /// <para><b>Cumulative buckets, not counts per bucket.</b> Prometheus defines <c>le</c> as "less than or
    /// equal", so every bucket includes all the ones below it and the last is <c>+Inf</c>. Emitting
    /// per-bucket counts instead produces a histogram that parses, renders, and is wrong — which is the
    /// failure mode worth naming, because nothing rejects it.</para>
    /// </summary>
    internal sealed class BucketHistogram
    {
        private readonly double[] _bounds;
        private readonly long[] _counts;
        private readonly Lock _gate = new();
        private double _sum;
        private long _count;

        /// <param name="bounds">Upper bounds, ascending. <c>+Inf</c> is implied and must not be listed.</param>
        public BucketHistogram(double[] bounds)
        {
            _bounds = bounds;

            // One more than the bounds: the overflow bucket that becomes +Inf.
            _counts = new long[bounds.Length + 1];
        }

        public void Record(double value)
        {
            var index = _bounds.Length;

            for (var i = 0; i < _bounds.Length; i++)
            {
                if (value <= _bounds[i])
                {
                    index = i;

                    break;
                }
            }

            // A lock rather than Interlocked on three fields: sum, count and the bucket must move together,
            // or a scrape landing between them reports a histogram whose buckets do not add up to its count.
            // Contention is irrelevant here — this is recorded once per request, not per token.
            lock (_gate)
            {
                _counts[index]++;
                _sum += value;
                _count++;
            }
        }

        public void Write(StringBuilder text, string name, string help)
        {
            text.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
            text.Append("# TYPE ").Append(name).Append(" histogram\n");

            long cumulative;
            double sum;
            long count;
            var snapshot = new long[_counts.Length];

            lock (_gate)
            {
                Array.Copy(_counts, snapshot, _counts.Length);
                sum = _sum;
                count = _count;
            }

            cumulative = 0;

            for (var i = 0; i < _bounds.Length; i++)
            {
                cumulative += snapshot[i];

                text.Append(name).Append("_bucket{le=\"")
                    .Append(_bounds[i].ToString("G", System.Globalization.CultureInfo.InvariantCulture))
                    .Append("\"} ").Append(cumulative).Append('\n');
            }

            text.Append(name).Append("_bucket{le=\"+Inf\"} ").Append(count).Append('\n');
            text.Append(name).Append("_sum ")
                .Append(sum.ToString("G17", System.Globalization.CultureInfo.InvariantCulture)).Append('\n');
            text.Append(name).Append("_count ").Append(count).Append('\n');
        }
    }
}
