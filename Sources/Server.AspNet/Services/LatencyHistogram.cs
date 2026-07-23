// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// A fixed-bucket latency histogram accumulated for the Prometheus <c>/metrics</c> endpoint. The Meter's
    /// own <c>Histogram&lt;double&gt;</c> records the same samples for OpenTelemetry / dotnet-counters, but it
    /// does not expose its accumulated buckets for scraping — so this keeps the cumulative counts, sum and
    /// count that Prometheus's histogram format needs. Buckets are chosen for LLM latencies (tens of ms to
    /// several seconds). Recording is under a short lock — it happens once per request, off the decode path.
    /// </summary>
    internal sealed class LatencyHistogram
    {
        // Upper bounds in seconds (le = less-than-or-equal), ascending.
        private static readonly double[] Bounds =
            [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

        private readonly long[] _bucketCounts = new long[Bounds.Length + 1];   // +1 = the +Inf overflow bucket
        private readonly object _lock = new();
        private long _count;
        private double _sum;

        public void Record(double seconds)
        {
            var index = 0;
            while (index < Bounds.Length && seconds > Bounds[index])
            {
                index++;
            }

            lock (_lock)
            {
                _bucketCounts[index]++;
                _count++;
                _sum += seconds;
            }
        }

        /// <summary>Writes the histogram in Prometheus exposition format; <paramref name="name"/> gains the
        /// <c>_seconds</c> unit suffix (e.g. <c>overfit_chat_ttft</c> → <c>overfit_chat_ttft_seconds</c>).</summary>
        public void Write(StringBuilder sb, string name, string help)
        {
            long[] snapshot;
            long count;
            double sum;
            lock (_lock)
            {
                snapshot = (long[])_bucketCounts.Clone();
                count = _count;
                sum = _sum;
            }

            sb.Append("# HELP ").Append(name).Append("_seconds ").Append(help).Append('\n');
            sb.Append("# TYPE ").Append(name).Append("_seconds histogram\n");

            var cumulative = 0L;
            for (var i = 0; i < Bounds.Length; i++)
            {
                cumulative += snapshot[i];
                sb.Append(name).Append("_seconds_bucket{le=\"")
                  .Append(Bounds[i].ToString("0.###", CultureInfo.InvariantCulture))
                  .Append("\"} ").Append(cumulative.ToString(CultureInfo.InvariantCulture)).Append('\n');
            }

            cumulative += snapshot[Bounds.Length];
            sb.Append(name).Append("_seconds_bucket{le=\"+Inf\"} ")
              .Append(cumulative.ToString(CultureInfo.InvariantCulture)).Append('\n');
            sb.Append(name).Append("_seconds_sum ")
              .Append(sum.ToString("0.######", CultureInfo.InvariantCulture)).Append('\n');
            sb.Append(name).Append("_seconds_count ")
              .Append(count.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }
    }
}
