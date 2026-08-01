// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;

namespace DevOnBike.Overfit.LabWorkload
{
    /// <summary>
    /// Prometheus exposition, hand-rolled.
    ///
    /// <para>Same choice the inference server made and for the same reason: the OpenTelemetry Prometheus
    /// exporter pulls in reflection-heavy dependencies, and a few dozen lines of text formatting cost less
    /// than the dependency. Here it buys something extra — <b>the metric names are chosen deliberately</b>,
    /// which is the point of a stand-in for a client's application. The guard has to reach them through its
    /// configuration file rather than by recognising anything.</para>
    ///
    /// <para>The runtime figures are real: <c>GC.GetTotalMemory</c>, <c>GC.GetGCMemoryInfo</c> and
    /// <c>ThreadPool.PendingWorkItemCount</c> report what the process is actually doing. Only the names are
    /// this project's, and they match what the lab's guard configuration maps.</para>
    /// </summary>
    internal sealed class WorkloadMetrics
    {
        /// <summary>
        /// Histogram bucket edges in seconds. Wide enough at the top to hold a stall, because a bucket set
        /// that tops out below the fault being injected turns every high quantile into the same number — the
        /// quantile pins to the last edge and the signal the detector reads goes flat.
        /// </summary>
        private static readonly double[] Buckets =
            [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];

        private readonly long[] _bucketCounts = new long[Buckets.Length];
        private readonly string _role;

        private long _requests;
        private long _errors;
        private long _sumMicroseconds;

        public WorkloadMetrics(string role) => _role = role;

        public void Observe(double seconds, bool failed)
        {
            Interlocked.Increment(ref _requests);
            Interlocked.Add(ref _sumMicroseconds, (long)(seconds * 1_000_000.0));

            if (failed)
            {
                Interlocked.Increment(ref _errors);
            }

            for (var i = 0; i < Buckets.Length; i++)
            {
                if (seconds <= Buckets[i])
                {
                    Interlocked.Increment(ref _bucketCounts[i]);
                }
            }
        }

        public string Render()
        {
            var text = new StringBuilder(2048);
            var requests = Interlocked.Read(ref _requests);
            var role = $"{{role=\"{_role}\"}}";

            Counter(text, "labapp_requests_total", "Requests completed, successfully or not.", requests, role);
            Counter(text, "labapp_errors_total", "Requests answered 5xx.", Interlocked.Read(ref _errors), role);

            text.Append("# HELP labapp_request_duration_seconds Service time.\n")
                .Append("# TYPE labapp_request_duration_seconds histogram\n");

            // Cumulative, and +Inf must equal the count or Prometheus rejects the family. The bucket counts
            // are incremented independently per observation, so they are already cumulative by construction.
            for (var i = 0; i < Buckets.Length; i++)
            {
                text.Append("labapp_request_duration_seconds_bucket{role=\"")
                    .Append(_role)
                    .Append("\",le=\"")
                    .Append(Buckets[i].ToString("0.###", CultureInfo.InvariantCulture))
                    .Append("\"} ")
                    .Append(Interlocked.Read(ref _bucketCounts[i]).ToString(CultureInfo.InvariantCulture))
                    .Append('\n');
            }

            text.Append("labapp_request_duration_seconds_bucket{role=\"")
                .Append(_role)
                .Append("\",le=\"+Inf\"} ")
                .Append(requests.ToString(CultureInfo.InvariantCulture))
                .Append('\n');

            text.Append("labapp_request_duration_seconds_sum")
                .Append(role)
                .Append(' ')
                .Append((Interlocked.Read(ref _sumMicroseconds) / 1_000_000.0)
                    .ToString("0.######", CultureInfo.InvariantCulture))
                .Append('\n');

            text.Append("labapp_request_duration_seconds_count")
                .Append(role)
                .Append(' ')
                .Append(requests.ToString(CultureInfo.InvariantCulture))
                .Append("\n\n");

            // Real runtime figures under this project's own names — the values are not simulated, only the
            // naming is ours, which is what makes the configuration file do real work.
            var info = GC.GetGCMemoryInfo();

            Gauge(text, "dotnet_gc_heap_size_bytes", "Managed heap size.", GC.GetTotalMemory(false), role);
            Gauge(text, "dotnet_gc_committed_bytes", "Committed heap bytes.", info.TotalCommittedBytes, role);
            Counter(text, "dotnet_gc_pause_seconds_total", "Total GC pause time.",
                GC.GetTotalPauseDuration().TotalSeconds, role);
            Gauge(text, "dotnet_threadpool_queue_length", "Work items waiting for a thread.",
                ThreadPool.PendingWorkItemCount, role);
            Gauge(text, "dotnet_process_working_set_bytes", "Process working set.",
                Environment.WorkingSet, role);

            return text.ToString();
        }

        private static void Counter(StringBuilder text, string name, string help, double value, string labels)
        {
            text.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n')
                .Append("# TYPE ").Append(name).Append(" counter\n")
                .Append(name).Append(labels).Append(' ')
                .Append(value.ToString("0.######", CultureInfo.InvariantCulture)).Append("\n\n");
        }

        private static void Gauge(StringBuilder text, string name, string help, double value, string labels)
        {
            text.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n')
                .Append("# TYPE ").Append(name).Append(" gauge\n")
                .Append(name).Append(labels).Append(' ')
                .Append(value.ToString("0.######", CultureInfo.InvariantCulture)).Append("\n\n");
        }
    }
}
