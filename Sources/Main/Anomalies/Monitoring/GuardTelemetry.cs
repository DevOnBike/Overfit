// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// What the guard itself is doing, in the form the guard expects of everything else it watches.
    ///
    /// <para><b>Written because the guard had the exact pathology it exists to eliminate.</b> It only logged.
    /// If its Prometheus queries began failing, or its loop stopped, or its scrape returned nothing, the
    /// result was an absence of incidents — <i>which looks precisely like a healthy cluster</i>. Every
    /// document in this subsystem argues that silence and health must be distinguishable; the component
    /// making that argument could not be distinguished from a healthy cluster itself.</para>
    ///
    /// <para><b>The load-bearing series is <c>overfit_guard_last_cycle_timestamp_seconds</c>.</b> Counters
    /// tell you what happened; only that one makes "the guard has not completed a cycle in fifteen minutes"
    /// expressible as an alert, and that is the single alert every deployment of this needs. A guard that has
    /// stopped is worse than one that never started, because somebody is relying on it.</para>
    ///
    /// <para><b>Blind channels are exported as a number, not just logged.</b> "How much of what I asked for
    /// can this thing actually see" is a question an operator should be able to graph over a month, not
    /// reconstruct from log lines.</para>
    ///
    /// <para>Thread-safe for readers: a scrape can arrive mid-cycle, and a torn count is a worse answer than
    /// a slightly stale one.</para>
    /// </summary>
    public sealed class GuardTelemetry
    {
        private long _cycles;
        private long _failedCycles;
        private long _stateWriteFailures;
        private long _findings;
        private long _opened;
        private long _resolved;
        private long _suppressed;
        private long _lastCycleUnixSeconds;
        private int _pods;
        private int _blind;
        private int _unevaluable;

        /// <summary>Records one completed cycle.</summary>
        public void Cycle(in GuardCycleResult result, int pods, DateTimeOffset at, bool suppressed)
        {
            Interlocked.Increment(ref _cycles);
            Interlocked.Add(ref _findings, result.Findings);
            Interlocked.Add(ref _opened, result.Opened);
            Interlocked.Add(ref _resolved, result.Resolved);
            Interlocked.Exchange(ref _lastCycleUnixSeconds, at.ToUnixTimeSeconds());
            Interlocked.Exchange(ref _pods, pods);
            Interlocked.Exchange(ref _blind, result.BlindMetrics);
            Interlocked.Exchange(ref _unevaluable, result.UnevaluableMetrics);

            if (suppressed)
            {
                Interlocked.Increment(ref _suppressed);
            }
        }

        /// <summary>
        /// Records a cycle that threw.
        ///
        /// <para>Counted separately from a completed one, and that distinction is the point: a loop that runs
        /// every five minutes and fails every time still updates no incident counters, so without this the
        /// only evidence is a log line nobody is watching.</para>
        /// </summary>
        public void Failed()
        {
            Interlocked.Increment(ref _failedCycles);
        }

        /// <summary>
        /// Records a cycle whose durable state could not be read or written.
        ///
        /// <para>Its own series because the failure is otherwise perfectly silent: the stores swallow their
        /// exceptions so a full volume degrades rather than crashes, cycles keep completing, incidents keep
        /// being reported, and the only symptom arrives at the next restart when everything reopens at once.
        /// An alert on this fires hours before that.</para>
        /// </summary>
        public void StateWriteFailed()
        {
            Interlocked.Increment(ref _stateWriteFailures);
        }

        /// <summary>Renders the Prometheus text exposition format.</summary>
        public string ToPrometheusText()
        {
            var text = new StringBuilder();

            Counter(text, "overfit_guard_cycles_total",
                "Evaluation cycles the guard has completed.", Interlocked.Read(ref _cycles));

            Counter(text, "overfit_guard_state_failures_total",
                "Cycles whose durable state could not be read or written. Incidents will not survive the next "
                + "restart, and the restart is when anyone would otherwise notice.",
                Interlocked.Read(ref _stateWriteFailures));

            Counter(text, "overfit_guard_cycle_failures_total",
                "Cycles that threw and were skipped. A guard failing every cycle reports no incidents, "
                + "which is indistinguishable from a healthy cluster.", Interlocked.Read(ref _failedCycles));

            Counter(text, "overfit_guard_findings_total",
                "Signals that reached a decided anomaly.", Interlocked.Read(ref _findings));

            Counter(text, "overfit_guard_incidents_opened_total",
                "Incidents seen for the first time — the only count that should page anyone.",
                Interlocked.Read(ref _opened));

            Counter(text, "overfit_guard_incidents_resolved_total",
                "Incidents that closed.", Interlocked.Read(ref _resolved));

            Counter(text, "overfit_guard_suppressed_cycles_total",
                "Cycles inside a declared maintenance window.", Interlocked.Read(ref _suppressed));

            Gauge(text, "overfit_guard_last_cycle_timestamp_seconds",
                "When the last cycle completed. Alert on this going stale: a guard that has stopped is "
                + "worse than one that never started, because somebody is relying on it.",
                Interlocked.Read(ref _lastCycleUnixSeconds));

            Gauge(text, "overfit_guard_pods", "Pods evaluated in the last cycle.", Volatile.Read(ref _pods));

            Gauge(text, "overfit_guard_blind_metrics",
                "Channels no pod reported. Each one produces no findings, which looks exactly like health.",
                Volatile.Read(ref _blind));

            Gauge(text, "overfit_guard_unevaluable_metrics",
                "Channels that were reported and still could not be judged.",
                Volatile.Read(ref _unevaluable));

            return text.ToString();
        }

        private static void Counter(StringBuilder text, string name, string help, double value)
            => Write(text, name, help, "counter", value);

        private static void Gauge(StringBuilder text, string name, string help, double value)
            => Write(text, name, help, "gauge", value);

        private static void Write(StringBuilder text, string name, string help, string type, double value)
        {
            text.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n')
                .Append("# TYPE ").Append(name).Append(' ').Append(type).Append('\n')
                .Append(name).Append(' ').Append(value.ToString("R", CultureInfo.InvariantCulture)).Append('\n');
        }
    }
}
