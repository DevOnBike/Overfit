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
    /// <para><b>The load-bearing series is <c>overfit_guard_last_cycle_timestamp_seconds</c>, and the alert
    /// on it has to be written a specific way.</b> Counters tell you what happened; only that one can express
    /// "the guard has not completed a cycle in fifteen minutes", and that is the single alert every
    /// deployment of this needs. A guard that has stopped is worse than one that never started, because
    /// somebody is relying on it.</para>
    ///
    /// <para><b>The obvious expression cannot fire in the case that matters, and this was measured rather
    /// than reasoned.</b> <c>time() - overfit_guard_last_cycle_timestamp_seconds &gt; 900</c> evaluates over
    /// an empty vector once the guard's pod is gone — the series goes with it — so the alert returns to
    /// <c>inactive</c> and reports "fine" precisely when the cluster has stopped being watched. Measured
    /// 2026-08-05 by scaling the guard to zero: <c>pending</c> for 60 seconds while the last sample was still
    /// returned, then <c>inactive</c> for six minutes, no alert in Alertmanager, no notification. The same
    /// test with <c>absent(...) or (time() - ... &gt; 900)</c>: firing after 60 seconds, two notifications
    /// delivered, none failed. Ship <c>k8s/lab/guard-alerts.yaml</c> rather than writing the rule from
    /// memory.</para>
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
        private long _labels;
        private long _realLabels;
        private long _muted;
        private int _activeSuppressions;
        private long _findings;
        private long _opened;
        private long _resolved;
        private long _suppressed;
        private long _lastCycleUnixSeconds;
        private int _pods;
        private int _blind;
        private int _unevaluable;

        /// <param name="scope">
        /// Which population this instrument is about. Empty for a single-scope process, which renders exactly
        /// the series it always did — an upgrade must not silently retarget somebody's alerts.
        /// </param>
        public GuardTelemetry(string scope = "")
        {
            Scope = scope ?? string.Empty;
        }

        /// <summary>
        /// The <c>scope</c> label every series of this instrument carries, or empty for none.
        ///
        /// <para><b>Not cosmetic, and the reason is one specific series.</b> With several scopes and no label,
        /// <c>overfit_guard_last_cycle_timestamp_seconds</c> becomes the MOST RECENT across scopes — so one
        /// healthy scope keeps it fresh while the rest are stalled and "the guard has stopped" never fires.
        /// That single alert is the one every deployment of this needs, and losing it would reintroduce the
        /// pathology this class was written to remove, through a change that looks like configuration.</para>
        /// </summary>
        public string Scope
        {
            get;
        }

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

        /// <summary>
        /// Records what operator feedback is currently doing to the guard.
        ///
        /// <para><b>This is the series that keeps the feature honest.</b> Everything an operator can press
        /// makes the guard quieter, and a mute nobody can see is indistinguishable from a detector that
        /// stopped working. An alert on <c>overfit_guard_suppressions_active</c> climbing, or on
        /// <c>overfit_guard_findings_muted_total</c> outrunning the findings that survive, is how a team
        /// notices they have silenced their way to a green dashboard.</para>
        /// </summary>
        public void Feedback(int activeSuppressions, int mutedThisCycle, int labels, int realLabels)
        {
            Interlocked.Exchange(ref _activeSuppressions, activeSuppressions);
            Interlocked.Add(ref _muted, mutedThisCycle);
            Interlocked.Exchange(ref _labels, labels);
            Interlocked.Exchange(ref _realLabels, realLabels);
        }

        /// <summary>One exported series: its name, its documentation, and how to read it off an instrument.</summary>
        private readonly record struct Series(string Name, string Help, string Type, Func<GuardTelemetry, double> Read);

        /// <summary>
        /// Every series this guard exports, in one table so that a metric cannot be documented in one place
        /// and rendered in another.
        ///
        /// <para>Delegates rather than reflection: this library is Native-AOT compiled in CI and a property
        /// walk would not survive trimming.</para>
        /// </summary>
        private static readonly Series[] Catalog =
        [
            new("overfit_guard_cycles_total", "Evaluation cycles the guard has completed.",
                "counter", t => Interlocked.Read(ref t._cycles)),

            new("overfit_guard_suppressions_active",
                "Operator suppressions muting a signal right now. Climbing without bound is a team silencing "
                + "its way to a green dashboard.",
                "gauge", t => Volatile.Read(ref t._activeSuppressions)),

            new("overfit_guard_findings_muted_total",
                "Findings dropped because an operator asked not to hear them.",
                "counter", t => Interlocked.Read(ref t._muted)),

            new("overfit_guard_labels_total", "Operator judgements recorded about past incidents.",
                "gauge", t => Interlocked.Read(ref t._labels)),

            new("overfit_guard_labels_real",
                "Judgements marking a finding as correct. These constrain every future floor proposal; a "
                + "feedback loop with none of them converges on a detector that reports nothing.",
                "gauge", t => Interlocked.Read(ref t._realLabels)),

            new("overfit_guard_state_failures_total",
                "Cycles whose durable state could not be read or written. Incidents will not survive the next "
                + "restart, and the restart is when anyone would otherwise notice.",
                "counter", t => Interlocked.Read(ref t._stateWriteFailures)),

            new("overfit_guard_cycle_failures_total",
                "Cycles that threw and were skipped. A guard failing every cycle reports no incidents, "
                + "which is indistinguishable from a healthy cluster.",
                "counter", t => Interlocked.Read(ref t._failedCycles)),

            new("overfit_guard_findings_total", "Signals that reached a decided anomaly.",
                "counter", t => Interlocked.Read(ref t._findings)),

            new("overfit_guard_incidents_opened_total",
                "Incidents seen for the first time — the only count that should page anyone.",
                "counter", t => Interlocked.Read(ref t._opened)),

            new("overfit_guard_incidents_resolved_total", "Incidents that closed.",
                "counter", t => Interlocked.Read(ref t._resolved)),

            new("overfit_guard_suppressed_cycles_total", "Cycles inside a declared maintenance window.",
                "counter", t => Interlocked.Read(ref t._suppressed)),

            // The HELP text an operator reads in their OWN Prometheus, which is the only documentation most
            // of them will ever see for this series — so the trap goes here rather than only in the source.
            new("overfit_guard_last_cycle_timestamp_seconds",
                "When the last cycle completed. Alert with absent() OR a staleness comparison, never the "
                + "comparison alone: this series disappears with the pod, so a time()-based rule goes "
                + "inactive exactly when the guard is gone. A guard that has stopped is worse than one that "
                + "never started, because somebody is relying on it.",
                "gauge", t => Interlocked.Read(ref t._lastCycleUnixSeconds)),

            new("overfit_guard_pods", "Pods evaluated in the last cycle.",
                "gauge", t => Volatile.Read(ref t._pods)),

            new("overfit_guard_blind_metrics",
                "Channels no pod reported. Each one produces no findings, which looks exactly like health.",
                "gauge", t => Volatile.Read(ref t._blind)),

            new("overfit_guard_unevaluable_metrics",
                "Channels that were reported and still could not be judged.",
                "gauge", t => Volatile.Read(ref t._unevaluable)),
        ];

        /// <summary>Renders the Prometheus text exposition format for this instrument alone.</summary>
        public string ToPrometheusText()
        {
            return Render([this]);
        }

        /// <summary>
        /// Renders several instruments into ONE exposition document.
        ///
        /// <para><b>Concatenating per-instrument renderings does not work, and fails in a way that takes the
        /// whole endpoint down rather than one scope.</b> The text format allows a metric name exactly one
        /// <c># HELP</c> and one <c># TYPE</c> line per document; a second one makes Prometheus reject the
        /// entire scrape. So headers are written once per series and one sample line follows per scope.</para>
        /// </summary>
        public static string Render(IReadOnlyList<GuardTelemetry> instruments)
        {
            ArgumentNullException.ThrowIfNull(instruments);

            var text = new StringBuilder();

            for (var s = 0; s < Catalog.Length; s++)
            {
                var series = Catalog[s];

                text.Append("# HELP ").Append(series.Name).Append(' ').Append(series.Help).Append('\n')
                    .Append("# TYPE ").Append(series.Name).Append(' ').Append(series.Type).Append('\n');

                for (var i = 0; i < instruments.Count; i++)
                {
                    var instrument = instruments[i];

                    text.Append(series.Name);

                    if (instrument.Scope.Length > 0)
                    {
                        text.Append("{scope=\"");
                        AppendEscaped(text, instrument.Scope);
                        text.Append("\"}");
                    }

                    text.Append(' ')
                        .Append(series.Read(instrument).ToString("R", CultureInfo.InvariantCulture))
                        .Append('\n');
                }
            }

            return text.ToString();
        }

        /// <summary>
        /// Escapes a label value per the text format: backslash, double quote and newline.
        ///
        /// <para>A scope name is derived from a namespace and a pod regex, and a regex may legitimately
        /// contain a backslash — <c>api-\d+</c> is an ordinary selector. Unescaped it would terminate the
        /// label value early and corrupt every series after it in the document.</para>
        /// </summary>
        private static void AppendEscaped(StringBuilder text, string value)
        {
            for (var i = 0; i < value.Length; i++)
            {
                var c = value[i];

                if (c == '\\' || c == '"')
                {
                    text.Append('\\').Append(c);

                    continue;
                }

                if (c == '\n')
                {
                    text.Append("\\n");

                    continue;
                }

                text.Append(c);
            }
        }
    }
}
