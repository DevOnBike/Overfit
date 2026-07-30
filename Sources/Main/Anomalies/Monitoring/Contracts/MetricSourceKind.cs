// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    /// What shape a source metric has, which is what decides the PromQL wrapped around it.
    ///
    /// <para><b>This exists so the client names a metric rather than writing a query.</b> Onboarding used to
    /// mean handing over full PromQL per feature — <c>histogram_quantile(0.95, sum by (pod, le) (rate(…)))</c>
    /// and the rest — which is the part nobody gets right first time and where a mistake returns an empty
    /// result that Prometheus reports as <c>success</c>. Saying "our request histogram is called X" is a
    /// question somebody can answer from memory.</para>
    /// </summary>
    public enum MetricSourceKind
    {
        /// <summary>
        /// A value that is what it is right now — heap size, queue length, working set. Summed by pod so
        /// per-container series collapse to one per pod, which cAdvisor requires: it emits both.
        /// </summary>
        Gauge = 0,

        /// <summary>
        /// A monotonically increasing counter. Wrapped in <c>rate()</c>, because the counter's absolute value
        /// is an artefact of how long the process has been up and comparing it across pods compares uptimes.
        /// </summary>
        Counter = 1,

        /// <summary>
        /// A counter whose <i>increase</i> over the window is the quantity of interest — restarts, OOM kills.
        /// Distinct from <see cref="Counter"/> because a rate turns "two restarts" into a fraction per second
        /// that reads as noise, while the count is the fact.
        /// </summary>
        EventCount = 2,

        /// <summary>
        /// A histogram, named <b>without</b> the <c>_bucket</c> suffix. Wrapped in
        /// <c>histogram_quantile</c> over <c>sum by (pod, le)</c> — and the <c>le</c> is not optional: drop
        /// the pod label through the quantile and every sample is discarded during parsing, a silent and
        /// total loss of the feature.
        /// </summary>
        HistogramSeconds = 3,

        /// <summary>
        /// Already a fraction of one — a ratio, a utilisation. Passed through untouched, because dividing or
        /// rating something already normalised produces a number with no meaning.
        /// </summary>
        Ratio = 4,
    }
}
