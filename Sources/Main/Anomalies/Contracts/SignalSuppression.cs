// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One operator's decision to stop hearing about one signal on one subject, until a stated moment.
    ///
    /// <para><b>The expiry is not optional and that is the whole design.</b> A suppression without an end date
    /// is a configuration change wearing the clothes of an acknowledgement: nobody reviews it, nothing
    /// reminds anyone it exists, and the pod most likely to carry one is the pod that eventually breaks. An
    /// operator who genuinely wants a permanent change has a ConfigMap for that, where it is visible in
    /// review and in version control.</para>
    ///
    /// <para><b>Scoped to a subject and a signal, never to a signal alone.</b> "Mute GC pause" across a
    /// deployment is indistinguishable from removing the channel, and it is the request an operator makes at
    /// three in the morning about one misbehaving replica. Narrowing it to the replica costs the operator
    /// nothing and costs the guard almost nothing.</para>
    /// </summary>
    /// <param name="Pod">Pod this covers, or empty for a workload-level subject.</param>
    /// <param name="Workload">Workload this covers. Required — a suppression with neither is namespace-wide.</param>
    /// <param name="Signal">Metric name.</param>
    /// <param name="Until">When it stops applying.</param>
    /// <param name="IncidentId">The incident the operator was looking at.</param>
    /// <param name="Reason">What they typed.</param>
    /// <param name="Magnitude">
    /// How large the dismissed finding was, in the signal's own units, or <see cref="double.NaN"/> when it
    /// had no measurable size.
    ///
    /// <para><b>Without this a dismissal silences a channel rather than an alert, and it was measured
    /// doing exactly that.</b> Replaying the injected-fault panel after a simulated shadow week — 23
    /// dismissals, each a seven-day mute — left <c>cpu 2.5x</c> undetected on one pod <i>and</i> on every
    /// pod at once. The shadow week's noise was small CPU gaps; muting the pair (pod, CpuUsageRatio) then
    /// hid a two-and-a-half-fold rise as well. Literally what the operator asked for, and nothing anyone
    /// means by it.</para>
    ///
    /// <para>See <see cref="Ceiling"/> for what is muted as a result.</para>
    /// </param>
    public readonly record struct SignalSuppression(
        string Pod,
        string Workload,
        string Signal,
        DateTimeOffset Until,
        long IncidentId,
        string Reason,
        double Magnitude = double.NaN)
    {
        /// <summary>
        /// Headroom over the dismissed size. The same 1.25 the floor proposals use, so a finding that is
        /// merely the same event again stays muted and one materially larger does not.
        /// </summary>
        private const double Margin = 1.25;

        /// <summary>
        /// The largest finding this mute covers, or <see cref="double.PositiveInfinity"/> when the dismissed
        /// one had no measurable size — there is then nothing to compare against, and refusing to mute would
        /// make the acknowledgement do nothing at all.
        /// </summary>
        public double Ceiling => double.IsFinite(Magnitude) && Magnitude > 0.0
            ? Magnitude * Margin
            : double.PositiveInfinity;
        /// <summary>Whether this still applies at <paramref name="at"/>.</summary>
        public bool IsActive(DateTimeOffset at) => at < Until;

        /// <summary>
        /// Whether it covers <paramref name="subject"/> and <paramref name="signal"/>.
        ///
        /// <para>A pod-scoped suppression covers exactly that pod. One with no pod covers the workload's own
        /// findings <b>and</b> its replicas, because an operator muting "the deployment's memory trend" means
        /// the deployment, and having to mute twelve pods one at a time is how a feature meant to build trust
        /// becomes the reason somebody mutes the whole tool instead.</para>
        /// </summary>
        /// <param name="magnitude">
        /// How large the finding being considered is. A finding <b>larger than</b> <see cref="Ceiling"/> is
        /// not covered: it is a different event on the same signal, and the operator dismissed the smaller
        /// one. <see cref="double.NaN"/> means unmeasurable and is covered.
        /// </param>
        public bool Covers(in IncidentSubject subject, string signal, double magnitude = double.NaN)
        {
            if (!string.Equals(Signal, signal, StringComparison.Ordinal))
            {
                return false;
            }

            if (double.IsFinite(magnitude) && magnitude > Ceiling)
            {
                return false;
            }

            if (Workload.Length > 0
                && !string.Equals(Workload, subject.Workload, StringComparison.Ordinal))
            {
                return false;
            }

            return Pod.Length == 0 || string.Equals(Pod, subject.Pod, StringComparison.Ordinal);
        }
    }
}
