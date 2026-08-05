// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>What one evaluation cycle did.</summary>
    /// <param name="Findings">Signals that reached a decided anomaly this cycle.</param>
    /// <param name="Incidents">Groups they formed.</param>
    /// <param name="Opened">Incidents seen for the first time — the only count that should page anyone.</param>
    /// <param name="Ongoing">Incidents already known and still running.</param>
    /// <param name="Resolved">Incidents that closed this cycle.</param>
    /// <param name="BlindMetrics">
    /// Metrics that this deployment has a query for and which <b>no pod</b> reported.
    ///
    /// <para><b>The number that stops silence from meaning two things.</b> A metric the application does not
    /// export produces no findings, which is indistinguishable from health at every layer below this one. A
    /// non-zero count here means the guard is partly blind, and an operator who is not told that will read
    /// "no incidents" as "nothing is wrong".</para>
    /// </param>
    /// <param name="PartialMetrics">
    /// Metrics some but not all pods reported. Usually legitimate — CFS throttling counters exist only on
    /// containers carrying a CPU limit — but the same shape appears when a rollout has changed what half the
    /// fleet exports, so it is counted rather than assumed.
    /// </param>
    /// <param name="UnevaluableMetrics">
    /// Metrics that pods <b>did</b> report and which still produced no peer verdict — too few members cleared
    /// the sample floor for a comparison to mean anything.
    ///
    /// <para><b>The third way silence happens, and the one that was invisible.</b>
    /// <paramref name="BlindMetrics"/> answers "did anybody report this" and
    /// <paramref name="PartialMetrics"/> answers "did everybody". Neither answers "could it be evaluated", and
    /// on the cluster lab that gap swallowed nine of eleven metrics at once: every one of them was reported by
    /// every pod, every one returned <c>InsufficientData</c>, the degraded replica went undetected, and the
    /// cycle reported <c>blind = 0</c> — indistinguishable from a healthy cluster at every layer above.</para>
    /// </param>
    public readonly record struct GuardCycleResult(
        int Findings,
        int Incidents,
        int Opened,
        int Ongoing,
        int Resolved,
        int BlindMetrics,
        int PartialMetrics,
        int UnevaluableMetrics = 0)
    {
        /// <summary>Whether anything happened that a human has not already been told about.</summary>
        public bool HasNews => Opened > 0 || Resolved > 0;

        /// <summary>
        /// Whether the guard's own coverage is degraded, regardless of what it detected — either nobody
        /// reported a metric, or too few pods reported enough of it to compare.
        /// </summary>
        public bool IsPartiallyBlind => BlindMetrics > 0 || UnevaluableMetrics > 0;
    }
}
