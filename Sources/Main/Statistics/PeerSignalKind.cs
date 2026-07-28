// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Whether a signal's magnitude can be explained by how much work its owner happened to receive. This is a
    /// precondition on peer comparison, not a hint: "eleven replicas alike, one different" silently assumes
    /// even load balancing, and sticky sessions, uneven sharding, a hot tenant or keep-alive skew all break
    /// that assumption — at which point the detector fires on <i>correct</i> behaviour.
    ///
    /// <para><b>Normalising by work is necessary and NOT sufficient — measured across three traffic skews,
    /// not assumed.</b> Three identical replicas in the local Kubernetes lab, one deliberately given a
    /// multiple of the others' traffic, ~600 requests per run:</para>
    /// <list type="table">
    /// <item><term>skew 2.3x</term><description>work 87.9 % · raw CPU 57.5 % · <b>cost/request 26.2 %</b></description></item>
    /// <item><term>skew 4.4x</term><description>work 161.6 % · raw CPU 116.8 % · <b>cost/request 44.3 %</b></description></item>
    /// <item><term>skew 8.4x</term><description>work 223.7 % · raw CPU 185.1 % · <b>cost/request 51.6 %</b></description></item>
    /// </list>
    ///
    /// <para>Dividing by the work metric removes most of the apparent difference and leaves a residue that
    /// <b>grows with the imbalance and never vanishes</b> — and the busiest replica always comes out with the
    /// <i>lowest</i> cost per request, not the highest. The per-replica figures show the mechanism plainly:
    /// the two lightly-loaded replicas sit at ~6.6–7.6 CPU-seconds per request in every run, while the
    /// busy one falls 5.52 → 4.74 → 4.01 as its share rises. That is fixed per-process overhead (background
    /// threads, GC, idle polling) amortised over more requests: a replica is cheaper per request precisely
    /// because it is busier.</para>
    ///
    /// <para>The shape fits an affine cost model, <c>cost ≈ fixed + marginal × work</c>, whose residue scales
    /// as <c>(1 − 1/skew)</c> and therefore saturates. Predicted relative residue across the three points,
    /// normalised to the largest skew: 0.571 / 0.857 / 1.0. Measured: 0.508 / 0.859 / 1.0 — within 0.2 % at
    /// the middle point, about 11 % off at the smallest. Consistent with the model; not a fit of it.</para>
    ///
    /// <para>So a peer comparison on unit cost under uneven load will still flag a difference — it will
    /// simply point at the wrong member. Comparing the <i>marginal</i> term is what "cost per request" was
    /// meant to express, and plain division conflates it with the fixed term. Until an affine fit exists,
    /// treat <see cref="LoadSensitive"/> as "the work metric is required before comparing", not as
    /// "dividing by it makes peers comparable".</para>
    ///
    /// <para><i>Caveat carried deliberately: one run per skew, 180 s each, two-minute rate windows, on a
    /// single-node lab where all three replicas share a host. The direction, the monotonic growth and the
    /// inverted ranking are solid; the absolute residue is not, and a single global (fixed, marginal) pair
    /// does not reproduce all three runs.</i></para>
    /// </summary>
    public enum PeerSignalKind
    {
        /// <summary>
        /// Uneven load cannot explain a deviation: restart count, OOMKilled, readiness, container terminations,
        /// node conditions. Comparable across peers as-is, and — usefully — detectable with no application
        /// instrumentation at all, straight from cAdvisor and kube-state-metrics.
        /// </summary>
        LoadIndependent = 0,

        /// <summary>
        /// Magnitude tracks throughput: memory, CPU, network bytes, connection counts. Requires
        /// <see cref="PeerSeries.Work"/> so the comparison runs on cost <i>per unit of work</i>; without it the
        /// detector reports <see cref="DetectionStatus.InsufficientData"/> rather than guessing.
        ///
        /// <para>Supplying the work metric is the entry requirement, not a guarantee of comparability — see
        /// the measured residue on the enum above. Under materially uneven traffic, prefer a
        /// <see cref="LoadIndependent"/> signal (restarts, OOMKilled, readiness) for the verdict and keep unit
        /// cost as supporting evidence.</para>
        /// </summary>
        LoadSensitive = 1,
    }
}
