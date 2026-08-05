// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// Splits a deployment's pods into the groups the operator declared they may be compared within.
    ///
    /// <para><b>Declared, because three different situations look identical and want opposite answers.</b> A
    /// rolling update, a canary and an elected leader all present as a minority of replicas behaving unlike
    /// the majority. A rollout should not be compared across, because the new cohort is starting from a cold
    /// working set. A canary should be, because comparing it against the baseline is the entire reason it
    /// exists. A leader should not be, because it legitimately does different work. No property of the
    /// metrics separates them.</para>
    ///
    /// <para><b>An attempt to infer this from the ReplicaSet was written and reverted within the hour.</b> It
    /// looked right for the rollout case and a test immediately showed what it cost: a canary is its own
    /// ReplicaSet, so splitting on it left the canary alone in a cohort below the minimum group size and made
    /// it invisible — turning off the comparison that was the point of deploying it. Autoscaling does not fit
    /// either, since pods an HPA adds join the <i>same</i> ReplicaSet as their older siblings.</para>
    ///
    /// <para><b>What makes the declaration practical is that the cluster already carries it.</b> The group
    /// comes from a pod label the operator names, read from kube-state-metrics — Patroni publishes
    /// <c>role</c>, and most database and queue operators do something equivalent. It updates itself on
    /// failover, which a hand-written list cannot, and a change in it is worth noticing in its own right.</para>
    ///
    /// <para><b>The cost, stated plainly.</b> A cohort is compared only against itself, so a group that is
    /// uniformly bad has no outlier within it and the peer family cannot see the problem. That is the masking
    /// bound, and it is accepted here on purpose: catching a whole cohort regressing together is the trend
    /// family's job. A cohort too small to compare returns <see cref="Statistics.DetectionStatus.InsufficientData"/>,
    /// which is counted rather than passed off as quiet.</para>
    ///
    /// <para>With nothing declared every pod carries an empty group, they all land together, and this changes
    /// nothing — which is what makes it safe to have on by default.</para>
    /// </summary>
    public static class PeerCohorts
    {
        /// <summary>
        /// Groups pod indices by declared peer group, preserving order within each group.
        /// </summary>
        /// <param name="groups">One key per pod, index-aligned with the window; empty means undeclared.</param>
        /// <param name="count">Pods to consider.</param>
        /// <returns>
        /// One list of indices per distinct key. A single list means nothing was declared, or everything was
        /// declared the same — the ordinary case, and the one that costs nothing.
        /// </returns>
        public static List<List<int>> Partition(IReadOnlyList<string> groups, int count)
        {
            ArgumentNullException.ThrowIfNull(groups);

            var cohorts = new List<List<int>>(2);
            var keys = new List<string>(2);

            for (var pod = 0; pod < count; pod++)
            {
                var key = pod < groups.Count ? groups[pod] : string.Empty;
                var found = -1;

                for (var i = 0; i < keys.Count; i++)
                {
                    if (string.Equals(keys[i], key, StringComparison.Ordinal))
                    {
                        found = i;

                        break;
                    }
                }

                if (found < 0)
                {
                    keys.Add(key);
                    cohorts.Add(new List<int>(count));
                    found = cohorts.Count - 1;
                }

                cohorts[found].Add(pod);
            }

            return cohorts;
        }
    }
}
