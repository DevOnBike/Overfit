// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The explanation an operator reads. Asserted on content rather than on formatting: the wording is
    /// expected to change, the facts it must carry are not.
    /// </summary>
    public sealed class IncidentNarrativeTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 7, 31, 12, 4, 16, TimeSpan.Zero);

        [Fact]
        public void NamesEveryIdentifierNeededToOpenTheObject()
        {
            var narrative = IncidentNarrative.Describe(Incident(
                new IncidentSubject("overfit", "overfit-server", "5bc8b85574",
                    "overfit-server-degraded-5bc8b85574-vs64g", "docker-desktop")));

            // Each of these answers a different question, which is why none of them is redundant: the
            // namespace picks the context, the workload is where a human reasons, the ReplicaSet separates
            // this rollout's pods from the last one's, and the node turns "some replicas" into "one node".
            Assert.Contains("overfit", narrative, StringComparison.Ordinal);
            Assert.Contains("overfit-server", narrative, StringComparison.Ordinal);
            Assert.Contains("5bc8b85574", narrative, StringComparison.Ordinal);
            Assert.Contains("overfit-server-degraded-5bc8b85574-vs64g", narrative, StringComparison.Ordinal);
            Assert.Contains("docker-desktop", narrative, StringComparison.Ordinal);
        }

        [Fact]
        public void StatesTheIntervalTheBehaviourWasObservedOver()
        {
            var narrative = IncidentNarrative.Describe(Incident(Pod()));

            // Absolute and UTC: this gets pasted into a dashboard query or a kubectl --since far more often
            // than it is read as prose, and "15 minutes ago" is wrong the moment it is stored.
            Assert.Contains("2026-07-31 12:04:16Z", narrative, StringComparison.Ordinal);
            Assert.Contains("15 min observed", narrative, StringComparison.Ordinal);
        }

        [Fact]
        public void CarriesTheDetectorsOwnReasoning()
        {
            var narrative = IncidentNarrative.Describe(Incident(Pod()));

            Assert.Contains("CpuUsageRatio", narrative, StringComparison.Ordinal);
            Assert.Contains("sits below the other 3 peers", narrative, StringComparison.Ordinal);
        }

        /// <summary>
        /// An empty pod is a statement, not a gap — common-mode findings are about the deployment, and
        /// naming a replica would be a false claim. The narrative has to say so, or the reader fills the
        /// silence with a guess.
        /// </summary>
        [Fact]
        public void AWorkloadLevelIncidentSaysNoReplicaIsResponsible()
        {
            var narrative = IncidentNarrative.Describe(Incident(
                new IncidentSubject("overfit", "overfit-server", string.Empty, string.Empty, string.Empty)));

            Assert.Contains("workload as a whole", narrative, StringComparison.Ordinal);
            Assert.DoesNotContain("pod ", narrative, StringComparison.Ordinal);
        }

        /// <summary>
        /// The caveat that stops a relative finding from being over-read. Without it "this replica is worse"
        /// looks like a measurement rather than one of two readings of the same evidence.
        /// </summary>
        [Fact]
        public void ASinglePodIncidentWarnsThatAttributionIsAmbiguous()
        {
            var narrative = IncidentNarrative.Describe(Incident(Pod()));

            Assert.Contains("CAVEAT", narrative, StringComparison.Ordinal);
            Assert.Contains("siblings got better", narrative, StringComparison.Ordinal);
        }

        [Fact]
        public void ASymptomSaysItIsNotACause()
        {
            var narrative = IncidentNarrative.Describe(Incident(Pod(), SignalClass.Symptom));

            Assert.Contains("says something hurts, not why", narrative, StringComparison.Ordinal);
        }

        [Fact]
        public void RejectsNull()
        {
            Assert.Throws<ArgumentNullException>(() => IncidentNarrative.Describe(null!));
        }

        private static IncidentSubject Pod()
            => new("overfit", "overfit-server", "5bc8b85574",
                "overfit-server-degraded-5bc8b85574-vs64g", "docker-desktop");

        private static Incident Incident(IncidentSubject subject, SignalClass signalClass = SignalClass.Resource)
        {
            var pipeline = new IncidentPipeline();

            pipeline.ObserveRule(
                subject,
                "CpuUsageRatio",
                new SustainedThresholdResult(
                    Status: DetectionStatus.Anomalous,
                    Reason: "'overfit-server-degraded-5bc8b85574-vs64g' sits below the other 3 peers "
                            + "beyond noise (delta 1.00).",
                    BreachFraction: 0.95,
                    BreachedSamples: 58,
                    UsableSamples: 61,
                    PeakValue: 1.0,
                    MedianValue: 0.98),
                T0,
                T0.AddMinutes(15),
                default,
                signalClass);

            var grouped = pipeline.Group(IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.SingleNode
            });

            Assert.Single(grouped);

            return grouped[0];
        }
    }
}
