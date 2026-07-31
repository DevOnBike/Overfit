// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies.Incidents
{
    public sealed class IncidentGrouperTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 7, 28, 12, 0, 0, TimeSpan.Zero);

        private static readonly IncidentGrouper Grouper = new();

        [Fact]
        public void NoFindings_ProduceNoIncidents()
        {
            Assert.Empty(Grouper.Group([], IncidentGroupingOptions.Balanced));
        }

        [Fact]
        public void ALoneFinding_SurvivesAsAnIncidentOfOne()
        {
            var findings = new[] { Finding("pod-a", "latency", SignalClass.Symptom, 0, 5) };

            var incidents = Grouper.Group(findings, IncidentGroupingOptions.Balanced);

            var incident = Assert.Single(incidents);
            Assert.Single(incident.Findings);
            Assert.Equal("latency", incident.Primary.Signal);
        }

        [Fact]
        public void SamePodOverlappingInTime_BecomesOneIncident()
        {
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10),
                Finding("pod-a", "cpu_throttling", SignalClass.Infrastructure, 1, 9)
            };

            var incidents = Grouper.Group(findings, IncidentGroupingOptions.Balanced);

            var incident = Assert.Single(incidents);
            Assert.Equal(2, incident.Findings.Count);
            Assert.Equal(1, incident.AffectedSubjects);
            Assert.Equal(2, incident.DistinctSignals);
        }

        [Fact]
        public void InfrastructureOutranksSymptom_EvenWhenItStartedLater()
        {
            // The lab's actual shape: p95 is what gets noticed, throttling is what an engineer can act on.
            var findings = new[]
            {
                Finding("pod-a", "response_time_p95", SignalClass.Symptom, 0, 10),
                Finding("pod-a", "cpu_throttling", SignalClass.Infrastructure, 4, 10)
            };

            var incident = Assert.Single(Grouper.Group(findings, IncidentGroupingOptions.Balanced));

            Assert.Equal("cpu_throttling", incident.Primary.Signal);
            Assert.Equal("cpu_throttling", incident.Findings[0].Signal);
        }

        [Fact]
        public void UnrelatedWorkloadsFarApart_StaySeparate()
        {
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 5, workload: "api", ns: "shop", node: "n1"),
                Finding("pod-z", "latency", SignalClass.Symptom, 600, 620, workload: "billing", ns: "finance", node: "n2")
            };

            Assert.Equal(2, Grouper.Group(findings, IncidentGroupingOptions.Balanced).Count);
        }

        [Fact]
        public void SimultaneousButTopologicallyUnrelated_StaySeparate()
        {
            // Same instant, nothing else in common: coincidence, not an incident.
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10, workload: "api", ns: "shop", node: "n1"),
                Finding("pod-z", "latency", SignalClass.Symptom, 0, 10, workload: "billing", ns: "finance", node: "n2")
            };

            Assert.Equal(2, Grouper.Group(findings, IncidentGroupingOptions.Balanced).Count);
        }

        [Fact]
        public void SameNode_LinksDifferentWorkloads()
        {
            // The fault nobody looks for: two unrelated workloads degrading together because the node is.
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10, workload: "api", ns: "shop", node: "n7"),
                Finding("pod-z", "latency", SignalClass.Symptom, 1, 11, workload: "billing", ns: "finance", node: "n7")
            };

            var incident = Assert.Single(Grouper.Group(findings, IncidentGroupingOptions.Balanced));

            Assert.Equal(2, incident.AffectedSubjects);
        }

        [Fact]
        public void OnASingleNodeCluster_TheNodeCoordinateCanBeSwitchedOff()
        {
            // Why these weights had to leave the detector as private constants. Docker Desktop, kind and
            // minikube put every pod on one node, so "same node" is a constant there — and a constant is not
            // evidence. Left at its default it relates every finding to every other; zeroed, the pair falls
            // back to the namespace link, which is deliberately too weak to merge on its own.
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10, workload: "api", ns: "shop", node: "n1"),
                Finding("pod-z", "latency", SignalClass.Symptom, 1, 11, workload: "billing", ns: "shop", node: "n1")
            };

            Assert.Single(Grouper.Group(findings, IncidentGroupingOptions.Balanced));

            var singleNode = IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.SingleNode
            };

            Assert.Equal(2, Grouper.Group(findings, singleNode).Count);
        }

        [Fact]
        public void RaisingTheNamespaceWeight_MergesWhatItOtherwiseCouldNot()
        {
            // The opposite environment: on a single-tenant cluster a shared namespace is nearly as strong as a
            // shared workload, and an operator has to be able to say so.
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10, workload: "api", ns: "shop", node: "n1"),
                Finding("pod-z", "errors", SignalClass.Symptom, 0, 10, workload: "billing", ns: "shop", node: "n2")
            };

            Assert.Equal(2, Grouper.Group(findings, IncidentGroupingOptions.Balanced).Count);

            var singleTenant = IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.Default with { SameNamespace = 0.65 }
            };

            Assert.Single(Grouper.Group(findings, singleTenant));
        }

        [Fact]
        public void DuringARollout_TheTwoHalvesCanBeKeptApart()
        {
            // The coordinate a rollout turns on. Two findings on the same Deployment but opposite sides of the
            // ReplicaSet split are about different programs, and an operator has to be able to say so — that is
            // exactly what was impossible while the weights were private constants and the field did not exist.
            var oldVersion = Finding("pod-old", "latency", SignalClass.Symptom, 0, 10,
                workload: "api", ns: "shop", node: "n1", replicaSet: "api-6d4b7c9f8x");
            var newVersion = Finding("pod-new", "latency", SignalClass.Symptom, 1, 11,
                workload: "api", ns: "shop", node: "n1", replicaSet: "api-7f9c2a1b4y");

            var findings = new[] { oldVersion, newVersion };

            // By default the shared workload still merges them: mid-rollout that is a defensible reading.
            Assert.Single(Grouper.Group(findings, IncidentGroupingOptions.Balanced));

            // Dropping the workload weight below the threshold separates the versions while leaving same-version
            // pods merged, which is the distinction the ReplicaSet coordinate exists to make.
            var versionAware = IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.Default with { SameWorkload = 0.2 }
            };

            Assert.Equal(2, Grouper.Group(findings, versionAware).Count);
        }

        [Fact]
        public void SameReplicaSet_OutranksSameWorkload()
        {
            // Two pods of one ReplicaSet are the same software; two pods of one workload during a rollout are
            // not. The weight ordering has to reflect that or the field buys nothing.
            var sameRs = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10,
                    workload: "api", ns: "shop", node: "n1", replicaSet: "api-6d4b7c9f8x"),
                Finding("pod-b", "latency", SignalClass.Symptom, 1, 11,
                    workload: "api", ns: "shop", node: "n1", replicaSet: "api-6d4b7c9f8x")
            };

            var crossRs = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10,
                    workload: "api", ns: "shop", node: "n1", replicaSet: "api-6d4b7c9f8x"),
                Finding("pod-b", "latency", SignalClass.Symptom, 1, 11,
                    workload: "api", ns: "shop", node: "n1", replicaSet: "api-7f9c2a1b4y")
            };

            // A threshold between the two weights: same-ReplicaSet clears it, cross-ReplicaSet does not.
            var between = IncidentGroupingOptions.Balanced with { MinRelatedness = 0.75 };

            Assert.Single(Grouper.Group(sameRs, between));
            Assert.Equal(2, Grouper.Group(crossRs, between).Count);

            Assert.True(TopologyWeights.Default.SameReplicaSet > TopologyWeights.Default.SameWorkload);
        }

        [Fact]
        public void InvalidWeights_AreRejectedRatherThanClamped()
        {
            var findings = new[] { Finding("pod-a", "latency", SignalClass.Symptom, 0, 5) };

            var negative = IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.Default with { SameNode = -0.1 }
            };

            var aboveOne = IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.Default with { SameWorkload = 1.5 }
            };

            // Zeroing the strongest link would mean "two findings about the same process are unrelated", which
            // is not a weakening of the evidence but a mistake.
            var podless = IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.Default with { SamePod = 0.0 }
            };

            Assert.Throws<ArgumentException>(() => Grouper.Group(findings, negative));
            Assert.Throws<ArgumentException>(() => Grouper.Group(findings, aboveOne));
            Assert.Throws<ArgumentException>(() => Grouper.Group(findings, podless));
        }

        [Fact]
        public void SameNamespaceAlone_IsNotEnoughToMerge()
        {
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10, workload: "api", ns: "shop", node: "n1"),
                Finding("pod-z", "errors", SignalClass.Symptom, 0, 10, workload: "billing", ns: "shop", node: "n2")
            };

            Assert.Equal(2, Grouper.Group(findings, IncidentGroupingOptions.Balanced).Count);
        }

        [Fact]
        public void CorrelatedSeries_LinkWhatTopologyCannot()
        {
            // Different workloads, different nodes, same namespace — topology scores 0.25 and would not
            // merge. The series moving together is the evidence that lifts it.
            var leader = new double[60];
            var follower = new double[60];

            for (var i = 0; i < leader.Length; i++)
            {
                leader[i] = Math.Sin(i * 0.2) * 100.0;
                follower[i] = (Math.Sin(i * 0.2) * 40.0) + 5.0;
            }

            var findings = new[]
            {
                Finding("pod-a", "queue_depth", SignalClass.Resource, 0, 10, workload: "api", ns: "shop", node: "n1")
                    with { Series = leader },
                Finding("pod-z", "queue_depth", SignalClass.Resource, 0, 10, workload: "billing", ns: "shop", node: "n2")
                    with { Series = follower }
            };

            var incident = Assert.Single(Grouper.Group(findings, IncidentGroupingOptions.Balanced));

            Assert.Equal(2, incident.AffectedSubjects);
        }

        [Fact]
        public void Strict_DisablesCorrelationLinking()
        {
            var leader = new double[60];
            var follower = new double[60];

            for (var i = 0; i < leader.Length; i++)
            {
                leader[i] = Math.Sin(i * 0.2) * 100.0;
                follower[i] = (Math.Sin(i * 0.2) * 40.0) + 5.0;
            }

            var findings = new[]
            {
                Finding("pod-a", "queue_depth", SignalClass.Resource, 0, 10, workload: "api", ns: "shop", node: "n1")
                    with { Series = leader },
                Finding("pod-z", "queue_depth", SignalClass.Resource, 0, 10, workload: "billing", ns: "shop", node: "n2")
                    with { Series = follower }
            };

            Assert.Equal(2, Grouper.Group(findings, IncidentGroupingOptions.Strict).Count);
        }

        [Fact]
        public void TheSpanBound_StopsATransitiveWalkAcrossTheDay()
        {
            // A chain of pairwise-adjacent findings on one pod, each overlapping its neighbour. Connected
            // components would make this one incident spanning six hours; the span bound must not.
            var findings = new List<SignalFinding>();

            for (var i = 0; i < 40; i++)
            {
                var start = i * 9;
                findings.Add(Finding("pod-a", $"signal_{i}", SignalClass.Resource, start, start + 10));
            }

            var incidents = Grouper.Group(
                findings.ToArray(),
                IncidentGroupingOptions.Balanced with
                {
                    MaxIncidentSpan = TimeSpan.FromMinutes(60)
                });

            Assert.True(incidents.Count > 1, "the chain collapsed into a single unbounded incident");

            foreach (var incident in incidents)
            {
                Assert.True(
                    incident.Duration <= TimeSpan.FromMinutes(60),
                    $"incident spans {incident.Duration}, past the stated bound");
            }
        }

        [Fact]
        public void GroupingDoesNotDependOnInputOrder()
        {
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10, severity: 0.4),
                Finding("pod-a", "cpu_throttling", SignalClass.Infrastructure, 2, 10, severity: 0.9),
                Finding("pod-b", "restarts", SignalClass.Infrastructure, 3, 8, severity: 0.6),
                Finding("pod-z", "latency", SignalClass.Symptom, 900, 910, severity: 0.2,
                        workload: "other", ns: "far", node: "n9")
            };

            var forward = Grouper.Group(findings, IncidentGroupingOptions.Balanced);

            Array.Reverse(findings);
            var reversed = Grouper.Group(findings, IncidentGroupingOptions.Balanced);

            Assert.Equal(forward.Count, reversed.Count);

            for (var i = 0; i < forward.Count; i++)
            {
                Assert.Equal(forward[i].Findings.Count, reversed[i].Findings.Count);
                Assert.Equal(forward[i].Primary.Signal, reversed[i].Primary.Signal);
            }
        }

        [Fact]
        public void IncidentsAreOrderedMostSevereFirst()
        {
            var findings = new[]
            {
                Finding("pod-a", "latency", SignalClass.Symptom, 0, 10, severity: 0.3),
                Finding("pod-z", "oom", SignalClass.Infrastructure, 0, 10, severity: 0.95,
                        workload: "other", ns: "far", node: "n9")
            };

            var incidents = Grouper.Group(findings, IncidentGroupingOptions.Balanced);

            Assert.Equal(2, incidents.Count);
            Assert.Equal("oom", incidents[0].Primary.Signal);
            Assert.Equal(0.95, incidents[0].PeakSeverity, 6);
        }

        [Fact]
        public void SummaryNamesThePrimaryAndTheBreadth()
        {
            var findings = new[]
            {
                Finding("pod-a", "response_time_p95", SignalClass.Symptom, 0, 10),
                Finding("pod-a", "cpu_throttling", SignalClass.Infrastructure, 1, 10)
            };

            var incident = Assert.Single(Grouper.Group(findings, IncidentGroupingOptions.Balanced));

            Assert.Contains("cpu_throttling", incident.Summary, StringComparison.Ordinal);
            Assert.Contains("pod-a", incident.Summary, StringComparison.Ordinal);
            Assert.Contains("+1 related finding", incident.Summary, StringComparison.Ordinal);
        }

        [Fact]
        public void DefaultOptions_AreRejected()
        {
            var findings = new[] { Finding("pod-a", "latency", SignalClass.Symptom, 0, 5) };

            Assert.Throws<ArgumentException>(() => Grouper.Group(findings, default));
        }

        [Fact]
        public void OversizedBatch_IsRejectedRatherThanScored()
        {
            var findings = new SignalFinding[IncidentGrouper.MaxFindingsPerCall + 1];

            for (var i = 0; i < findings.Length; i++)
            {
                findings[i] = Finding($"pod-{i}", "latency", SignalClass.Symptom, 0, 5);
            }

            Assert.Throws<ArgumentException>(() => Grouper.Group(findings, IncidentGroupingOptions.Balanced));
        }

        private static SignalFinding Finding(
            string pod,
            string signal,
            SignalClass signalClass,
            int startMinutes,
            int endMinutes,
            double severity = 0.5,
            string workload = "overfit-server",
            string ns = "overfit",
            string node = "node-1",
            string replicaSet = "")
        {
            return new SignalFinding(
                new IncidentSubject(ns, workload, replicaSet, pod, node),
                signal,
                signalClass,
                T0.AddMinutes(startMinutes),
                T0.AddMinutes(endMinutes),
                severity,
                $"{signal} deviated on {pod}");
        }
    }
}
