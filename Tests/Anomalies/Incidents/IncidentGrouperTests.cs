// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;

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
                IncidentGroupingOptions.Balanced with { MaxIncidentSpan = TimeSpan.FromMinutes(60) });

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
            string node = "node-1")
        {
            return new SignalFinding(
                new IncidentSubject(ns, workload, pod, node),
                signal,
                signalClass,
                T0.AddMinutes(startMinutes),
                T0.AddMinutes(endMinutes),
                severity,
                $"{signal} deviated on {pod}");
        }
    }
}
