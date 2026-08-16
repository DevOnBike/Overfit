// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Runtime.ExceptionServices;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Rules;
using DevOnBike.Overfit.Anomalies.Training;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The guard's reason and incident strings must be the same bytes on every machine that runs the same
    /// build.
    ///
    /// <para><b>This exists because the defect is invisible from one box (<c>XC-64</c>, 2026-08-16).</b> CI
    /// failed one test of 2931: <c>LevelShiftDetector</c> interpolated <c>{options.MinRelativeChange:P0}</c>,
    /// which the invariant culture a Linux runner gets from <c>LANG=C.UTF-8</c> renders <c>25 %</c> and the
    /// Windows dev box renders <c>25%</c>. Every test written under one culture agrees with itself, so the
    /// only thing that can see this is a test that runs the same code under two.</para>
    ///
    /// <para><b>The oracle is the same producer run under two cultures, not a literal.</b> Comparing against
    /// hard-coded text would pin whichever culture the author had; comparing invariant output against hostile
    /// output pins the property the operator depends on — that the text does not move — and covers every hole
    /// in the string, including the ones with no format specifier. The capability assertions at the top are
    /// what stop it passing vacuously: a culture that happens to format numbers the invariant way would make
    /// every comparison below trivially true.</para>
    ///
    /// <para><b>Why a dedicated thread rather than the exclusive collection.</b>
    /// <see cref="CultureInfo.CurrentCulture"/> is per thread, so a culture set on a thread this test owns
    /// cannot reach a test running in parallel on another one — which is stronger than setting it on the
    /// xunit worker and restoring in a <c>finally</c>, and cheaper than serialising the whole suite through
    /// <c>ExclusiveProcessMeasurementCollection</c>.</para>
    /// </summary>
    public sealed class DiagnosticTextIsCultureInvariantTests
    {
        /// <summary>
        /// Two ways the same defect presents, and one culture does not cover both: <c>pl-PL</c> moves the
        /// decimal separator to a comma and separates the percent sign with a non-breaking space, while
        /// <c>tr-TR</c> puts the percent sign in front of the number entirely.
        /// </summary>
        [Theory]
        [InlineData("pl-PL")]
        [InlineData("tr-TR")]
        public void EveryDiagnosticStringIsTheSameBytesUnderAnyCulture(string cultureName)
        {
            var hostile = CultureInfo.GetCultureInfo(cultureName);

            Assert.NotEqual(
                0.25.ToString("P0", CultureInfo.InvariantCulture),
                0.25.ToString("P0", hostile));
            Assert.NotEqual(
                1.5.ToString("G4", CultureInfo.InvariantCulture),
                1.5.ToString("G4", hostile));

            foreach (var (name, produce) in Producers())
            {
                var invariant = Under(CultureInfo.InvariantCulture, produce);
                var underHostile = Under(hostile, produce);

                Assert.False(string.IsNullOrEmpty(invariant), name + " produced nothing to compare");

                // The name rides along in the compared value so a failure says which string moved.
                Assert.Equal(name + " => " + invariant, name + " => " + underHostile);
            }
        }

        /// <summary>
        /// The literal an operator greps for, and the one CI asserts on. Pinned separately because the
        /// comparison above would be satisfied by two identically-wrong strings.
        /// </summary>
        [Fact]
        public void PercentagesReadTheWayTheInvariantCultureRendersThem()
        {
            var reason = Under(CultureInfo.GetCultureInfo("pl-PL"), TinyLevelShiftReason);

            Assert.Contains("below the 25 %", reason, StringComparison.Ordinal);
        }

        private static (string Name, Func<string> Produce)[] Producers()
        {
            return new (string, Func<string>)[]
            {
                ("level-shift/too-small", TinyLevelShiftReason),
                ("level-shift/step", StepReason),
                ("level-shift/absolute-floor", AbsoluteFloorReason),
                ("level-shift/insufficient", InsufficientSamplesReason),
                ("sustained/under-fraction", SustainedUnderFractionReason),
                ("sustained/held", SustainedHeldReason),
                ("trend/rising-towards-a-limit", TrendReason),
                ("peer/no-coherent-norm", PeerWithoutACentreReason),
                ("incident/peer-narrative", PeerIncidentNarrative),
                // Three durations, because the narrative humanises the observed span in three different
                // units and only the hours branch carries a decimal — "3.5 h" against "3,5 h".
                ("incident/narrative-seconds", () => PeerIncidentNarrative(TimeSpan.FromSeconds(45))),
                ("incident/narrative-hours", () => PeerIncidentNarrative(TimeSpan.FromMinutes(210))),
                ("gpt/anomaly-score", AnomalyScoreText),
                ("training/progress", TrainingProgressText),
                ("training/result", TrainingResultText)
            };
        }

        private static string TinyLevelShiftReason()
        {
            return new LevelShiftDetector()
                .Detect(Step(from: 100.0, to: 102.0, at: 40), LevelShiftOptions.Balanced)
                .Reason;
        }

        private static string StepReason()
        {
            return new LevelShiftDetector()
                .Detect(Step(from: 100.0, to: 250.0, at: 40), LevelShiftOptions.Balanced)
                .Reason;
        }

        private static string AbsoluteFloorReason()
        {
            var options = LevelShiftOptions.Balanced with
            {
                MinAbsoluteChange = 1000.0
            };

            return new LevelShiftDetector().Detect(Step(from: 100.0, to: 250.0, at: 40), options).Reason;
        }

        private static string InsufficientSamplesReason()
        {
            return new LevelShiftDetector().Detect(new double[] { 1, 2, 3, 4 }, LevelShiftOptions.Balanced).Reason;
        }

        private static string SustainedUnderFractionReason()
        {
            var window = new double[40];

            for (var i = 0; i < 5; i++)
            {
                window[i * 3] = 0.4;
            }

            return new SustainedThresholdRule()
                .Evaluate(window, SustainedThresholdOptions.ForCpuThrottling)
                .Reason;
        }

        private static string SustainedHeldReason()
        {
            var window = new double[40];

            Array.Fill(window, 0.4125);

            return new SustainedThresholdRule()
                .Evaluate(window, SustainedThresholdOptions.ForCpuThrottling)
                .Reason;
        }

        private static string TrendReason()
        {
            var values = new double[240];
            var seconds = new double[240];
            var rng = new Random(20260816);

            for (var i = 0; i < values.Length; i++)
            {
                seconds[i] = i * 30.0;
                values[i] = 1000.0 + (i * 2.0) + (rng.NextDouble() * 2.0);
            }

            return new TrendDetector().Detect(values, seconds, TrendOptions.Balanced, limit: 3000.0).Reason;
        }

        private static string PeerWithoutACentreReason()
        {
            var peers = new List<PeerSeries>
            {
                new("pod-a", Series(seed: 1, level: 100.0)),
                new("pod-b", Series(seed: 2, level: 100.0)),
                new("pod-hot-a", Series(seed: 3, level: 160.0)),
                new("pod-hot-b", Series(seed: 4, level: 160.0)),
                new("pod-cold-a", Series(seed: 5, level: 55.0)),
                new("pod-cold-b", Series(seed: 6, level: 55.0))
            };
            var findings = new PeerOutlierFinding[peers.Count];
            var result = new PeerGroupOutlierDetector()
                .Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Contains("sit more than", result.Reason, StringComparison.Ordinal);

            return result.Reason;
        }

        /// <summary>
        /// The whole operator-facing path in one string: the peer detector's numbers, the pipeline's
        /// composed reason for a single member, and the narrative's duration.
        /// </summary>
        private static string PeerIncidentNarrative()
        {
            return PeerIncidentNarrative(TimeSpan.FromMinutes(15));
        }

        private static string PeerIncidentNarrative(TimeSpan observed)
        {
            var peers = new List<PeerSeries>();

            for (var p = 0; p < 5; p++)
            {
                peers.Add(new PeerSeries($"pod-{p}", Series(seed: p, level: 100.0)));
            }

            peers.Add(new PeerSeries("pod-hot", Series(seed: 99, level: 300.0)));

            var findings = new PeerOutlierFinding[peers.Count];
            var result = new PeerGroupOutlierDetector()
                .Detect(peers, PeerSignalKind.LoadIndependent, PeerOutlierOptions.Balanced, findings);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);

            var subjects = new IncidentSubject[peers.Count];

            for (var p = 0; p < peers.Count; p++)
            {
                subjects[p] = new IncidentSubject("overfit", "overfit-server", "5bc8b85574", peers[p].Name, "node-1");
            }

            var start = new DateTimeOffset(2026, 8, 16, 9, 0, 0, TimeSpan.Zero);
            var pipeline = new IncidentPipeline();

            pipeline.ObservePeerGroup("CpuUsageRatio", result, findings, subjects, start, start + observed);

            var grouped = pipeline.Group(IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.SingleNode
            });

            Assert.NotEmpty(grouped);

            return IncidentNarrative.Describe(grouped[0]);
        }

        private static string AnomalyScoreText()
        {
            return new AnomalyScore
            {
                PodName = "overfit-server-5bc8b85574-vs64g",
                Score = 3.14159f,
                WorstMetric = "CpuUsageRatio",
                ExpectedValue = 0.25f,
                ActualValue = 1.75f
            }.ToString();
        }

        private static string TrainingProgressText()
        {
            return new TrainingProgress
            {
                Phase = "train",
                Step = 1200,
                TotalSteps = 4000,
                TrainLoss = 1.2345f,
                ValLoss = 0.9876f,
                Elapsed = TimeSpan.FromSeconds(125)
            }.ToString();
        }

        private static string TrainingResultText()
        {
            return new OfflineTrainingResult
            {
                SnapshotsLoaded = 1234567,
                FinalValLoss = 0.4321f,
                TrainingTime = TimeSpan.FromSeconds(310),
                CheckpointPath = "c:/checkpoints/guard.bin"
            }.ToString();
        }

        /// <summary>
        /// Runs <paramref name="produce"/> on a thread this test owns, so the culture it sets is visible to
        /// nothing else in the suite.
        /// </summary>
        private static string Under(CultureInfo culture, Func<string> produce)
        {
            var produced = string.Empty;
            ExceptionDispatchInfo? failure = null;
            var thread = new Thread(() =>
            {
                try
                {
                    CultureInfo.CurrentCulture = culture;
                    CultureInfo.CurrentUICulture = culture;
                    produced = produce();
                }
                catch (Exception ex)
                {
                    failure = ExceptionDispatchInfo.Capture(ex);
                }
            });

            thread.Start();
            thread.Join();
            failure?.Throw();

            return produced;
        }

        /// <summary>Tight noise around each level, so the halves separate on the level and not on the noise.</summary>
        private static double[] Step(double from, double to, int at, int samples = 80)
        {
            var rng = new Random(20260801);
            var series = new double[samples];

            for (var i = 0; i < samples; i++)
            {
                var level = i < at ? from : to;

                series[i] = level * (0.99 + (0.02 * rng.NextDouble()));
            }

            return series;
        }

        private static double[] Series(int seed, double level, int samples = 60)
        {
            var rng = new Random(seed);
            var values = new double[samples];

            for (var i = 0; i < samples; i++)
            {
                values[i] = level * (0.98 + (0.04 * rng.NextDouble()));
            }

            return values;
        }
    }
}
