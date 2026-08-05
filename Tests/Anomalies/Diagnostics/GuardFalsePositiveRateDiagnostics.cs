// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// How many incidents <b>the guard that ships</b> invents on a synthetic cluster where nothing is wrong.
    ///
    /// <para><b>Why this exists next to <see cref="FalsePositiveRateDiagnostics"/> rather than replacing a
    /// number in it.</b> That diagnostic hand-drives <c>TrendDetector</c>, <c>PeerGroupOutlierDetector</c> and
    /// <c>IncidentPipeline</c> and never constructs an <see cref="AnomalyGuard"/> at all. Its 484 lines
    /// contain no <c>RunCycle</c>, no <c>ConfiguredFloorSource</c>, no <c>FloorCalibrator</c>, no
    /// common-mode decomposition, no level-shift gate and no threshold rules. So its headline — <b>250 false
    /// incidents a day on 20 healthy pods</b> — is a true statement about a subset of the detector stack and
    /// not about the product, and comparing it with the cluster lab's measured <b>5 a day</b> compares two
    /// different programs. That 50x gap was on the open list as "the simulator and the lab disagree, and
    /// until we know why we do not know which number to quote a client". They do not disagree; they were
    /// never measuring the same thing.</para>
    ///
    /// <para>The two levers the older harness is missing are the two with the largest measured effect. The
    /// calibrated absolute floors are what took the lab from 112 a day to 5, and the common-mode
    /// decomposition removed ten of eleven healthy-replica findings when it was introduced. A harness
    /// without either cannot produce a number anyone should act on.</para>
    ///
    /// <para><b>Deployable posture, deliberately.</b> Options carry no configured floors, so
    /// <see cref="AnomalyGuard"/> builds a <c>ConfiguredFloorSource</c> over its own calibrator — exactly
    /// what a client runs on day one before anybody has tuned anything. That makes this comparable with the
    /// lab's day-one behaviour rather than with the lab's hand-tuned ConfigMap, and it is the honest
    /// direction to be wrong in: a tuned deployment can only be quieter.</para>
    ///
    /// <para>Knobs: <c>OVERFIT_GFP_PODS</c>, <c>OVERFIT_GFP_HOURS</c>, <c>OVERFIT_GFP_SEED</c>,
    /// <c>OVERFIT_GFP_RESTARTS</c>. One population is one draw — run several seeds before believing a rate.</para>
    /// </summary>
    public sealed class GuardFalsePositiveRateDiagnostics
    {
        private const double ScrapeSeconds = 15.0;
        private const int WindowMinutes = 20;
        private const int StepMinutes = 5;

        private static readonly DateTimeOffset Origin = new(2026, 8, 5, 0, 0, 0, TimeSpan.Zero);

        private readonly ITestOutputHelper _output;

        public GuardFalsePositiveRateDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void MeasuresWhatTheShippedGuardInventsOnAHealthyPopulation()
        {
            var pods = Env("OVERFIT_GFP_PODS", 20);
            var hours = Env("OVERFIT_GFP_HOURS", 72);
            var seed = Env("OVERFIT_GFP_SEED", 20260729);
            var restarts = Env("OVERFIT_GFP_RESTARTS", 1) != 0 ? 1.0 : 0.0;

            // Zero injected faults, so every incident is false by definition. That is the whole design: no
            // ground truth to argue about.
            var cluster = new SyntheticCluster(
                pods, hours, ScrapeSeconds, seed, restartsPerPodPerDay: restarts);

            var samples = (int)(WindowMinutes * 60 / ScrapeSeconds);
            var step = (int)(StepMinutes * 60 / ScrapeSeconds);

            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add(SyntheticCluster.PodName(p));
            }

            var sink = new CountingSink();

            // No MinAbsoluteGap / MinAbsoluteTrendChange: the guard then builds its floors from its own
            // calibrator, which is the untuned deployment.
            var options = new AnomalyGuardOptions
            {
                Namespace = "synthetic",
                Workload = "synthetic-workload",
                Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
            };

            var guard = new AnomalyGuard(options, sink, IncidentTrackingOptions.Balanced);

            var cycles = 0;
            var opened = 0;
            var quiet = 0;

            for (var start = 0; start + samples <= cluster.Samples; start += step, cycles++)
            {
                var window = new MetricWindow(
                    names, samples,
                    Origin.AddSeconds(start * ScrapeSeconds),
                    TimeSpan.FromSeconds(ScrapeSeconds));

                for (var p = 0; p < pods; p++)
                {
                    for (var m = 0; m < (int)MetricIndex.Count; m++)
                    {
                        cluster.Window(p, (MetricIndex)m, start, samples)
                            .CopyTo(window.Series(p, (MetricIndex)m));
                    }
                }

                var at = Origin.AddSeconds(start * ScrapeSeconds).AddMinutes(WindowMinutes);
                var outcome = guard.RunCycle(window, at);

                opened += outcome.Opened;

                if (outcome.Findings == 0)
                {
                    quiet++;
                }
            }

            var cyclesPerDay = 24.0 * 60.0 / StepMinutes;
            var perDay = cycles == 0 ? 0.0 : opened / (double)cycles * cyclesPerDay;

            var report = new StringBuilder();

            report.AppendLine(CultureInfo.InvariantCulture,
                $"pods {pods}   hours {hours}   seed {seed}   restarts/pod/day {restarts}");
            report.AppendLine(CultureInfo.InvariantCulture,
                $"cycles {cycles}   opened {opened}   quiet {quiet} ({(cycles == 0 ? 0 : quiet * 100.0 / cycles):F0}%)");
            report.AppendLine(CultureInfo.InvariantCulture,
                $"rate   {perDay:F1} incidents per day");

            // A Poisson interval, because a count this small read as a rate without one is a point estimate
            // dressed up as a measurement.
            var low = Math.Max(0.0, opened - (1.96 * Math.Sqrt(opened)));
            var high = opened + (1.96 * Math.Sqrt(opened));

            report.AppendLine(CultureInfo.InvariantCulture,
                $"       95% Poisson on {opened} events: "
                + $"{(cycles == 0 ? 0 : low / cycles * cyclesPerDay):F1}-{(cycles == 0 ? 0 : high / cycles * cyclesPerDay):F1} per day");

            report.AppendLine();
            report.AppendLine("opened by signal:");

            foreach (var (signal, count) in sink.OpenedBySignal.OrderByDescending(e => e.Value))
            {
                report.AppendLine(CultureInfo.InvariantCulture, $"   {signal,-26}{count,5}");
            }

            _output.WriteLine(report.ToString());
        }

        private static int Env(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }

        /// <summary>Counts opened incidents by signal. Rows are only valid during the call, so they are copied.</summary>
        private sealed class CountingSink : IIncidentSink
        {
            public Dictionary<string, int> OpenedBySignal { get; } = new(StringComparer.Ordinal);

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (rows[i].State != IncidentState.Opened)
                    {
                        continue;
                    }

                    var signal = rows[i].Signal;

                    OpenedBySignal[signal] = OpenedBySignal.GetValueOrDefault(signal) + 1;
                }
            }
        }
    }
}
