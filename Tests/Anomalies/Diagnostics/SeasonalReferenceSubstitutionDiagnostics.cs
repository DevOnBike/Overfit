// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// `AN-F1`: seasonal history tripled the incident rate on an identical window — 11 opened against 33,
    /// 296 findings against 517 — and the standing explanation was refuted by measurement.
    ///
    /// <para><b>This measures a different mechanism, found by reading the code rather than by fitting.</b>
    /// `AnomalyGuard.RunTrend` chooses the per-pod reference like this:</para>
    ///
    /// <code>
    /// var expectation = seasonal;
    /// …
    /// if (expectation.IsEmpty) { expectation = common; }
    /// </code>
    ///
    /// <para>So <b>with</b> history each pod's trend is judged against the seasonal expectation, and
    /// <b>without</b> it against the cross-peer common component. Those are not substitutes. The common
    /// component removes <i>what the replicas are doing together right now</i>; the seasonal one removes
    /// <i>what this workload usually does at this hour</i>. Substituting the second for the first stops
    /// removing common-mode movement from every per-pod series — so on a fleet that moves together, each
    /// replica reports the fleet's own climb as its own trend, and a fleet of twelve produces twelve
    /// findings where the other reference produces none.</para>
    ///
    /// <para><b>Driven at the detector, not through the guard, and that is deliberate.</b> The lever is one
    /// argument to <see cref="TrendDetector.Detect"/>; building a guard, a sink, grouping and incident
    /// tracking around it would add four things that can also change the count. Both references are produced
    /// by the real <see cref="CrossPeerBaseline"/> and the real detector — this isolates the argument, it
    /// does not reimplement it.</para>
    ///
    /// <para><b>Set `OVERFIT_RUN_LONG` to run it.</b> It reports; the only assertions are the ones that would
    /// make the report meaningless.</para>
    /// </summary>
    public sealed class SeasonalReferenceSubstitutionDiagnostics
    {
        private const int Pods = 12;
        private const int Length = 80;
        private const double StepSeconds = 15.0;

        [LongFact]
        public void SubstitutingTheSeasonalReferenceStopsRemovingCommonModeMovement()
        {
            var report = new StringBuilder();

            report.Append("=== AN-F1: which reference the per-pod trend is judged against ===\n\n");
            report.Append(
                "Each row is the same window judged twice. `common` is what RunTrend uses when there is NO\n"
                + "seasonal history; `seasonal` is what it uses when there is. Counts are pods reported\n"
                + "Anomalous out of 12.\n\n");
            report.Append("  fleet movement        common   seasonal\n");

            var detector = new TrendDetector();
            var options = TrendOptions.Balanced with { MinAbsoluteChangeOverWindow = 1.089e6 };
            var times = new double[Length];

            for (var i = 0; i < Length; i++)
            {
                times[i] = i * StepSeconds;
            }

            var rows = new List<(string Label, int Common, int Seasonal)>();

            foreach (var (label, climb) in new[]
                     {
                         ("flat", 0.0),
                         ("+5% together", 0.05),
                         ("+15% together", 0.15),
                         ("+30% together", 0.30),
                     })
            {
                var pods = Fleet(climb);
                var common = CommonComponent(pods);

                // What a week of quiet days produces: the workload's usual level, flat across the window.
                // It is the honest shape for this test — a seasonal expectation that is WRONG would be a
                // different hypothesis, and the refuted one.
                var seasonal = new double[Length];
                Array.Fill(seasonal, pods[0][0]);

                var againstCommon = CountAnomalous(detector, pods, times, options, common);
                var againstSeasonal = CountAnomalous(detector, pods, times, options, seasonal);

                rows.Add((label, againstCommon, againstSeasonal));
                report.Append(CultureInfo.InvariantCulture,
                    $"  {label,-18}  {againstCommon,6}   {againstSeasonal,8}\n");
            }

            report.Append(
                "\nReading: the flat fleet must agree — if it does not, the two references differ for a reason\n"
                + "that has nothing to do with common-mode movement and this whole explanation is wrong. On the\n"
                + "moving fleets, `common` should stay near zero (the movement is shared, so each pod's residual\n"
                + "against it is flat) while `seasonal` rises towards one finding per pod.\n");

            var flat = rows[0];
            var biggest = rows[^1];

            report.Append(CultureInfo.InvariantCulture,
                $"\nflat: common={flat.Common} seasonal={flat.Seasonal} (must match)\n"
                + $"largest movement: common={biggest.Common} seasonal={biggest.Seasonal}\n");

            // The refuting assertion, and it comes first on purpose: if the two references disagree on a
            // fleet that is NOT moving, they differ for some reason other than common-mode removal and the
            // whole explanation above is wrong.
            Assert.True(
                flat.Common == flat.Seasonal,
                "the two references disagree on a FLAT fleet, so the difference is not about common-mode "
                + "movement and this diagnostic explains nothing:" + Environment.NewLine + report);

            // The mechanism itself. Measured 2026-08-10: 0 against 12, at +15% and at +30%.
            Assert.True(
                biggest.Seasonal > biggest.Common,
                "substituting the seasonal reference did NOT increase per-pod findings on a fleet moving "
                + "together, which refutes the AN-F1 mechanism:" + Environment.NewLine + report);

            Assert.True(
                biggest.Common == 0,
                "the cross-peer reference reported a pod on a fleet whose movement is entirely shared — "
                + "removing exactly that is what it is for:" + Environment.NewLine + report);
        }

        /// <summary>
        /// The half of this that stops the fix being a regression: **one pod genuinely diverging from a flat
        /// fleet must still be reported against the cross-peer reference.**
        ///
        /// <para>Without it, "fewer findings" and "blind" are the same measurement. The existing
        /// `SeasonalExpectationTests` cannot answer this — traced by hand, all four of its pods move
        /// identically in every scenario, so nothing there exercises a divergence and none of it would have
        /// caught the original `AN-F1` regression or would catch a re-regression.</para>
        /// </summary>
        [Fact]
        public void APodDivergingFromAFlatFleetIsStillReportedAgainstTheCommonReference()
        {
            var detector = new TrendDetector();
            var options = TrendOptions.Balanced with { MinAbsoluteChangeOverWindow = 1.089e6 };
            var times = new double[Length];

            for (var i = 0; i < Length; i++)
            {
                times[i] = i * StepSeconds;
            }

            var pods = Fleet(0.0);

            // Pod 0 climbs 15% while its eleven siblings stay flat — the shape the channel exists to catch,
            // and the exact case a common-mode reference must NOT absorb.
            var rng = new Random(20260811);
            var start = pods[0][0];

            for (var i = 0; i < Length; i++)
            {
                var phase = i / (double)(Length - 1);

                pods[0][i] = start * (1.0 + (0.15 * phase)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.05));
            }

            var common = CommonComponent(pods);
            var verdict = detector.Detect(pods[0], times, options, double.NaN, common);

            Assert.True(
                verdict.Status == DetectionStatus.Anomalous,
                "the diverging pod was NOT reported against the cross-peer reference, so the AN-F1 fix "
                + $"trades false positives for blindness. Status was {verdict.Status}.");

            // And the eleven that did not move must stay quiet, or "reported" is just noise.
            var quiet = 0;

            for (var pod = 1; pod < pods.Count; pod++)
            {
                if (detector.Detect(pods[pod], times, options, double.NaN, common).Status
                    == DetectionStatus.Anomalous)
                {
                    quiet++;
                }
            }

            Assert.True(
                quiet == 0,
                $"{quiet} flat sibling(s) were reported alongside the diverging pod — one pod moving must "
                + "not drag its peers into the finding.");
        }

        private static int CountAnomalous(
            TrendDetector detector,
            IReadOnlyList<double[]> pods,
            double[] times,
            TrendOptions options,
            double[] expectation)
        {
            var anomalous = 0;

            for (var pod = 0; pod < pods.Count; pod++)
            {
                var verdict = detector.Detect(pods[pod], times, options, double.NaN, expectation);

                if (verdict.Status == DetectionStatus.Anomalous)
                {
                    anomalous++;
                }
            }

            return anomalous;
        }

        /// <summary>The real cross-peer component, not an average of my own.</summary>
        private static double[] CommonComponent(IReadOnlyList<double[]> pods)
        {
            var peers = new List<PeerSeries>(pods.Count);

            for (var pod = 0; pod < pods.Count; pod++)
            {
                peers.Add(new PeerSeries($"pod-{pod}", pods[pod]));
            }

            var common = new double[Length];

            Assert.True(
                CrossPeerBaseline.TryBuild(peers, common, new double[pods.Count]),
                "the cross-peer baseline refused this fleet, so there is no `common` arm to compare against");

            return common;
        }

        /// <summary>
        /// Twelve replicas moving together, on 5% noise. The noise is not decoration: a perfectly smooth ramp
        /// has a lag-1 autocorrelation near 1.0 which the detector penalises through variance inflation, so a
        /// noiseless fixture is rejected and every arm would read zero for the wrong reason.
        /// </summary>
        private static List<double[]> Fleet(double climb)
        {
            var rng = new Random(20260810);
            var pods = new List<double[]>(Pods);

            for (var pod = 0; pod < Pods; pod++)
            {
                var series = new double[Length];
                var start = 40e6 * (1.0 + ((rng.NextDouble() - 0.5) * 0.02));

                for (var i = 0; i < Length; i++)
                {
                    var phase = i / (double)(Length - 1);

                    series[i] = start * (1.0 + (climb * phase))
                                * (1.0 + ((rng.NextDouble() - 0.5) * 0.05));
                }

                pods.Add(series);
            }

            return pods;
        }
    }
}
