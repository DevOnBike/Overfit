// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Measures what the live view costs the HOST arms, in one process, ABAB.
    /// <para>
    /// This exists because a display that refreshes while C1, C2 and C3 are being timed competes with
    /// them for the same cores, and "it is only a console write" is an assumption rather than a number.
    /// CLAUDE.md records the general form of the failure from 2026-08-20: an instrument written for one
    /// measurement measures itself until it is checked separately. The whole point of this probe is that
    /// its numbers can be trusted, so the thing added to watch it has to be measured too.
    /// </para>
    /// <para>
    /// The design mirrors the probe's own rules. Both arms run in ONE process and ONE sitting; the
    /// blocks alternate ABAB so drift over the sitting falls on both equally rather than wearing the
    /// costume of the variable; only one lever moves, which is whether <see cref="LiveView.Refresh"/>
    /// does anything; and the refresh happens between rounds and never between arms.
    /// </para>
    /// <para>
    /// <b>Read the cadence before reading the ratio.</b> Nothing here is threaded, so a repaint is never
    /// concurrent with an arm - it sits between rounds, and what it costs the next round is the state it
    /// leaves behind rather than any cycle it steals during one. That makes the ratio a function of HOW
    /// OFTEN the repaint happens, which is what <c>--live-perturbation-every=N</c> varies: at N=1 every
    /// timed reading follows a repaint, and at N equal to the block length only one in twelve does, so
    /// the median of the block is taken from rounds that never saw one. The shipped probe repaints
    /// between phases, which is the second case.
    /// </para>
    /// <para>
    /// It needs no GPU. The host arms are the ones at risk and <see cref="StubTelemetry"/> stands in for
    /// the device, so this is fully runnable on a machine with no NVIDIA card.
    /// </para>
    /// </summary>
    internal static class LivePerturbation
    {
        private const int Blocks = 8;
        private const int RoundsPerBlock = 12;

        public static void Run(ProbeOptions options, Report report, TextWriter log)
        {
            report.TopBanners.Add(
                "--live-perturbation: this run measured what the LIVE VIEW costs the host arms and timed " +
                "nothing else. It is a fact about the instrument, not about any GPU.");

            var cell = (options.Quick ? Cell.Quick : Cell.Production)[0];
            log.WriteLine($"perturbation: quantizing {cell.Name}...");
            var weights = new CellWeights(cell, options.Seed);
            var n = options.Batches[0];
            using var fixture = new CellFixture(weights, n, options.Seed);

            var arms = new List<Arm>
            {
                Arm.Cpu(ArmNames.C3, fixture.RunC3Forward),
                Arm.Cpu(ArmNames.C4, fixture.RunC4Backward),
                Arm.Cpu(ArmNames.C1, fixture.RunC1Forward, after: fixture.ResetC1),
            };

            var state = new LiveProbeState { Cell = cell.Name, N = n, Phase = "perturbation A/B" };
            using var telemetry = new StubTelemetry(options.Seed);
            var view = LiveView.TryCreate(telemetry, state, force: true);

            if (view is null)
            {
                report.TopBanners.Add(
                    $"THE LIVE VIEW COULD NOT START, so the perturbation was NOT measured: {LiveView.Unavailable}. " +
                    "A run under a redirected terminal cannot answer this question - use a real console.");
                return;
            }

            // Per arm: the readings taken while the view was suspended, and while it was refreshing.
            var suspended = new Dictionary<string, List<double>>();
            var refreshing = new Dictionary<string, List<double>>();
            foreach (var arm in arms)
            {
                suspended[arm.Name] = [];
                refreshing[arm.Name] = [];
            }

            using (view)
            {
                view.Run(() =>
                {
                    // One untimed block first: the arms must be warm before either side is recorded, or
                    // the cold start lands entirely on whichever block happens to run first.
                    RunBlock(arms, view, live: false, null, options.LivePerturbationEvery);

                    for (var block = 0; block < Blocks; block++)
                    {
                        var live = block % 2 == 1;
                        state.Phase = live ? "block B - view refreshing" : "block A - view suspended";
                        RunBlock(arms, view, live, live ? refreshing : suspended, options.LivePerturbationEvery);
                    }
                });
            }

            foreach (var arm in arms)
            {
                var a = Median(suspended[arm.Name]);
                var b = Median(refreshing[arm.Name]);
                var line = string.Create(
                    CultureInfo.InvariantCulture,
                    $"{arm.Name,-24} suspended {a,9:F4} ms   refreshing {b,9:F4} ms   " +
                    $"ratio {(a > 0 ? b / a : 0),6:F3}   ({suspended[arm.Name].Count} readings each side)");
                report.LivePerturbation.Add(line);
                log.WriteLine("  " + line);
            }

            var cadence = options.LivePerturbationEvery == 1
                ? "one refresh per round, between rounds"
                : $"one refresh every {options.LivePerturbationEvery} rounds, between rounds";
            report.LivePerturbation.Add(
                $"shape {cell.Name} k {cell.K} -> m {cell.M} n={n}; {Blocks} blocks of {RoundsPerBlock} " +
                $"rounds, alternating, one process, one sitting; {cadence}.");

            if (Console.IsErrorRedirected)
            {
                report.LivePerturbation.Add(
                    "TAKEN WITH STANDARD ERROR REDIRECTED. The frame was built and written either way, so " +
                    "the layout cost is the real one; the terminal I/O cost of an interactive console is " +
                    "not covered by this number and could differ.");
            }
        }

        private static void RunBlock(
            IReadOnlyList<Arm> arms,
            LiveView view,
            bool live,
            Dictionary<string, List<double>>? into,
            int every)
        {
            view.Suspended = !live;

            for (var round = 0; round < RoundsPerBlock; round++)
            {
                foreach (var arm in arms)
                {
                    var ms = arm.TimeOnce();
                    into?[arm.Name].Add(ms);
                }

                // Between rounds, never between arms. Suspended blocks reach the same call and it returns
                // immediately, so the two sides differ by the repaint and nothing else - not by an extra
                // branch, not by a different loop shape. The modulo is evaluated in both blocks for the
                // same reason.
                if (round % every == 0)
                {
                    view.Refresh();
                }
            }
        }

        private static double Median(List<double> values)
        {
            if (values.Count == 0)
            {
                return 0;
            }

            var sorted = values.ToArray();
            Array.Sort(sorted);
            return sorted.Length % 2 == 1
                ? sorted[sorted.Length / 2]
                : 0.5 * (sorted[(sorted.Length / 2) - 1] + sorted[sorted.Length / 2]);
        }
    }
}
