// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Replays the windows the deployed guard actually judged during the <c>AN-D3</c> fleet-wide CPU
    /// injection of 2026-08-11, through the product's own <see cref="CrossPeerBaseline"/> and
    /// <see cref="LevelShiftDetector"/>.
    ///
    /// <para><b>Why this exists rather than arithmetic in a script.</b> The guard's log prints finding lines
    /// only when an incident CHANGES STATE, so once the incident was open the cycles that mattered — 09:52
    /// and 09:57, the ones where the step sat inside the window — reported <c>findings=3</c> and printed
    /// nothing about what those findings were. The instrument could not answer the question the experiment
    /// was asking, so the question is put to the same code on the same data instead.</para>
    ///
    /// <para>Reads CSVs written by the lab helper into <c>Tests/bin</c>; skips cleanly when they are absent,
    /// which is every machine except the one that ran the injection.</para>
    /// </summary>
    public sealed class FleetWideCpuStepReplayDiagnostics
    {
        private readonly ITestOutputHelper _output;

        public FleetWideCpuStepReplayDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void WhatEachJudgedWindowWouldHaveSaid()
        {
            // The lab's configured CPU floor. MinAbsoluteLevelShift resolves from the TREND table
            // (ConfiguredFloorSource.cs:70) and Resolve returns a configured value outright, so this is the
            // number the deployed gate used — not a calibrated one.
            var options = LevelShiftOptions.Balanced with
            {
                MinAbsoluteChange = 0.000326
            };

            foreach (var name in new[] { "094719", "095219", "095719", "100719" })
            {
                var path = Path.Combine(AppContext.BaseDirectory, $"an-d3-{name}.csv");

                if (!File.Exists(path))
                {
                    _output.WriteLine($"{name}: brak {path}");

                    continue;
                }

                var pods = ReadCsv(path);
                var length = pods[0].Length;
                var peers = pods.Select((values, i) =>
                    new PeerSeries($"pod{i:d2}", values.AsMemory())).ToList();
                var common = new double[length];

                if (!CrossPeerBaseline.TryBuild(peers, common, new double[peers.Count]))
                {
                    _output.WriteLine($"{name}: TryBuild odmowil ({peers.Count} podow)");

                    continue;
                }

                var verdict = new LevelShiftDetector().Detect(common, options);
                var step = LevelShiftDetector.StepSize(common);

                _output.WriteLine(
                    $"{name}: {peers.Count} podow x {length} probek | krok {step:G4} rdzenia | "
                    + $"{verdict.Status} | {verdict.Reason}");
            }
        }

        private static List<double[]> ReadCsv(string path)
        {
            var lines = File.ReadAllLines(path);
            var columns = lines[0].Split(',').Length - 1;
            var pods = new List<double[]>(columns);

            for (var c = 0; c < columns; c++)
            {
                pods.Add(new double[lines.Length - 1]);
            }

            for (var row = 1; row < lines.Length; row++)
            {
                var cells = lines[row].Split(',');

                for (var c = 0; c < columns; c++)
                {
                    pods[c][row - 1] = double.TryParse(
                        cells[c + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
                        ? value
                        : double.NaN;
                }
            }

            return pods;
        }
    }
}
