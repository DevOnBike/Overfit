// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Fits <see cref="SyntheticClusterShape"/> to the recorded lab window automatically.
    ///
    /// <para><b>The loop from <c>docs/autoresearch-program.md</c>.</b> The same hill-climb that was performed
    /// by hand — change a constant, run the comparison, read a table, keep or revert — at roughly one step
    /// per millisecond instead of one per five minutes.</para>
    ///
    /// <para><b>This is a (1+1) evolution strategy</b>: one parent, one mutation, keep if better. The only
    /// thing separating it from a genetic algorithm is the mutation operator — deterministic coordinate steps
    /// here, crossover and bit-flips there. Neither can invent a parameter that is not already in the genome,
    /// which is exactly what happened when the lab's p50 could not be matched: no setting of the nine
    /// parameters could satisfy both its interquartile spread and its range, and the fix was a tenth
    /// mechanism (<c>LatencyBurstDelay</c>) that came from reading the table, not from searching it.</para>
    ///
    /// <para>Reports rather than writes. The winning shape is printed as constant values to paste into
    /// <see cref="SyntheticCluster"/>, because a search that edits its own generator is a search whose
    /// baseline moves under it — and because those constants carry documentation a code generator would
    /// flatten.</para>
    ///
    /// <para><b>The objective is only as good as the reference.</b> <see cref="LabWindowValidator"/> runs
    /// first and the search refuses to start on a window it rejects. Fitting to a contaminated recording is
    /// the one outcome here worse than not running at all: fast, repeatable, and wrong.</para>
    /// </summary>
    public sealed class SyntheticClusterCalibrationSearch
    {
        /// <summary>Lower and upper bound per parameter, index-aligned with <see cref="SyntheticClusterShape"/>.</summary>
        private static readonly (double Lo, double Hi)[] Bounds =
        [
            (0.02, 2.00),   // LatencyScatterP50
            (0.02, 3.00),   // LatencyScatterP95
            (0.02, 3.00),   // LatencyScatterP99
            (0.02, 2.00),   // TrafficScatter
            (0.02, 2.00),   // CpuScatter
            (1.00, 3.00),   // TrafficBurstFactor
            (1.00, 3.00),   // CpuBurstFactor
            (0.00, 0.60),   // HeapPromotionStep
            (0.00, 2.00),   // LatencyBurstDelay
            (0.01, 1.20),   // MemorySawtoothAmplitude
            (20.0, 900.0),  // MemoryCycleSamples
        ];

        private static readonly double[] Steps = [1.60, 1.30, 1.15, 1.07, 1.03];

        private readonly ITestOutputHelper _output;

        public SyntheticClusterCalibrationSearch(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void FitsTheGeneratorToTheRecordedLabWindow()
        {
            var (lab, healthy) = Reference();
            var report = new StringBuilder();

            report.Append($"lab: {healthy.Count} healthy pods, {lab.Length} samples\n\n");

            var start = SyntheticClusterShape.Measured;
            report.Append($"installed shape scores {ClusterProfiler.Score(lab, healthy, start):F4}\n\n");

            var result = Descend(lab, healthy, start, line => report.Append("  " + line + "\n"));

            report.Append($"\n{result.Evaluations} evaluations, final score {result.Score:F4}\n\n");
            report.Append("paste into SyntheticCluster:\n");
            AppendShape(report, result.Shape);

            report.Append("\nper-channel, after fitting (gen / lab):\n");
            report.Append($"{"metric",-24}{"iqr gen",10}{"iqr lab",10}{"rng gen",10}{"rng lab",10}"
                          + $"{"retr gen",10}{"retr lab",10}\n");

            foreach (var metric in ClusterProfiler.Scored)
            {
                var reference = ClusterProfiler.FromWindow(lab, healthy, metric);
                var generated = ClusterProfiler.OverSeeds(3, 16, metric, lab.Length, result.Shape);

                if (reference is not { } r || generated is not { } g)
                {
                    report.Append($"{metric,-24}   absent on one side\n");

                    continue;
                }

                report.Append($"{metric,-24}{g.Iqr,10:P1}{r.Iqr,10:P1}{g.Range,10:P1}{r.Range,10:P1}"
                              + $"{g.Retrace,10:P0}{r.Retrace,10:P0}\n");
            }

            _output.WriteLine(report.ToString());

            Assert.True(true, "reported, not asserted — the fitted shape is the product");
        }

        /// <summary>
        /// Does this objective need a population, or is one climber enough?
        ///
        /// <para><b>The cheap experiment that answers "should this be a genetic algorithm".</b> Population
        /// methods earn their cost on rugged landscapes with many local optima; on a smooth one they spend
        /// more evaluations to reach the point a single climber already finds. Rather than argue it, start the
        /// same descent from many random shapes and look at where they land.</para>
        ///
        /// <para>Two different findings are possible and they call for opposite responses. Scattered final
        /// <b>scores</b> mean local optima, and a population would help. Identical scores reached from
        /// different <b>parameters</b> mean the objective has flat directions — a parameter that the data
        /// cannot identify, which no search of any kind can fix, and which is worth knowing before anyone
        /// reads meaning into its fitted value.</para>
        /// </summary>
        [LongFact]
        public void MultiStartDescentShowsWhetherTheLandscapeNeedsAPopulation()
        {
            const int Restarts = 10;

            var (lab, healthy) = Reference();
            var rng = new Random(20260731);
            var report = new StringBuilder();

            var installed = SyntheticClusterShape.Measured;
            var installedScore = ClusterProfiler.Score(lab, healthy, installed);

            report.Append($"installed shape scores {installedScore:F4}\n\n");
            report.Append($"{"start",6}{"from",10}{"to",10}{"evals",7}   parameters at the optimum\n");

            var finals = new List<double>(Restarts);
            var shapes = new List<SyntheticClusterShape>(Restarts);

            for (var r = 0; r < Restarts; r++)
            {
                var start = RandomShape(rng);
                var before = ClusterProfiler.Score(lab, healthy, start);
                var result = Descend(lab, healthy, start, null);

                finals.Add(result.Score);
                shapes.Add(result.Shape);

                var values = new StringBuilder();

                for (var i = 0; i < SyntheticClusterShape.Count; i++)
                {
                    values.Append(result.Shape[i].ToString("F3", CultureInfo.InvariantCulture)).Append(' ');
                }

                report.Append($"{r,6}{before,10:F4}{result.Score,10:F4}{result.Evaluations,7}   {values}\n");
            }

            var sorted = new List<double>(finals);
            sorted.Sort();

            report.Append($"\nfinal scores: best {sorted[0]:F4}, median {sorted[sorted.Count / 2]:F4}, "
                          + $"worst {sorted[^1]:F4}, spread {sorted[^1] - sorted[0]:F4}\n");

            // The spread across restarts is only meaningful next to what descending achieves at all: if the
            // landscape's ruggedness is small compared to the improvement one descent buys, a population
            // would be optimising noise.
            report.Append($"one descent improves the installed shape by {installedScore - sorted[0]:F4}\n");

            report.Append("\nper-parameter spread across the optima:\n");

            for (var i = 0; i < SyntheticClusterShape.Count; i++)
            {
                var lo = double.PositiveInfinity;
                var hi = double.NegativeInfinity;

                foreach (var shape in shapes)
                {
                    lo = Math.Min(lo, shape[i]);
                    hi = Math.Max(hi, shape[i]);
                }

                var band = Bounds[i].Hi - Bounds[i].Lo;

                report.Append($"  {SyntheticClusterShape.Name(i),-20} {lo,8:F3} .. {hi,8:F3}"
                              + $"   {(hi - lo) / band,7:P0} of its allowed range\n");
            }

            _output.WriteLine(report.ToString());

            Assert.True(true, "reported, not asserted — the spread is the answer");
        }

        private static (MetricWindow Lab, List<int> Healthy) Reference()
        {
            Assert.True(LabWindowFixture.Exists, $"fixture missing: {LabWindowFixture.Path}");

            var (lab, faulted) = LabWindowFixture.Load();
            var gate = LabWindowValidator.Validate(lab, faulted);

            Assert.True(
                gate.IsUsable,
                $"refusing to calibrate against a window that failed its own gate — {gate.Describe()}");

            var healthy = new List<int>();

            for (var p = 0; p < lab.Pods.Count; p++)
            {
                if (!faulted.Contains(lab.Pods[p]))
                {
                    healthy.Add(p);
                }
            }

            return (lab, healthy);
        }

        /// <summary>
        /// Coordinate descent with a shrinking multiplicative step. Multiplicative because the parameters span
        /// two orders of magnitude, so one additive step would be a coarse move on one and a rounding error
        /// on another.
        /// </summary>
        private static (SyntheticClusterShape Shape, double Score, int Evaluations) Descend(
            MetricWindow lab,
            IReadOnlyList<int> healthy,
            SyntheticClusterShape start,
            Action<string>? log)
        {
            var shape = start;
            var best = ClusterProfiler.Score(lab, healthy, shape);
            var evaluations = 1;

            foreach (var step in Steps)
            {
                var improved = true;

                while (improved)
                {
                    improved = false;

                    for (var i = 0; i < SyntheticClusterShape.Count; i++)
                    {
                        foreach (var direction in new[] { step, 1.0 / step })
                        {
                            var value = Math.Clamp(shape[i] * direction, Bounds[i].Lo, Bounds[i].Hi);

                            if (Math.Abs(value - shape[i]) < 1e-9)
                            {
                                continue;
                            }

                            var candidate = shape.With(i, value);
                            var score = ClusterProfiler.Score(lab, healthy, candidate);
                            evaluations++;

                            if (score >= best - 1e-6)
                            {
                                continue;
                            }

                            log?.Invoke($"{SyntheticClusterShape.Name(i),-20} "
                                        + $"{shape[i]:F4} -> {value:F4}   {best:F4} -> {score:F4}");

                            shape = candidate;
                            best = score;
                            improved = true;
                        }
                    }
                }
            }

            return (shape, best, evaluations);
        }

        /// <summary>
        /// A random shape inside the bounds. Log-uniform where the parameter is a scale: a uniform draw over
        /// 0.02 to 3.0 puts almost every start in the top decade, so the restarts would only ever explore one
        /// corner and "they all converge" would mean nothing.
        /// </summary>
        private static SyntheticClusterShape RandomShape(Random rng)
        {
            var shape = SyntheticClusterShape.Measured;

            for (var i = 0; i < SyntheticClusterShape.Count; i++)
            {
                var (lo, hi) = Bounds[i];

                var value = lo > 0.0
                    ? Math.Exp(Math.Log(lo) + (rng.NextDouble() * (Math.Log(hi) - Math.Log(lo))))
                    : lo + (rng.NextDouble() * (hi - lo));

                shape = shape.With(i, value);
            }

            return shape;
        }

        private static void AppendShape(StringBuilder report, SyntheticClusterShape shape)
        {
            for (var i = 0; i < SyntheticClusterShape.Count; i++)
            {
                report.Append($"  {SyntheticClusterShape.Name(i)} = "
                              + $"{shape[i].ToString("G4", CultureInfo.InvariantCulture)};\n");
            }
        }
    }
}
