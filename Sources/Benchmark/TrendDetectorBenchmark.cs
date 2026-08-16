// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Statistics;

namespace Benchmarks
{
    /// <summary>
    /// What one trend evaluation costs, and where the time actually goes.
    ///
    /// <para>This is a <b>budget</b> question before it is an A/B one. <see cref="TrendDetector"/> is
    /// <c>O(n²)</c> in pairs and sits on the per-series, per-cycle path of the cluster guard: a 600-sample
    /// window means ~180 000 pairs, and a modest cluster is 100 pods x 13 metrics = 1300 series every scrape
    /// interval. Whether <c>MaxSamplesForFit = 600</c> is affordable at that scale is not something to reason
    /// about — it is a number.</para>
    ///
    /// <para><b>The ablation.</b> Three arms decompose the shipped detector so the dominant term is visible
    /// rather than inferred: pair accumulation alone, plus the sort, plus everything else the detector does.
    /// The repo has been wrong before about which half of a routine costs the time, so the arms are nested
    /// deliberately.</para>
    ///
    /// <para><b>The candidate.</b> The shipped path calls <c>slopes.Sort()</c> and then takes the median of
    /// the sorted span — a full <c>O(n log n)</c> ordering to answer an <c>O(n)</c> question.
    /// <see cref="TheilSen_QuickSelect"/> replaces it with Hoare selection. The hypothesis is that this is a
    /// straight win at 180 000 elements; the hypothesis is worth exactly nothing until the ratio column
    /// says so.</para>
    ///
    /// <para>Deliberately NOT on the shared <c>BenchmarkConfig</c>, for the reason given in
    /// <see cref="MannWhitneyBenchmark"/>: that job pins <c>InvocationCount=1</c> and would leave the small
    /// windows measuring timer noise.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class TrendDetectorBenchmark
    {
        private readonly TrendDetector _detector = new();

        private double[] _values = [];
        private double[] _times = [];
        private double[] _slopes = [];
        private double[] _scratch = [];

        /// <summary>
        /// Observations in the window. 120 is an hour at a 30 s scrape, 600 is the detector's thinning cap —
        /// five hours at the same rate, and the largest window that reaches the fit untouched.
        /// </summary>
        [Params(60, 120, 300, 600)]
        public int Samples
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260728);

            _values = new double[Samples];
            _times = new double[Samples];

            // A memory series that leaks: a real upward slope, multiplicative noise, and the occasional
            // scrape artefact that is the whole reason a robust estimator was chosen over least squares.
            var bytes = 380.0 * 1024 * 1024;
            for (var i = 0; i < Samples; i++)
            {
                _times[i] = i * 30.0;
                bytes += 22.0 * 1024;

                var noise = 1.0 + ((rng.NextDouble() - 0.5) * 0.01);
                _values[i] = bytes * noise;

                if (rng.NextDouble() < 0.02)
                {
                    _values[i] *= 6.0;
                }
            }

            var pairs = Samples * (Samples - 1) / 2;
            _slopes = new double[pairs];
            _scratch = new double[pairs];
        }

        /// <summary>The shipped detector, end to end — the number the product actually pays.</summary>
        [Benchmark(Baseline = true)]
        public double Detect_Full()
        {
            return _detector.Detect(_values, _times, TrendOptions.Balanced).SlopePerSecond;
        }

        /// <summary>Pair accumulation only: the <c>O(n²)</c> sweep, with no ordering on top of it.</summary>
        [Benchmark]
        public long Pairs_Only()
        {
            return AccumulatePairs(_times, _values, _slopes);
        }

        /// <summary>Pairs plus the full sort — the shipped Theil-Sen, isolated from the rest of the detector.</summary>
        [Benchmark]
        public double TheilSen_Sort()
        {
            AccumulatePairs(_times, _values, _slopes);
            var span = _slopes.AsSpan();
            span.Sort();

            return SortedMedian(span);
        }

        /// <summary>
        /// The shipped replacement: the same median by selection, calling the real
        /// <see cref="MedianSelector"/> rather than a copy of it, so this arm cannot drift away from what
        /// <see cref="Detect_Full"/> actually runs. Copies into scratch first because selection permutes its
        /// input and both arms must start from identical state.
        /// </summary>
        [Benchmark]
        public double TheilSen_QuickSelect()
        {
            AccumulatePairs(_times, _values, _slopes);
            _slopes.AsSpan().CopyTo(_scratch);

            return MedianSelector.MedianInPlace(_scratch.AsSpan());
        }

        // Verbatim copy of TrendDetector.AccumulatePairs — internals are visible to this assembly, but the
        // method is private, and a copy is the honest way to isolate it without widening the library's surface.
        private static long AccumulatePairs(
            ReadOnlySpan<double> times,
            ReadOnlySpan<double> observations,
            Span<double> slopes)
        {
            var written = 0;
            var s = 0L;

            for (var i = 0; i < times.Length - 1; i++)
            {
                for (var j = i + 1; j < times.Length; j++)
                {
                    var difference = observations[j] - observations[i];
                    slopes[written] = difference / (times[j] - times[i]);
                    written++;

                    if (difference > 0.0)
                    {
                        s++;
                        continue;
                    }

                    if (difference < 0.0)
                    {
                        s--;
                    }
                }
            }

            return s;
        }

        private static double SortedMedian(ReadOnlySpan<double> sorted)
        {
            var middle = sorted.Length / 2;

            if (sorted.Length % 2 == 1)
            {
                return sorted[middle];
            }

            return 0.5 * (sorted[middle - 1] + sorted[middle]);
        }

    }
}
