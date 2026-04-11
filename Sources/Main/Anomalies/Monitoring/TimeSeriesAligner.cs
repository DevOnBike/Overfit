// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    ///     Aligns raw Prometheus series into fixed-size windows on a shared time grid.
    ///     Input:  flat List of RawMetricSeries (mixed pods, mixed metrics)
    ///     Output: AlignResult — one RawPodWindow per pod + parallel PodIndex
    /// </summary>
    public sealed class TimeSeriesAligner
    {
        private readonly AlignerOptions _options;

        public TimeSeriesAligner(AlignerOptions options)
        {
            _options = options;
        }
        public AlignResult Align(List<RawMetricSeries> allSeries, long globalStartMs)
        {
            // Internal grouping — caller does not need to pre-sort by pod
            var grouped = new Dictionary<PodKey, List<RawMetricSeries>>();
            foreach (var series in allSeries)
            {
                if (!grouped.TryGetValue(series.Pod, out var list))
                {
                    grouped[series.Pod] = list = [];
                }

                list.Add(series);
            }

            var result = new AlignResult();

            foreach (var (podKey, seriesList) in grouped)
            {
                result.PodIndex.Add(podKey);
                result.Windows.Add(AlignPod(podKey.DC, seriesList, globalStartMs));
            }

            return result;
        }

        private RawPodWindow AlignPod(
            DataCenter dc,
            List<RawMetricSeries> seriesList,
            long globalStartMs)
        {
            var data = new float[_options.WindowSize * _options.MetricCount];

            data.AsSpan().Fill(float.NaN);

            foreach (var series in seriesList)
            {
                int mId = series.MetricTypeId;
                if ((uint)mId >= (uint)_options.MetricCount) continue;

                series.Samples.Sort(static (a, b) => a.Timestamp.CompareTo(b.Timestamp));

                AlignMetric(series.Samples, globalStartMs, data, mId);
            }

            return new RawPodWindow
            {
                DC = dc,
                Data = data
            };
        }

        // ---------------------------------------------------------------------------

        private void AlignMetric(
            List<RawSample> samples,
            long globalStartMs,
            float[] data,
            int metricTypeId)
        {
            var stepMs = _options.StepMs;
            var tolMs = _options.ToleranceMs;
            var mCount = _options.MetricCount;
            var lastKnown = float.NaN;
            var gapCount = 0;

            for (var t = 0; t < _options.WindowSize; t++)
            {
                var targetMs = globalStartMs + (long)t * stepMs;
                var value = FindNearest(samples, targetMs, tolMs);

                if (float.IsNaN(value))
                {
                    gapCount++;
                    value = gapCount <= _options.MaxGapSteps ? lastKnown : float.NaN;
                }
                else
                {
                    lastKnown = value;
                    gapCount = 0;
                }

                data[t * mCount + metricTypeId] = value;
            }
        }

        // ---------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float FindNearest(List<RawSample> samples, long targetMs, int toleranceMs)
        {
            if (samples.Count == 0) return float.NaN;

            int lo = 0, hi = samples.Count - 1;

            while (lo <= hi)
            {
                var mid = lo + hi >>> 1;
                var diff = samples[mid].Timestamp - targetMs;

                if (diff < 0) lo = mid + 1;
                else if (diff > 0) hi = mid - 1;
                else return samples[mid].Value;
            }

            var best = float.NaN;
            var minDiff = toleranceMs + 1L;

            if ((uint)lo < (uint)samples.Count)
            {
                var d = Math.Abs(samples[lo].Timestamp - targetMs);
                if (d <= toleranceMs && d < minDiff)
                {
                    minDiff = d;
                    best = samples[lo].Value;
                }
            }

            if ((uint)hi < (uint)samples.Count)
            {
                var d = Math.Abs(samples[hi].Timestamp - targetMs);
                if (d <= toleranceMs && d < minDiff) { best = samples[hi].Value; }
            }

            return best;
        }
    }
}