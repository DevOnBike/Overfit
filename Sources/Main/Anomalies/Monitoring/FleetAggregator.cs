// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    public sealed class FleetAggregator
    {
        private readonly FleetAggregatorOptions _options;

        public FleetAggregator(FleetAggregatorOptions options)
        {
            _options = options;
        }

        // ---------------------------------------------------------------------------
        // Entry point
        // ---------------------------------------------------------------------------

        public AggregationResult Aggregate(AlignResult result)
        {
            var windowSize = _options.WindowSize;
            var metricCount = _options.MetricCount;
            var dcCount = 2; // West + East

            // Step 1 — compute per-DC medians
            var fleetBaseline = new float[dcCount * windowSize * metricCount];
            ComputeFleetMedians(result, fleetBaseline, windowSize, metricCount);

            // Step 2 — compute per-pod deviations from their DC median
            var podCount = result.Windows.Count;
            var podDeviations = new float[podCount * windowSize * metricCount];

            ComputePodDeviations(result, podDeviations, fleetBaseline, windowSize, metricCount);

            return new AggregationResult
            {
                FleetBaseline = fleetBaseline,
                PodDeviations = podDeviations,
                PodIndex = result.PodIndex,
                DcCount = dcCount,
                PodCount = podCount,
                WindowSize = windowSize,
                MetricCount = metricCount
            };
        }

        // ---------------------------------------------------------------------------
        // Step 1 — fleet median per DC
        // ---------------------------------------------------------------------------

        private static void ComputeFleetMedians(
            AlignResult result,
            float[] fleetBaseline,
            int windowSize,
            int metricCount)
        {
            // Collect pod data per DC into lists for median computation
            var dcPods = new List<float[]>[2];
            dcPods[0] = []; // West
            dcPods[1] = []; // East

            for (var i = 0; i < result.Windows.Count; i++)
            {
                var window = result.Windows[i];
                var dcIdx = (int)result.PodIndex[i].DC;
                dcPods[dcIdx].Add(window.Data);
            }

            // Compute median per DC, per timestep, per metric
            for (var dc = 0; dc < 2; dc++)
            {
                var pods = dcPods[dc];
                if (pods.Count == 0)
                {
                    continue;
                }

                for (var t = 0; t < windowSize; t++)
                {
                    for (var m = 0; m < metricCount; m++)
                    {
                        var dataIdx = t * metricCount + m;
                        var baselineIdx = dc * windowSize * metricCount + t * metricCount + m;

                        fleetBaseline[baselineIdx] = ComputeMedian(pods, dataIdx);
                    }
                }
            }
        }

        // ---------------------------------------------------------------------------
        // Step 2 — pod deviation from DC median
        // ---------------------------------------------------------------------------

        private static void ComputePodDeviations(
            AlignResult result,
            float[] podDeviations,
            float[] fleetBaseline,
            int windowSize,
            int metricCount)
        {
            for (var i = 0; i < result.Windows.Count; i++)
            {
                var podData = result.Windows[i].Data;
                var dcIdx = (int)result.PodIndex[i].DC;

                for (var t = 0; t < windowSize; t++)
                {
                    for (var m = 0; m < metricCount; m++)
                    {
                        var dataIdx = t * metricCount + m;
                        var baselineIdx = dcIdx * windowSize * metricCount + dataIdx;
                        var devIdx = i * windowSize * metricCount + dataIdx;

                        podDeviations[devIdx] = podData[dataIdx] - fleetBaseline[baselineIdx];
                    }
                }
            }
        }

        // ---------------------------------------------------------------------------
        // Median — sort-free selection via nth_element approach
        // For small N (typically 20 pods per DC) simple sort is fine
        // ---------------------------------------------------------------------------

        private static float ComputeMedian(List<float[]> pods, int dataIdx)
        {
            var n = pods.Count;
            var values = new float[n];

            for (var i = 0; i < n; i++)
            {
                values[i] = pods[i][dataIdx];
            }

            Array.Sort(values);

            return n % 2 == 1
                ? values[n / 2]
                : (values[n / 2 - 1] + values[n / 2]) * 0.5f;
        }
    }

    // ---------------------------------------------------------------------------
    // KONFIGURACJA
    // ---------------------------------------------------------------------------

}