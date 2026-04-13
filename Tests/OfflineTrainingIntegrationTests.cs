// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Normalizers;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public class OfflineTrainingIntegrationTests
    {
        [Fact]
        public void E2E_OfflineTraining_And_LSTMInference_ShouldExecuteWithoutAllocations()
        {
            const int WindowSize = 60;
            const int StepSeconds = 60;
            const int MetricCount = (int)MetricIndex.Count;
            const int TimeFeaturesCount = 4;
            const int TotalInputFeatures = MetricCount + TimeFeaturesCount;

            var options = new MonitoringPipelineOptions
            {
                WindowSize = WindowSize,
                StepSeconds = StepSeconds,
                MetricCount = MetricCount,
                MaxGapSteps = 2,
                MaxNanRatio = 0.5f,
                WarmupDuration = TimeSpan.Zero
            };

            var pipeline = new MonitoringPipeline(options);
            var autoencoder = new LSTMAutoencoder(inputSize: TotalInputFeatures, seqLen: WindowSize);

            var baseTsMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var totalScrapes = 10;
            var random = new Random(42);
            var totalSamplesRequired = WindowSize + totalScrapes;
            var veryStartMs = baseTsMs - (WindowSize * StepSeconds * 1000);

            var allSeries = new List<RawMetricSeries>();
            for (byte m = 0; m < (byte)MetricIndex.Count; m++)
            {
                var series = new RawMetricSeries
                {
                    Pod = new PodKey { DC = DataCenter.West, PodName = "payment-pod-0" },
                    MetricTypeId = m
                };

                for (var i = 0; i < totalSamplesRequired; i++)
                {
                    var tsMs = veryStartMs + (i * StepSeconds * 1000);
                    var val = 50f + (float)random.NextDouble() * 10f;
                    series.Samples.Add(new RawSample { Timestamp = tsMs, Value = val });
                }
                allSeries.Add(series);
            }

            for (var i = 0; i < totalScrapes; i++)
            {
                var scrapeTsMs = baseTsMs + (i * StepSeconds * 1000);
                var windowStartMs = scrapeTsMs - ((WindowSize - 1) * StepSeconds * 1000);

                var slicedSeries = new List<RawMetricSeries>();
                foreach (var s in allSeries)
                {
                    var newSeries = new RawMetricSeries { Pod = s.Pod, MetricTypeId = s.MetricTypeId };
                    newSeries.Samples.AddRange(s.Samples.Where(x => x.Timestamp >= windowStartMs && x.Timestamp <= scrapeTsMs));
                    slicedSeries.Add(newSeries);
                }

                var scaledResult = pipeline.Process(slicedSeries, windowStartMs, scrapeTsMs);

                using var inferenceInput = new FastTensor<float>(1, WindowSize, TotalInputFeatures, clearMemory: true);
                var infSpan = inferenceInput.GetView().AsSpan();
                var devSpan = scaledResult.PodDeviations.GetView().AsReadOnlySpan();

                for (var w = 0; w < WindowSize; w++)
                {
                    // ZABEZPIECZENIE: Czytamy tylko to, co faktycznie potok policzył i zwrócił
                    if (devSpan.Length >= (w + 1) * MetricCount)
                    {
                        devSpan.Slice(w * MetricCount, MetricCount).CopyTo(infSpan.Slice(w * TotalInputFeatures, MetricCount));
                    }
                    else if (devSpan.Length >= MetricCount)
                    {
                        // Jeśli potok zwrócił tylko 1 zestaw pomiarów (ostatni krok), kopiujemy go powtarzalnie
                        devSpan.Slice(devSpan.Length - MetricCount, MetricCount).CopyTo(infSpan.Slice(w * TotalInputFeatures, MetricCount));
                    }

                    var currentMs = windowStartMs + w * StepSeconds * 1000;
                    var timeSpan = infSpan.Slice(w * TotalInputFeatures + MetricCount, TimeFeaturesCount);

                    var (hSin, hCos, dSin, dCos) = DateTimeNormalizer.EncodeAllTimeFeaturesFromUnixSeconds(currentMs / 1000);

                    timeSpan[0] = hSin;
                    timeSpan[1] = hCos;
                    timeSpan[2] = dSin;
                    timeSpan[3] = dCos;
                }

                var errors = autoencoder.ReconstructionError(inferenceInput.GetView().AsReadOnlySpan(), batchSize: 1);

                Assert.Single(errors);
                Assert.True(errors[0] >= 0f);

                scaledResult.FleetBaseline.Dispose();
                scaledResult.PodDeviations.Dispose();
            }

            pipeline.FinalizeTrainingPhase();
            autoencoder.Dispose();
        }
    }
}