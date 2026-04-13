// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using System;
using System.Collections.Generic;
using System.Linq;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Normalizers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class TrainingIntegrationTests
    {
        private readonly ITestOutputHelper _output;

        public TrainingIntegrationTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void E2E_LSTMAutoencoder_ShouldLearnAndReduceMseLoss()
        {
            const int WindowSize = 30;
            const int StepSeconds = 60;
            const int MetricCount = (int)MetricIndex.Count;
            const int TimeFeaturesCount = 4;
            const int TotalFeatures = MetricCount + TimeFeaturesCount;

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
            using var autoencoder = new LSTMAutoencoder(inputSize: TotalFeatures, seqLen: WindowSize);
            using var optimizer = new Adam(autoencoder.Parameters(), learningRate: 0.01f);

            var baseTsMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var totalScrapes = 50;
            var totalSamplesRequired = WindowSize + totalScrapes;
            var veryStartMs = baseTsMs - (WindowSize * StepSeconds * 1000);

            var allSeries = new List<RawMetricSeries>();
            for (byte m = 0; m < (byte)MetricIndex.Count; m++)
            {
                var series = new RawMetricSeries
                {
                    Pod = new PodKey { DC = DataCenter.West, PodName = "api-pod-1" },
                    MetricTypeId = m
                };

                for (var i = 0; i < totalSamplesRequired; i++)
                {
                    var tsMs = veryStartMs + (i * StepSeconds * 1000);
                    var val = 50f + MathF.Sin(i * 0.2f) * 20f;
                    series.Samples.Add(new RawSample { Timestamp = tsMs, Value = val });
                }
                allSeries.Add(series);
            }

            var graph = new ComputationGraph();
            var initialLoss = -1f;
            var finalLoss = -1f;

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

                using var trainingInput = new FastTensor<float>(1, WindowSize, TotalFeatures, clearMemory: true);
                var infSpan = trainingInput.GetView().AsSpan();
                var devSpan = scaledResult.PodDeviations.GetView().AsReadOnlySpan();

                for (var w = 0; w < WindowSize; w++)
                {
                    // ZABEZPIECZENIE: Sprawdzamy czy mamy dane dla całego okna czy tylko ostatni krok
                    if (devSpan.Length >= (w + 1) * MetricCount)
                    {
                        devSpan.Slice(w * MetricCount, MetricCount).CopyTo(infSpan.Slice(w * TotalFeatures, MetricCount));
                    }
                    else if (devSpan.Length >= MetricCount)
                    {
                        devSpan.Slice(devSpan.Length - MetricCount, MetricCount).CopyTo(infSpan.Slice(w * TotalFeatures, MetricCount));
                    }

                    var currentMs = windowStartMs + w * StepSeconds * 1000;
                    var timeSpan = infSpan.Slice(w * TotalFeatures + MetricCount, TimeFeaturesCount);

                    var (hSin, hCos, dSin, dCos) = DateTimeNormalizer.EncodeAllTimeFeaturesFromUnixSeconds(currentMs / 1000);

                    timeSpan[0] = hSin;
                    timeSpan[1] = hCos;
                    timeSpan[2] = dSin;
                    timeSpan[3] = dCos;
                }

                graph.Reset();
                optimizer.ZeroGrad();

                using var inputNode = new AutogradNode(trainingInput, false);
                using var reconstruction = autoencoder.Forward(graph, inputNode);
                using var loss = TensorMath.MSELoss(graph, reconstruction, inputNode);

                var currentLoss = loss.DataView.AsReadOnlySpan()[0];
                if (i == 0) initialLoss = currentLoss;
                if (i == totalScrapes - 1) finalLoss = currentLoss;

                graph.Backward(loss);
                optimizer.Step();

                scaledResult.FleetBaseline.Dispose();
                scaledResult.PodDeviations.Dispose();
            }

            _output.WriteLine($"Initial Loss: {initialLoss:F4}, Final Loss: {finalLoss:F4}");
            Assert.True(finalLoss < initialLoss, "Model did not learn to reconstruct the sequence properly.");
        }
    }
}