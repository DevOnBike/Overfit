using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Data.Normalizers;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests.Integration
{
    public class TrainingIntegrationTests
    {
        private readonly ITestOutputHelper _output;

        public TrainingIntegrationTests(ITestOutputHelper output)
        {
            _output = output; // Umożliwia logowanie wyników z wewnątrz testu xUnit
        }

        [Fact]
        public void E2E_LSTMAutoencoder_ShouldLearnAndReduceMseLoss()
        {
            // -------------------------------------------------------------------------
            // ARRANGE - Architektura
            // -------------------------------------------------------------------------
            const int WindowSize = 30; // Krótsze okno dla szybszego testu
            const int StepSeconds = 60;
            const int MetricCount = (int)MetricIndex.Count; // 12
            const int TimeFeaturesCount = 4;
            const int TotalFeatures = MetricCount + TimeFeaturesCount; // 16

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

            // Inicjalizacja sieci LSTM
            var autoencoder = new LSTMAutoencoder(
                inputSize: TotalFeatures,
                seqLen: WindowSize,
                encoderHidden: 32, // Mniejsza sieć do szybkiego testu
                latentSize: 16,
                decoderHidden: 32);

            // Wygenerowanie 10 paczek "prostych do nauczenia" danych (np. przewidywalna fala)
            var historicalBatches = GenerateLearnableBatches(WindowSize, StepSeconds, totalScrapes: 10);

            // Faza Fit (Skalery)
            foreach (var batch in historicalBatches)
            {
                pipeline.Process(batch.Series, batch.WindowStartMs, batch.ScrapeTimestampMs);
            }
            pipeline.FinalizeTrainingPhase(); // Zamrożenie skalerów

            // Wyciągnięcie przetransformowanych tensorów dla Autoenkodera
            var trainingTensors = new List<FastTensor<float>>();
            foreach (var batch in historicalBatches)
            {
                var scaledResult = pipeline.Process(batch.Series, batch.WindowStartMs, batch.ScrapeTimestampMs);
                var devSpan = scaledResult.PodDeviations.AsSpan();

                for (var pod = 0; pod < scaledResult.PodIndex.Count; pod++)
                {
                    // Tensor dla jednego poda [Batch=1, SeqLen=30, Features=16]
                    var tensor = new FastTensor<float>(1, WindowSize, TotalFeatures);
                    var tensorSpan = tensor.AsSpan();

                    for (var t = 0; t < WindowSize; t++)
                    {
                        var stepSecondsUnix = (batch.WindowStartMs / 1000) + (t * StepSeconds);
                        var offset = t * TotalFeatures;

                        var timeFeat = DateTimeNormalizer.EncodeAllTimeFeaturesFromUnixSeconds(stepSecondsUnix);
                        tensorSpan[offset + 0] = timeFeat.HourSin;
                        tensorSpan[offset + 1] = timeFeat.HourCos;
                        tensorSpan[offset + 2] = timeFeat.DaySin;
                        tensorSpan[offset + 3] = timeFeat.DayCos;

                        for (var m = 0; m < MetricCount; m++)
                        {
                            var devIdx = (pod * WindowSize * MetricCount) + (t * MetricCount) + m;
                            tensorSpan[offset + TimeFeaturesCount + m] = devSpan[devIdx];
                        }
                    }
                    trainingTensors.Add(tensor);
                }
            }

            // -------------------------------------------------------------------------
            // ACT - Uczenie modelu przy użyciu Adam Optimizer i Autograd
            // -------------------------------------------------------------------------

            autoencoder.Train(); // Aktywacja grafu dla wag

            // Ustawiamy dość wysoki Learning Rate, aby zobaczyć szybki spadek w 5 epokach
            using var optimizer = new Adam(autoencoder.Parameters(), learningRate: 0.05f);
            using var scheduler = new LRScheduler(optimizer, autoencoder.Parameters().ToArray(), msg => _output.WriteLine(msg));

            const int epochs = 5;
            var firstEpochLoss = 0f;
            var lastEpochLoss = 0f;

            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var epochLoss = 0f;

                foreach (var inputTensor in trainingTensors)
                {
                    // Cykl życia grafu dla jednego okna
                    var graph = new ComputationGraph();
                    var inputNode = new AutogradNode(inputTensor, requiresGrad: false);

                    // 1. FORWARD PASS
                    var outputNode = autoencoder.Forward(graph, inputNode);

                    // 2. STRATA (MSE)
                    var lossNode = TensorMath.MSELoss(graph, outputNode, inputNode);

                    // 3. BACKWARD PASS (Liczenie gradientów do t=0)
                    graph.Backward(lossNode);

                    // 4. AKTUALIZACJA WAG
                    optimizer.Step();
                    optimizer.ZeroGrad();

                    epochLoss += lossNode.Data.AsSpan()[0]; // Zbieranie błędu
                }

                epochLoss /= trainingTensors.Count;
                _output.WriteLine($"Epoch {epoch} | MSE: {epochLoss:F6}");

                scheduler.Step(epochLoss);

                if (epoch == 1) firstEpochLoss = epochLoss;
                if (epoch == epochs) lastEpochLoss = epochLoss;
            }

            // -------------------------------------------------------------------------
            // ASSERT - Dowód na uczenie się
            // -------------------------------------------------------------------------
            Assert.True(lastEpochLoss < firstEpochLoss, $"Błąd na końcu ({lastEpochLoss:F4}) musi być mniejszy niż na początku ({firstEpochLoss:F4}).");

            // Sieć powinna znacząco zmniejszyć błąd (przynajmniej o 30%)
            Assert.True(lastEpochLoss < firstEpochLoss * 0.7f, "Model nie zbiega wystarczająco szybko!");
        }

        // ====================================================================
        // HELPER: Generator logów "nauczalnych" (sinusoida)
        // ====================================================================
        private List<(long ScrapeTimestampMs, long WindowStartMs, List<RawMetricSeries> Series)> GenerateLearnableBatches(
            int windowSize, int stepSeconds, int totalScrapes)
        {
            var batches = new List<(long, long, List<RawMetricSeries>)>();
            var baseTsMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

            var totalSamplesRequired = windowSize + totalScrapes;
            var veryStartMs = baseTsMs - (windowSize * stepSeconds * 1000);

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
                    var tsMs = veryStartMs + (i * stepSeconds * 1000);
                    // Podstawiamy piękną, łagodną falę, którą LSTM uwielbia modelować
                    var val = 50f + MathF.Sin(i * 0.2f) * 20f;
                    series.Samples.Add(new RawSample { Timestamp = tsMs, Value = val });
                }
                allSeries.Add(series);
            }

            for (var i = 0; i < totalScrapes; i++)
            {
                var scrapeTsMs = baseTsMs + (i * stepSeconds * 1000);
                var windowStartMs = scrapeTsMs - ((windowSize - 1) * stepSeconds * 1000);

                batches.Add((scrapeTsMs, windowStartMs, allSeries));
            }

            return batches;
        }
    }
}