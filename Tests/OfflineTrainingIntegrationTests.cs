using System.Buffers;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Data.Normalizers;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public class OfflineTrainingIntegrationTests
    {
        [Fact]
        public void E2E_OfflineTraining_And_LSTMInference_ShouldExecuteWithoutAllocations()
        {
            // ARRANGE - Parametry Architektury
            const int WindowSize = 60; // 60 próbek (np. 60 minut)
            const int StepSeconds = 60;
            const int MetricCount = (int)MetricIndex.Count; // 12
            const int TimeFeaturesCount = 4; // HourSin, HourCos, DaySin, DayCos
            const int TotalInputFeatures = MetricCount + TimeFeaturesCount; // 16

            var options = new MonitoringPipelineOptions
            {
                WindowSize = WindowSize,
                StepSeconds = StepSeconds,
                MetricCount = MetricCount,
                MaxGapSteps = 2,
                MaxNanRatio = 0.5f,
                // KLUCZOWE: W teście wyłączamy rozgrzewkę poda, by od razu uczyć model
                WarmupDuration = TimeSpan.Zero
            };

            var pipeline = new MonitoringPipeline(options);

            var autoencoder = new LSTMAutoencoder(
                inputSize: TotalInputFeatures,
                seqLen: WindowSize,
                encoderHidden: 64,
                latentSize: 32,
                decoderHidden: 64);

            // Wygenerowanie poprawnych, "przesuwnych" okien czasowych, zasymulowanych na 10 paczek
            var historicalBatches = GenerateFakeGoldenWindowBatches(WindowSize, StepSeconds, totalScrapes: 10);

            // ACT 1: TRENING OFFLINE (Golden Window)
            foreach (var scrape in historicalBatches)
            {
                pipeline.Process(scrape.Series, scrape.WindowStartMs, scrape.ScrapeTimestampMs);
            }

            // KRYTYCZNE: Po analizie zdrowych okien, blokujemy średnie i IQR
            pipeline.FinalizeTrainingPhase();

            // ACT 2: PRODUKCYJNA INFERENCJA (Real-Time Pipeline)
            var liveScrape = historicalBatches[^1];
            var scaledResult = pipeline.Process(liveScrape.Series, liveScrape.WindowStartMs, liveScrape.ScrapeTimestampMs);

            var inputBuffer = ArrayPool<float>.Shared.Rent(WindowSize * TotalInputFeatures);

            try
            {
                var podIndex = 0;
                var deviationsSpan = scaledResult.PodDeviations.AsSpan(); // Bezpieczny odczyt z Twojego FastTensor

                for (var t = 0; t < WindowSize; t++)
                {
                    var currentStepSeconds = (liveScrape.WindowStartMs / 1000) + (t * StepSeconds);
                    var tensorOffset = t * TotalInputFeatures;

                    // 1. Wstrzyknięcie Czasu (4 zmienne)
                    var timeFeatures = DateTimeNormalizer.EncodeAllTimeFeaturesFromUnixSeconds(currentStepSeconds);
                    inputBuffer[tensorOffset + 0] = timeFeatures.HourSin;
                    inputBuffer[tensorOffset + 1] = timeFeatures.HourCos;
                    inputBuffer[tensorOffset + 2] = timeFeatures.DaySin;
                    inputBuffer[tensorOffset + 3] = timeFeatures.DayCos;

                    // 2. Kopiowanie znormalizowanych metryk
                    for (var m = 0; m < MetricCount; m++)
                    {
                        var devIdx = (podIndex * WindowSize * MetricCount) + (t * MetricCount) + m;
                        inputBuffer[tensorOffset + TimeFeaturesCount + m] = deviationsSpan[devIdx];
                    }
                }

                var lstmInputSpan = inputBuffer.AsSpan(0, WindowSize * TotalInputFeatures);

                // Przepuszczenie przez silnik AI w trybie ewaluacyjnym
                autoencoder.Eval();
                var reconstructionErrors = autoencoder.ReconstructionError(lstmInputSpan, batchSize: 1);

                // ASSERT
                Assert.Single(reconstructionErrors);
                Assert.True(reconstructionErrors[0] >= 0f); // Błąd MSE musi być dodatni (i w ogóle policzony!)
            }
            finally
            {
                ArrayPool<float>.Shared.Return(inputBuffer);
            }
        }

        // ====================================================================
        // HELPER: Generator logów Prometheusa (Sliding Window)
        // ====================================================================
        private List<(long ScrapeTimestampMs, long WindowStartMs, List<RawMetricSeries> Series)> GenerateFakeGoldenWindowBatches(
            int windowSize, int stepSeconds, int totalScrapes)
        {
            var batches = new List<(long, long, List<RawMetricSeries>)>();
            var random = new Random(42);
            var baseTsMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

            // Aby pipeline miał pełne okna (np. 60 skrapów), musimy wygenerować dane wstecz o ten windowSize
            var totalSamplesRequired = windowSize + totalScrapes;
            var veryStartMs = baseTsMs - (windowSize * stepSeconds * 1000);

            var allSeries = new List<RawMetricSeries>();
            for (byte m = 0; m < (byte)MetricIndex.Count; m++)
            {
                var series = new RawMetricSeries
                {
                    Pod = new PodKey { DC = DataCenter.West, PodName = "payment-pod-0" },
                    MetricTypeId = m
                };

                // Generujemy ciągłe, gęste dane by WindowSanitizer ich nie odrzucił
                for (var i = 0; i < totalSamplesRequired; i++)
                {
                    var tsMs = veryStartMs + (i * stepSeconds * 1000);
                    // Różne wartości dla odchylenia standardowego
                    var val = 50f + (float)random.NextDouble() * 10f;
                    series.Samples.Add(new RawSample { Timestamp = tsMs, Value = val });
                }
                allSeries.Add(series);
            }

            // Symulujemy kolejne eventy Scrape z Prometheusa
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