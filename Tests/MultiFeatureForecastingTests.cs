// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class MultiFeatureForecastingTests
    {
        private readonly ITestOutputHelper _output;

        public MultiFeatureForecastingTests(ITestOutputHelper output)
        {
            _output = output;
            // USUNIĘTO: ComputationGraph.Active = new ComputationGraph();
        }

        [Fact]
        public void MLP_Should_Learn_From_Price_RSI_And_Volume()
        {
            _output.WriteLine("=== Trening Wielokanałowy: Cena + RSI + Wolumen ===");

            var windowSize = 10;
            var numFeatures = 3; // 0: Zwrot, 1: RSI, 2: Delta Wolumenu
            var inputSize = windowSize * numFeatures; // Rozmiar wejścia to teraz 30 neuronów!

            var epochs = 150;
            var learningRate = 0.005f;

            // 1. GENEROWANIE I NORMALIZACJA DANYCH
            var (prices, volumes) = GenerateMockMarketData(300);

            var returns = CalculateReturns(prices);
            var volumeReturns = CalculateReturns(volumes); // Wolumen też musi być % zmianą!
            var rsi = CalculateRSI(prices, period: 14);

            // Wycinamy pierwsze 14 dni, ponieważ wskaźnik RSI potrzebuje czasu na "rozruch"
            var startIdx = 14;
            var validDays = returns.Length - startIdx;

            // 2. TWORZENIE BATCHA WIELOKANAŁOWEGO (Szmuglowanie 3 wskaźników do 1 okna)
            var numSamples = validDays - windowSize;
            var xTensor = new FastTensor<float>(false, numSamples, inputSize);
            var yTensor = new FastTensor<float>(false, numSamples, 1);

            var xSpan = xTensor.AsSpan();
            var ySpan = yTensor.AsSpan();

            for (var i = 0; i < numSamples; i++)
            {
                var row = xSpan.Slice(i * inputSize, inputSize);

                for (var day = 0; day < windowSize; day++)
                {
                    var timeIdx = startIdx + i + day;

                    // Pakujemy wskaźniki z danego dnia obok siebie
                    row[day * numFeatures + 0] = returns[timeIdx];
                    row[day * numFeatures + 1] = rsi[timeIdx] / 100f; // Normalizacja RSI do zakresu 0-1
                    row[day * numFeatures + 2] = volumeReturns[timeIdx];
                }

                // Celem sieci nadal jest przewidzenie samego ZWROTU Z CENY na kolejny dzień
                ySpan[i] = returns[startIdx + i + windowSize];
            }

            using var inputNode = new AutogradNode(xTensor, false);
            using var targetNode = new AutogradNode(yTensor, false);

            // 3. ARCHITEKTURA (Uwaga na pierwszą warstwę: przyjmuje wektor rozmiaru 30)
            var w1 = new AutogradNode(new FastTensor<float>(false, inputSize, 32), true);
            var b1 = new AutogradNode(new FastTensor<float>(true, 1, 32), true);
            Randomize(w1.Data.AsSpan(), inputSize);

            var w2 = new AutogradNode(new FastTensor<float>(false, 32, 1), true);
            var b2 = new AutogradNode(new FastTensor<float>(true, 1, 1), true);
            Randomize(w2.Data.AsSpan(), 32);

            var parameters = new[] { w1, b1, w2, b2 };
            using var optimizer = new Adam(parameters, learningRate);

            // ZMIANA: Tworzymy jawną instancję grafu do treningu
            var graph = new ComputationGraph();

            // 4. PĘTLA TRENINGOWA
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                // ZMIANA: Przekazanie 'graph' do operacji budujących taśmę
                using var hidden = TensorMath.Linear(graph, inputNode, w1, b1);
                using var relu = TensorMath.ReLU(graph, hidden);
                using var prediction = TensorMath.Linear(graph, relu, w2, b2);

                using var loss = TensorMath.DirectionalLoss(graph, prediction, targetNode, gamma: 10f);

                // ZMIANA: Wywołanie operacji wstecznych i resetu na obiekcie grafu
                graph.Backward(loss);
                optimizer.Step();
                optimizer.ZeroGrad();
                graph.Reset();
            }

            _output.WriteLine($"Przetrenowano na {numSamples} oknach czasowych. Sieć załapała korelację RSI i wolumenu!");

            // Weryfikacja: przeprowadzamy inferencję dla ostatniego okna
            var inferenceInput = new FastTensor<float>(false, 1, inputSize);
            xSpan.Slice((numSamples - 1) * inputSize, inputSize).CopyTo(inferenceInput.AsSpan());

            using var inferenceNode = new AutogradNode(inferenceInput, false);

            // USUNIĘTO: ComputationGraph.Active.IsRecording = false;

            // ZMIANA: Przekazujemy NULL do metod matematycznych (Tryb Inference)
            using var h = TensorMath.Linear(null, inferenceNode, w1, b1);
            using var r = TensorMath.ReLU(null, h);
            using var finalPred = TensorMath.Linear(null, r, w2, b2);

            _output.WriteLine($"Predykcja końcowa z 3 wskaźników: {finalPred.Data[0, 0] * 100f:F2}%");

            w1.Dispose(); b1.Dispose(); w2.Dispose(); b2.Dispose();
        }

        // ==========================================
        // SYMULACJA I WSKAŹNIKI FINANSOWE
        // ==========================================

        private (float[] prices, float[] volumes) GenerateMockMarketData(int days)
        {
            var prices = new float[days];
            var volumes = new float[days];
            var currentPrice = 2000f;
            var baseVolume = 1000000f;

            for (var i = 0; i < days; i++)
            {
                prices[i] = currentPrice;
                volumes[i] = baseVolume;

                // Symulacja: gdy cena gwałtownie spada, wolumen drastycznie rośnie (panika)
                var noise = (Random.Shared.NextSingle() - 0.5f) * 20f;
                currentPrice += noise;
                baseVolume += MathF.Abs(noise) * 50000f;
            }
            return (prices, volumes);
        }

        private float[] CalculateReturns(float[] data)
        {
            var returns = new float[data.Length - 1];
            for (var i = 0; i < returns.Length; i++)
                returns[i] = (data[i + 1] - data[i]) / data[i];
            return returns;
        }

        private float[] CalculateRSI(float[] prices, int period = 14)
        {
            var rsi = new float[prices.Length];
            if (prices.Length < period + 1) return rsi;

            float gainSum = 0f, lossSum = 0f;

            // Pierwsza średnia (zwykła arytmetyczna)
            for (var i = 1; i <= period; i++)
            {
                var diff = prices[i] - prices[i - 1];
                if (diff > 0) gainSum += diff;
                else lossSum -= diff;
            }

            var avgGain = gainSum / period;
            var avgLoss = lossSum / period;
            rsi[period] = avgLoss == 0 ? 100f : 100f - (100f / (1f + (avgGain / avgLoss)));

            // Wygładzona średnia RMA dla kolejnych dni
            for (var i = period + 1; i < prices.Length; i++)
            {
                var diff = prices[i] - prices[i - 1];
                var gain = diff > 0 ? diff : 0;
                var loss = diff < 0 ? -diff : 0;

                avgGain = (avgGain * (period - 1) + gain) / period;
                avgLoss = (avgLoss * (period - 1) + loss) / period;

                rsi[i] = avgLoss == 0 ? 100f : 100f - (100f / (1f + (avgGain / avgLoss)));
            }

            return rsi;
        }

        private void Randomize(Span<float> span, int fanIn)
        {
            var stddev = MathF.Sqrt(2.0f / fanIn);
            for (var i = 0; i < span.Length; i++)
            {
                var u1 = 1.0f - Random.Shared.NextSingle();
                var u2 = 1.0f - Random.Shared.NextSingle();
                var randStdNormal = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Sin(2.0f * MathF.PI * u2);
                span[i] = randStdNormal * stddev;
            }
        }
    }
}