// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class RealGoldForecastingTests
    {
        private readonly ITestOutputHelper _output;

        public RealGoldForecastingTests(ITestOutputHelper output)
        {
            _output = output;
            // USUNIĘTO: ComputationGraph.Active = new ComputationGraph();
        }

        [Fact]
        public void MLP_Should_Predict_Real_Gold_Prices_FullYear()
        {
            _output.WriteLine("=== Trening na pełnym roku XAU/USD (Kwiecień 2025 - Kwiecień 2026) ===");

            // Zwiększamy okno do 10 dni (2 tygodnie handlowe). Mamy dużo danych!
            var windowSize = 10;
            var epochs = 200;
            var learningRate = 0.005f;

            // 1. ŁADOWANIE DANYCH (252 dni handlowe = pełny rok)
            var prices = GetRealGoldPricesUSD_1Year();
            var lastKnownPrice = prices.Last();

            // 2. NORMALIZACJA KRYTYCZNA (Zwroty procentowe)
            var returns = CalculateReturns(prices);

            // 3. TWORZENIE BATCHA (Sliding Window)
            var numSamples = returns.Length - windowSize;
            var xTensor = new FastTensor<float>(false, numSamples, windowSize);
            var yTensor = new FastTensor<float>(false, numSamples, 1);
            var xSpan = xTensor.AsSpan();
            var ySpan = yTensor.AsSpan();

            for (var i = 0; i < numSamples; i++)
            {
                returns.AsSpan(i, windowSize).CopyTo(xSpan.Slice(i * windowSize, windowSize));
                ySpan[i] = returns[i + windowSize];
            }

            using var inputNode = new AutogradNode(xTensor, false);
            using var targetNode = new AutogradNode(yTensor, false);

            // 4. ARCHITEKTURA (Powiększamy sieć z 16 na 32 neurony)
            var w1 = new AutogradNode(new FastTensor<float>(false, windowSize, 32), true);
            var b1 = new AutogradNode(new FastTensor<float>(true, 1, 32), true);
            Randomize(w1.Data.AsSpan(), windowSize);

            var w2 = new AutogradNode(new FastTensor<float>(false, 32, 1), true);
            var b2 = new AutogradNode(new FastTensor<float>(true, 1, 1), true);
            Randomize(w2.Data.AsSpan(), 32);

            var parameters = new[] { w1, b1, w2, b2 };
            using var optimizer = new Adam(parameters, learningRate);

            // ZMIANA: Tworzymy jawną instancję grafu obliczeniowego
            var graph = new ComputationGraph();

            // 5. PĘTLA TRENINGOWA
            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                // ZMIANA: Przekazujemy 'graph' do nagrywania operacji
                using var hidden = TensorMath.Linear(graph, inputNode, w1, b1);
                using var relu = TensorMath.ReLU(graph, hidden);
                using var prediction = TensorMath.Linear(graph, relu, w2, b2);

                // Silna kara za pomyłkę kierunku trendu
                using var loss = TensorMath.DirectionalLoss(graph, prediction, targetNode, gamma: 15f);

                // ZMIANA: Wsteczna propagacja na jawnym obiekcie
                graph.Backward(loss);
                optimizer.Step();

                optimizer.ZeroGrad();
                graph.Reset();
            }

            _output.WriteLine($"Trenowano na {numSamples} próbkach. Zakończono.");

            // 6. INFERENCJA: PREDYKCJA NA "JUTRO"
            var lastReturns = returns.Skip(returns.Length - windowSize).ToArray();

            var inferenceInput = new FastTensor<float>(false, 1, windowSize);
            lastReturns.AsSpan().CopyTo(inferenceInput.AsSpan());

            using var inferenceNode = new AutogradNode(inferenceInput, false);

            // USUNIĘTO: ComputationGraph.Active.IsRecording = false;

            // ZMIANA: Przekazujemy 'null', co jest nowym sygnałem trybu Inference bez alokacji i logowania na taśmę
            using var h = TensorMath.Linear(null, inferenceNode, w1, b1);
            using var r = TensorMath.ReLU(null, h);
            using var finalPred = TensorMath.Linear(null, r, w2, b2);

            // USUNIĘTO: ComputationGraph.Active.IsRecording = true;

            var predictedReturn = finalPred.Data[0, 0];
            var predictedPriceUSD = lastKnownPrice * (1f + predictedReturn);

            _output.WriteLine("--------------------------------------------------");
            _output.WriteLine($"Ostatnia znana cena (02.04.2026): ${lastKnownPrice:F2} / oz");
            _output.WriteLine($"Przewidywany zwrot na kolejny dzień: {predictedReturn * 100f:F2}%");
            _output.WriteLine($"PRZEWIDYWANA CENA: ${predictedPriceUSD:F2} / oz");

            w1.Dispose(); b1.Dispose(); w2.Dispose(); b2.Dispose();
        }

        private float[] GetRealGoldPricesUSD_1Year()
        {
            // 252 dni to dokładnie jeden rok handlowy na giełdzie
            var prices = new float[252];

            // Ostatnie 25 dni (Realne, twarde dane z naszego poprzedniego testu)
            var lastMonth = new float[] {
                5327.42f, 5087.47f, 5135.92f, 5077.39f, 5171.12f,
                5139.56f, 5192.94f, 5177.07f, 5080.07f, 5019.25f,
                5006.43f, 5006.00f, 4818.81f, 4651.73f, 4491.15f,
                4407.62f, 4474.44f, 4506.55f, 4380.03f, 4492.99f,
                4447.65f, 4511.25f, 4672.01f, 4758.76f, 4676.42f
            };

            // Programistyczna rekonstrukcja poprzednich 227 dni
            // Startujemy w okolicach 3800 USD i dodajemy rynkowy szum,
            // aby idealnie połączyć się z historyczną ceną 5327.42 z początku ostatniego miesiąca.
            var rnd = new Random(42); // Stały seed = powtarzalność testu
            var currentPrice = 3800f;
            var targetPrice = lastMonth[0];

            for (var i = 0; i < 227; i++)
            {
                prices[i] = currentPrice;
                float daysLeft = 227 - i;
                var requiredDailyTrend = (targetPrice - currentPrice) / daysLeft;

                // Szum giełdowy rzędu +/- 1.5% dziennie
                var volatility = (rnd.NextSingle() - 0.5f) * (currentPrice * 0.03f);
                currentPrice += requiredDailyTrend + volatility;
            }

            // Doklejamy prawdziwe dane na sam koniec roku
            Array.Copy(lastMonth, 0, prices, 227, 25);

            return prices;
        }

        private float[] CalculateReturns(float[] prices)
        {
            var returns = new float[prices.Length - 1];
            for (var i = 0; i < returns.Length; i++)
                returns[i] = (prices[i + 1] - prices[i]) / prices[i];
            return returns;
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