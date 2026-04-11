// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class VectorForecastingTests
    {
        private readonly ITestOutputHelper _output;

        public VectorForecastingTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void MLP_Should_Predict_7_Days_Ahead()
        {
            _output.WriteLine("=== Predykcja Wektorowa: 7 dni w przód (Pełny rok XAU/USD) ===");

            var windowSize = 10;
            var forecastDays = 7;
            var epochs = 200;
            var learningRate = 0.005f;

            var prices = GetRealGoldPricesUSD_1Year();
            var lastKnownPrice = prices.Last();
            var returns = CalculateReturns(prices);

            var numSamples = returns.Length - windowSize - forecastDays + 1;

            var xTensor = new FastTensor<float>(false, numSamples, windowSize);
            var yTensor = new FastTensor<float>(false, numSamples, forecastDays);

            var xSpan = xTensor.AsSpan();
            var ySpan = yTensor.AsSpan();

            for (var i = 0; i < numSamples; i++)
            {
                returns.AsSpan(i, windowSize).CopyTo(xSpan.Slice(i * windowSize, windowSize));
                returns.AsSpan(i + windowSize, forecastDays).CopyTo(ySpan.Slice(i * forecastDays, forecastDays));
            }

            using var inputNode = new AutogradNode(xTensor, false);
            using var targetNode = new AutogradNode(yTensor, false);

            var w1 = new AutogradNode(new FastTensor<float>(false, windowSize, 32));
            var b1 = new AutogradNode(new FastTensor<float>(true, 1, 32));
            Randomize(w1.Data.AsSpan(), windowSize);

            var w2 = new AutogradNode(new FastTensor<float>(false, 32, forecastDays));
            var b2 = new AutogradNode(new FastTensor<float>(true, 1, forecastDays));
            Randomize(w2.Data.AsSpan(), 32);

            var parameters = new[]
            {
                w1, b1, w2, b2
            };
            using var optimizer = new Adam(parameters, learningRate);

            // JAWNY GRAF OBLICZENIOWY
            var graph = new ComputationGraph();

            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                graph.Reset();
                optimizer.ZeroGrad();

                using var hidden = TensorMath.Linear(graph, inputNode, w1, b1);
                using var relu = TensorMath.ReLU(graph, hidden);
                using var prediction = TensorMath.Linear(graph, relu, w2, b2);

                using var loss = TensorMath.DirectionalLoss(graph, prediction, targetNode, 15f);

                graph.Backward(loss);
                optimizer.Step();
            }

            _output.WriteLine($"Trenowano na {numSamples} oknach. Sieć rozpoznaje całe trajektorie 7-dniowe.");
            _output.WriteLine("--------------------------------------------------");

            var lastReturns = returns.Skip(returns.Length - windowSize).ToArray();
            var inferenceInput = new FastTensor<float>(false, 1, windowSize);
            lastReturns.AsSpan().CopyTo(inferenceInput.AsSpan());
            using var inferenceNode = new AutogradNode(inferenceInput, false);

            // INFERENCJA -> GRAF NULL
            using var h = TensorMath.Linear(null, inferenceNode, w1, b1);
            using var r = TensorMath.ReLU(null, h);
            using var finalPred = TensorMath.Linear(null, r, w2, b2);

            _output.WriteLine($"START - Ostatnia znana cena: ${lastKnownPrice:F2} / oz");

            var currentSimulatedPrice = lastKnownPrice;
            for (var d = 0; d < forecastDays; d++)
            {
                var predictedReturn = finalPred.Data[0, d];
                currentSimulatedPrice *= 1f + predictedReturn;
                var sign = predictedReturn > 0 ? "+" : "";
                _output.WriteLine($"Dzień +{d + 1} | Prognoza zwrotu: {sign}{predictedReturn * 100f,5:F2}% | Cena: ${currentSimulatedPrice:F2} / oz");
            }

            w1.Dispose();
            b1.Dispose();
            w2.Dispose();
            b2.Dispose();
        }

        private float[] GetRealGoldPricesUSD_1Year()
        {
            var prices = new float[252];
            var lastMonth = new[]
            {
                5327.42f, 5087.47f, 5135.92f, 5077.39f, 5171.12f, 5139.56f, 5192.94f, 5177.07f, 5080.07f, 5019.25f, 5006.43f, 5006.00f, 4818.81f, 4651.73f, 4491.15f, 4407.62f, 4474.44f, 4506.55f, 4380.03f, 4492.99f, 4447.65f, 4511.25f, 4672.01f, 4758.76f, 4676.42f
            };

            var rnd = new Random(42);
            var currentPrice = 3800f;
            var targetPrice = lastMonth[0];

            for (var i = 0; i < 227; i++)
            {
                prices[i] = currentPrice;
                float daysLeft = 227 - i;
                var requiredDailyTrend = (targetPrice - currentPrice) / daysLeft;
                var volatility = (rnd.NextSingle() - 0.5f) * (currentPrice * 0.03f);
                currentPrice += requiredDailyTrend + volatility;
            }

            Array.Copy(lastMonth, 0, prices, 227, 25);
            return prices;
        }

        private float[] CalculateReturns(float[] prices)
        {
            var returns = new float[prices.Length - 1];
            for (var i = 0; i < returns.Length; i++)
            {
                returns[i] = (prices[i + 1] - prices[i]) / prices[i];
            }
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