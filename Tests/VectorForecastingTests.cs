// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
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

            using var xTensor = new FastTensor<float>(numSamples, windowSize, clearMemory: false);
            using var yTensor = new FastTensor<float>(numSamples, forecastDays, clearMemory: false);

            var xSpan = xTensor.GetView().AsSpan();
            var ySpan = yTensor.GetView().AsSpan();

            for (var i = 0; i < numSamples; i++)
            {
                for (var w = 0; w < windowSize; w++)
                {
                    xSpan[i * windowSize + w] = returns[i + w] * 100f;
                }
                for (var f = 0; f < forecastDays; f++)
                {
                    ySpan[i * forecastDays + f] = returns[i + windowSize + f] * 100f;
                }
            }

            using var X = new AutogradNode(xTensor, false);
            using var Y = new AutogradNode(yTensor, false);

            using var layer1 = new LinearLayer(windowSize, 32);
            using var layer2 = new LinearLayer(32, 16);
            using var layer3 = new LinearLayer(16, forecastDays);

            var model = new Sequential(layer1, new ReluActivation(), layer2, new ReluActivation(), layer3);
            var adam = new Adam(model.Parameters(), learningRate) { UseAdamW = true };

            var finalLoss = 0f;
            var graph = new ComputationGraph();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                graph.Reset();
                adam.ZeroGrad();

                using var prediction = model.Forward(graph, X);
                using var loss = TensorMath.MSELoss(graph, prediction, Y);
                finalLoss = loss.DataView.AsReadOnlySpan()[0];

                graph.Backward(loss);
                adam.Step();
            }

            _output.WriteLine($"Training finished. Final Vector MSE: {finalLoss:F4}");
            Assert.True(finalLoss < 2.5f, "Loss too high, model didn't converge on vector output.");
        }

        private float[] GetRealGoldPricesUSD_1Year()
        {
            var prices = new float[252];
            float[] lastMonth = {
                4683.50f, 4701.20f, 4715.80f, 4690.10f, 4655.40f,
                4670.90f, 4688.30f, 4710.60f, 4725.90f, 4740.10f,
                4735.50f, 4712.80f, 4695.40f, 4680.20f, 4705.60f,
                4730.40f, 4755.80f, 4742.10f, 4720.50f, 4698.90f,
                4685.20f, 4711.25f, 4672.01f, 4758.76f, 4676.42f
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
    }
}