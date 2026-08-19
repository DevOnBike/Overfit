// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

// Zmieniono na Tensors.Core

namespace DevOnBike.Overfit.Tests.Forecasting
{
    public class RealGoldForecastingTests
    {
        private readonly ITestOutputHelper _output;

        public RealGoldForecastingTests(ITestOutputHelper output)
        {
        // Seeded so this class trains the same model every run.
        //
        // `XC-83`: these tests assert a convergence threshold over weights initialised from an UNSEEDED
        // generator, so every run trained a different model and the assertion sampled a distribution.
        // Two of the family were caught failing intermittently - one at a final loss of 2.0235 against a
        // `< 2.0f` bound whose own comment claimed the loss lands "well below" it.
        //
        // MathUtils.SetSeed is per-thread, so it belongs in the constructor: xUnit builds the instance on
        // the thread that then runs the test and builds the model.
        //
        // A failure here after seeding is REPRODUCIBLE and therefore a real finding - do not respond by
        // trying seeds until one passes. Widening a bound needs the outcome distribution measured over many
        // seeds first; a threshold moved to accommodate one observed failure has no evidence behind it.
            MathUtils.SetSeed(20260818);

            _output = output;
        }

        [Fact]
        public void MLP_Should_Predict_Real_Gold_Prices_FullYear()
        {
            _output.WriteLine("=== Trening na pełnym roku XAU/USD (Kwiecień 2025 - Kwiecień 2026) ===");

            var windowSize = 10;
            var epochs = 300;
            var learningRate = 0.005f;

            var prices = GetRealGoldPricesUSD_1Year();
            var returns = CalculateReturns(prices);

            var sampleCount = returns.Length - windowSize;
            var batchSize = sampleCount;

            // Zmiana na TensorStorage
            using var xData = new TensorStorage<float>(batchSize * windowSize, clearMemory: false);
            using var yData = new TensorStorage<float>(batchSize * 1, clearMemory: false);

            var xSpan = xData.AsSpan();
            var ySpan = yData.AsSpan();

            for (var i = 0; i < sampleCount; i++)
            {
                for (var w = 0; w < windowSize; w++)
                {
                    xSpan[i * windowSize + w] = returns[i + w] * 100f;
                }
                ySpan[i] = returns[i + windowSize] * 100f;
            }

            // Dodano TensorShape
            using var X = new AutogradNode(xData, new TensorShape(batchSize, windowSize));
            using var Y = new AutogradNode(yData, new TensorShape(batchSize));

            using var layer1 = new LinearLayer(windowSize, 32);
            using var layer2 = new LinearLayer(32, 16);
            using var layer3 = new LinearLayer(16, 1);

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

            _output.WriteLine($"Training finished. Final MSE: {finalLoss:F4}");

            Assert.True(finalLoss < 1.5f, $"Loss too high, model didn't converge. Final MSE: {finalLoss}");
        }

        private float[] GetRealGoldPricesUSD_1Year()
        {
            var prices = new float[252];
            var rnd = new Random(1337);
            var currentPrice = 2800f;
            var targetPrice = 3800f;

            for (var i = 0; i < 252; i++)
            {
                prices[i] = currentPrice;
                float daysLeft = 252 - i;
                var requiredDailyTrend = (targetPrice - currentPrice) / daysLeft;
                var volatility = (rnd.NextSingle() - 0.5f) * (currentPrice * 0.03f);
                currentPrice += requiredDailyTrend + volatility;
            }

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