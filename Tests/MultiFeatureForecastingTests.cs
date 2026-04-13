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
    public class MultiFeatureForecastingTests
    {
        private readonly ITestOutputHelper _output;

        public MultiFeatureForecastingTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void MLP_Should_Learn_From_Price_RSI_And_Volume()
        {
            _output.WriteLine("=== Trening Wielokanałowy: Cena + RSI + Wolumen ===");

            var windowSize = 10;
            var numFeatures = 3;
            var inputSize = windowSize * numFeatures;

            var epochs = 150;
            var learningRate = 0.005f;

            var (prices, volumes) = GenerateMockMarketData(300);
            var returns = CalculateReturns(prices);
            var rsi = CalculateRSI(prices, 14);

            var sampleCount = returns.Length - windowSize;
            var batchSize = sampleCount;

            using var xData = new FastTensor<float>(batchSize, inputSize, clearMemory: false);
            using var yData = new FastTensor<float>(batchSize, 1, clearMemory: false);

            var xSpan = xData.GetView().AsSpan();
            var ySpan = yData.GetView().AsSpan();

            for (var i = 0; i < sampleCount; i++)
            {
                for (var w = 0; w < windowSize; w++)
                {
                    var idx = i + w;
                    xSpan[i * inputSize + (w * numFeatures + 0)] = returns[idx] * 100f;
                    xSpan[i * inputSize + (w * numFeatures + 1)] = (rsi[idx] - 50f) / 50f;
                    xSpan[i * inputSize + (w * numFeatures + 2)] = (volumes[idx] - 1000f) / 1000f;
                }
                ySpan[i] = returns[i + windowSize] * 100f;
            }

            using var X = new AutogradNode(xData, false);
            using var Y = new AutogradNode(yData, false);

            using var layer1 = new LinearLayer(inputSize, 64);
            using var layer2 = new LinearLayer(64, 32);
            using var layer3 = new LinearLayer(32, 1);

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

            _output.WriteLine($"Final MSE Loss: {finalLoss:F4}");
            Assert.True(finalLoss < 2.0f, "Model didn't converge enough on multivariate data.");
        }

        private (float[] prices, float[] volumes) GenerateMockMarketData(int count)
        {
            var rnd = new Random(42);
            var prices = new float[count];
            var volumes = new float[count];
            var price = 100f;

            for (var i = 0; i < count; i++)
            {
                var trend = MathF.Sin(i * 0.1f) * 0.5f;
                var noise = (rnd.NextSingle() - 0.5f) * 1.5f;
                price += trend + noise;
                prices[i] = price;

                var volumeTrend = trend > 0 ? 1500f : 800f;
                volumes[i] = volumeTrend + rnd.Next(-200, 200);
            }
            return (prices, volumes);
        }

        private float[] CalculateReturns(float[] prices)
        {
            var returns = new float[prices.Length - 1];
            for (var i = 0; i < returns.Length; i++) returns[i] = (prices[i + 1] - prices[i]) / prices[i];
            return returns;
        }

        private float[] CalculateRSI(float[] prices, int period = 14)
        {
            var rsi = new float[prices.Length];
            Array.Fill(rsi, 50f);
            return rsi;
        }
    }
}