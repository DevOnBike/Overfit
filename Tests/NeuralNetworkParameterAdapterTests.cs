// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Evolutionary.Adapters;

namespace DevOnBike.Overfit.Tests
{
    public sealed class NeuralNetworkParameterAdapterTests
    {
        [Fact]
        public void Constructor_ExposesTotalParameterCount()
        {
            using var layer = new LinearLayer(4, 3);
            var adapter = new NeuralNetworkParameterAdapter(layer);

            // Linear(4 -> 3): weights 4*3 + biases 3 = 15 parameters.
            Assert.Equal(15, adapter.ParameterCount);
        }

        [Fact]
        public void WriteToVector_ThenReadFromVector_RoundTrips()
        {
            using var layer = new LinearLayer(4, 3);
            var adapter = new NeuralNetworkParameterAdapter(layer);

            var original = new float[adapter.ParameterCount];
            adapter.WriteToVector(original);

            var rewritten = new float[adapter.ParameterCount];
            adapter.ReadFromVector(original);
            adapter.WriteToVector(rewritten);

            Assert.Equal(original, rewritten);
        }

        [Fact]
        public void ReadFromVector_WritesCorrectValuesToParameters()
        {
            using var layer = new LinearLayer(4, 3);
            var adapter = new NeuralNetworkParameterAdapter(layer);

            var vector = new float[adapter.ParameterCount];
            for (var i = 0; i < vector.Length; i++)
            {
                vector[i] = i * 0.1f;
            }

            adapter.ReadFromVector(vector);

            var expectedWeights = vector.Take(12).ToArray();
            var expectedBiases = vector.Skip(12).Take(3).ToArray();

            var wSpan = layer.Weights.DataView.AsReadOnlySpan();
            for (var i = 0; i < expectedWeights.Length; i++)
            {
                Assert.Equal(expectedWeights[i], wSpan[i]);
            }

            var bSpan = layer.Bias.DataView.AsReadOnlySpan(); // Zmieniono z Biases na Bias
            for (var i = 0; i < expectedBiases.Length; i++)
            {
                Assert.Equal(expectedBiases[i], bSpan[i]);
            }
        }

        [Fact]
        public void ParameterCount_MatchesSumOfParameterTensorSizes_ForSequential()
        {
            // Sequential with two Linear layers: Linear(8->4) + ReLU + Linear(4->2).
            // Parameters: 8*4 + 4 + 4*2 + 2 = 32 + 4 + 8 + 2 = 46.
            using var seq = new Sequential(
                new LinearLayer(8, 4),
                new ReluActivation(),
                new LinearLayer(4, 2));

            var adapter = new NeuralNetworkParameterAdapter(seq);

            Assert.Equal(46, adapter.ParameterCount);
        }

        [Fact]
        public void IsAllocationStable_AfterWarmup()
        {
            using var layer = new LinearLayer(16, 8);
            var adapter = new NeuralNetworkParameterAdapter(layer);

            var buffer = new float[adapter.ParameterCount];
            adapter.WriteToVector(buffer);
            adapter.ReadFromVector(buffer); // warmup

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 5_000; i++)
            {
                adapter.WriteToVector(buffer);
                adapter.ReadFromVector(buffer);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            // Zero-allocation is expected for parameter reading/writing.
            Assert.Equal(0, allocated);
        }
    }
}