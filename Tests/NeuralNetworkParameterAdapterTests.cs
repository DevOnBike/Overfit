// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Evolutionary.Adapters;
using DevOnBike.Overfit.Tensors;

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
        public void ReadFromVector_ActuallyMutatesUnderlyingModuleParameters()
        {
            using var layer = new LinearLayer(4, 3);
            var adapter = new NeuralNetworkParameterAdapter(layer);

            // Overwrite with all-ones.
            var ones = new float[adapter.ParameterCount];
            ones.AsSpan().Fill(1f);
            adapter.ReadFromVector(ones);

            // Every weight and bias should now read back as 1.
            var weights = layer.Weights.DataView.AsReadOnlySpan();
            for (var i = 0; i < weights.Length; i++)
            {
                Assert.Equal(1f, weights[i], 5);
            }

            var biases = layer.Biases.DataView.AsReadOnlySpan();
            for (var i = 0; i < biases.Length; i++)
            {
                Assert.Equal(1f, biases[i], 5);
            }
        }

        [Fact]
        public void ReadFromVector_InvalidatesInferenceCache_SoNextForwardUsesNewWeights()
        {
            // Construct a layer, warm up the Eval-mode transposed-weight cache
            // by running one inference, then mutate the weights through the adapter
            // and confirm the next inference reflects the new weights.
            using var layer = new LinearLayer(2, 1);
            var adapter = new NeuralNetworkParameterAdapter(layer);

            layer.Eval();

            // Seed parameters: weights = [2, 3], bias = [0].
            // Output for input [1, 1] should be 1*2 + 1*3 + 0 = 5.
            var seeded = new float[adapter.ParameterCount];
            seeded[0] = 2f; // weight[0,0]
            seeded[1] = 3f; // weight[1,0]
            seeded[2] = 0f; // bias[0]
            adapter.ReadFromVector(seeded);

            var input = new float[] { 1f, 1f };
            var output = new float[1];
            layer.ForwardInference(input, output);

            Assert.Equal(5f, output[0], 4);

            // Mutate parameters through the adapter: weights = [10, 20], bias = [1].
            // Output for input [1, 1] should now be 1*10 + 1*20 + 1 = 31.
            // If the transposed-weight cache was not invalidated, ForwardInference
            // would still return 5 — this is the regression this test guards against.
            var mutated = new float[adapter.ParameterCount];
            mutated[0] = 10f;
            mutated[1] = 20f;
            mutated[2] = 1f;
            adapter.ReadFromVector(mutated);

            layer.ForwardInference(input, output);

            Assert.Equal(31f, output[0], 4);
        }

        [Fact]
        public void WriteToVector_ThrowsOnWrongLength()
        {
            using var layer = new LinearLayer(4, 3);
            var adapter = new NeuralNetworkParameterAdapter(layer);

            var tooShort = new float[adapter.ParameterCount - 1];
            Assert.Throws<ArgumentException>(() => adapter.WriteToVector(tooShort));
        }

        [Fact]
        public void ReadFromVector_ThrowsOnWrongLength()
        {
            using var layer = new LinearLayer(4, 3);
            var adapter = new NeuralNetworkParameterAdapter(layer);

            var tooLong = new float[adapter.ParameterCount + 1];
            Assert.Throws<ArgumentException>(() => adapter.ReadFromVector(tooLong));
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

            Assert.True(allocated <= 512, $"WriteToVector + ReadFromVector allocated {allocated} bytes.");
        }
    }
}