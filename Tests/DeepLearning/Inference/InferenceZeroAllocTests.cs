// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Licensing;

namespace DevOnBike.Overfit.Tests
{
    public sealed class InferenceZeroAllocTests
    {
        [Fact]
        public void Sequential_SingleLinear_Inference_AllocatesZeroBytes()
        {
            OverfitLicense.SuppressNotice = true;

            const int inputSize = 784;
            const int outputSize = 10;
            const int iterations = 10_000;

            var input = new float[inputSize];
            var output = new float[outputSize];

            FillDeterministic(input);

            using var model = new Sequential(
                new LinearLayer(inputSize, outputSize));

            model.Eval();
            model.PrepareInference(maxIntermediateElements: inputSize + outputSize + 1024);

            // Warmup: JIT + cache.
            for (var i = 0; i < 32; i++)
            {
                model.ForwardInference(input, output);
            }

            ForceFullGc();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < iterations; i++)
            {
                model.ForwardInference(input, output);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();
            var allocated = after - before;

            Assert.True(
                allocated == 0,
                $"Expected zero allocations, got {allocated} B over {iterations} inference calls.");
        }

        [Fact]
        public void Sequential_MultiLayer_Inference_AllocatesZeroBytes()
        {
            OverfitLicense.SuppressNotice = true;

            const int inputSize = 784;
            const int hiddenSize = 128;
            const int outputSize = 10;
            const int iterations = 10_000;

            var input = new float[inputSize];
            var output = new float[outputSize];

            FillDeterministic(input);

            using var model = new Sequential(
                new LinearLayer(inputSize, hiddenSize),
                new ReluActivation(),
                new LinearLayer(hiddenSize, outputSize));

            model.Eval();
            model.PrepareInference(maxIntermediateElements: inputSize + hiddenSize + outputSize + 1024);

            // Warmup: JIT + cache.
            for (var i = 0; i < 32; i++)
            {
                model.ForwardInference(input, output);
            }

            ForceFullGc();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < iterations; i++)
            {
                model.ForwardInference(input, output);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();
            var allocated = after - before;

            Assert.True(
                allocated == 0,
                $"Expected zero allocations, got {allocated} B over {iterations} inference calls.");
        }

        private static void FillDeterministic(float[] data)
        {
            var seed = 0x12345678u;

            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;

                var normalized = (seed & 0x00FFFFFF) / 16777216f;
                data[i] = normalized * 2f - 1f;
            }
        }

        private static void ForceFullGc()
        {
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Forced,
                blocking: true,
                compacting: true);

            GC.WaitForPendingFinalizers();

            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Forced,
                blocking: true,
                compacting: true);
        }
    }
}