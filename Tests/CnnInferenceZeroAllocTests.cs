// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Licensing;

namespace DevOnBike.Overfit.Tests
{
    public sealed class CnnInferenceZeroAllocTests
    {
        [Fact]
        public void Sequential_Cnn_Inference_AllocatesZeroBytes()
        {
            OverfitLicense.SuppressNotice = true;

            const int inputChannels = 1;
            const int inputH = 28;
            const int inputW = 28;

            const int convOutChannels = 8;
            const int kernel = 3;

            const int convOutH = inputH - kernel + 1;
            const int convOutW = inputW - kernel + 1;

            const int pool = 2;
            const int poolOutH = convOutH / pool;
            const int poolOutW = convOutW / pool;

            const int outputClasses = 10;
            const int iterations = 10_000;

            var input = new float[inputChannels * inputH * inputW];
            var output = new float[outputClasses];

            FillDeterministic(input);

            using var model = new Sequential(
                new ConvLayer(inputChannels, convOutChannels, inputH, inputW, kernel),
                new ReluActivation(),
                new MaxPool2DLayer(convOutChannels, convOutH, convOutW, pool),
                new GlobalAveragePool2DLayer(convOutChannels, poolOutH, poolOutW),
                new LinearLayer(convOutChannels, outputClasses));

            model.Eval();
            model.PrepareInference(maxIntermediateElements: 64 * 1024);

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
                $"Expected zero allocations, got {allocated} B over {iterations} CNN inference calls.");
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