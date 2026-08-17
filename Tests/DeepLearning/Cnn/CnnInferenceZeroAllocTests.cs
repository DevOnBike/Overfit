// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Licensing;
using DevOnBike.Overfit.Tests.TestSupport.Helpers;

namespace DevOnBike.Overfit.Tests.DeepLearning.Cnn
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

            // Built here, before anything is measured: the closure allocates once, at this line, and the
            // warm-up runs through it so the delegate itself is JIT-warm by the time the window opens.
            Action body = () => model.ForwardInference(input, output);

            for (var i = 0; i < 32; i++)
            {
                body();
            }

            ForceFullGc();

            // The two-half overload, not the one-total one: when this fails it does not reproduce, so the run
            // that catches it has to say whether the bytes were front-loaded tier-up work or a real per-call
            // allocation. See XC-73.
            AssertAllocation.NoPerCallAllocation("CNN inference", iterations, body);
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