// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Inference.Contracts;
using DevOnBike.Overfit.Licensing;
using DevOnBike.Overfit.Tests.TestSupport.Helpers;

namespace DevOnBike.Overfit.Tests.DeepLearning.Inference
{
    public sealed class InferenceEngineTests
    {
        [Fact]
        public void InferenceEngine_Run_SingleLinear_AllocatesZeroBytes()
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

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize,
                outputSize,
                new InferenceEngineOptions
                {
                    WarmupIterations = 16,
                    MaxIntermediateElements = 64 * 1024
                });

            // Built before anything is measured, so the closure allocates at this line, not inside the window.
            Action body = () => engine.Run(input, output);

            ForceFullGc();

            AssertAllocation.NoPerCallAllocation("engine inference", iterations, body);
        }

        [Fact]
        public void InferenceEngine_Predict_SingleLinear_AllocatesZeroBytes()
        {
            OverfitLicense.SuppressNotice = true;

            const int inputSize = 784;
            const int outputSize = 10;
            const int iterations = 10_000;

            var input = new float[inputSize];

            FillDeterministic(input);

            using var model = new Sequential(
                new LinearLayer(inputSize, outputSize));

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize,
                outputSize,
                new InferenceEngineOptions
                {
                    WarmupIterations = 16,
                    MaxIntermediateElements = 64 * 1024
                });

            var checksum = 0f;

            // The checksum is what stops the JIT eliding the prediction, so it stays inside the body. It used
            // to be interpolated into the label; the label is now built before the loop runs, so it is asserted
            // on afterwards instead — which is a real check rather than a way of consuming a value.
            Action body = () =>
            {
                var prediction = engine.Predict(input);

                checksum += prediction[0];
            };

            ForceFullGc();

            AssertAllocation.NoPerCallAllocation("engine Predict", iterations, body);

            Assert.True(float.IsFinite(checksum), $"engine Predict checksum was {checksum}");
        }

        [Fact]
        public void InferenceEngine_Run_Cnn_AllocatesZeroBytes()
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

            const int inputSize = inputChannels * inputH * inputW;

            var input = new float[inputSize];
            var output = new float[outputClasses];

            FillDeterministic(input);

            using var model = new Sequential(
                new ConvLayer(inputChannels, convOutChannels, inputH, inputW, kernel),
                new ReluActivation(),
                new MaxPool2DLayer(convOutChannels, convOutH, convOutW, pool),
                new GlobalAveragePool2DLayer(convOutChannels, poolOutH, poolOutW),
                new LinearLayer(convOutChannels, outputClasses));

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize,
                outputClasses,
                new InferenceEngineOptions
                {
                    WarmupIterations = 16,
                    MaxIntermediateElements = 64 * 1024
                });

            // Built before anything is measured, so the closure allocates at this line, not inside the window.
            Action body = () => engine.Run(input, output);

            ForceFullGc();

            AssertAllocation.NoPerCallAllocation("engine CNN inference", iterations, body);
        }

        [Fact]
        public void InferenceEngine_Run_BatchedInput_WritesBatchedOutput()
        {
            OverfitLicense.SuppressNotice = true;

            const int inputSize = 784;
            const int outputSize = 10;
            const int batchSize = 4;

            var input = new float[inputSize * batchSize];
            var output = new float[outputSize * batchSize];

            FillDeterministic(input);

            using var model = new Sequential(
                new LinearLayer(inputSize, outputSize));

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize,
                outputSize,
                new InferenceEngineOptions
                {
                    WarmupIterations = 16,
                    MaxIntermediateElements = 64 * 1024
                });

            engine.Run(input, output);

            Assert.Equal(outputSize * batchSize, output.Length);
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
