using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Inference.Contracts;
using DevOnBike.Overfit.Licensing;

namespace DevOnBike.Overfit.Tests
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

            ForceFullGc();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < iterations; i++)
            {
                engine.Run(input, output);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();
            var allocated = after - before;

            Assert.True(
                allocated == 0,
                $"Expected zero allocations, got {allocated} B over {iterations} inference calls.");
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

            ForceFullGc();

            var before = GC.GetAllocatedBytesForCurrentThread();

            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                var prediction = engine.Predict(input);
                checksum += prediction[0];
            }

            var after = GC.GetAllocatedBytesForCurrentThread();
            var allocated = after - before;

            Assert.True(
                allocated == 0,
                $"Expected zero allocations, got {allocated} B over {iterations} Predict calls. Checksum={checksum}");
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

            ForceFullGc();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < iterations; i++)
            {
                engine.Run(input, output);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();
            var allocated = after - before;

            Assert.True(
                allocated == 0,
                $"Expected zero allocations, got {allocated} B over {iterations} CNN inference calls.");
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
