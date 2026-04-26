// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Licensing;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests
{
    public sealed class InferenceParityTests
    {
        [Fact]
        public void Sequential_SingleLinear_Inference_Matches_GraphForward()
        {
            OverfitLicense.SuppressNotice = true;

            const int inputSize = 784;
            const int outputSize = 10;

            var input = new float[inputSize];
            var inferenceOutput = new float[outputSize];
            var graphOutput = new float[outputSize];

            FillDeterministic(input);

            using var model = new Sequential(
                new LinearLayer(inputSize, outputSize));

            model.Eval();
            model.PrepareInference(maxIntermediateElements: inputSize + outputSize + 1024);

            model.ForwardInference(input, inferenceOutput);

            using var graph = new ComputationGraph();

            using var inputStorage = new TensorStorage<float>(inputSize, clearMemory: false);
            input.CopyTo(inputStorage.AsSpan());

            using var inputNode = new AutogradNode(
                inputStorage,
                new TensorShape(1, inputSize),
                requiresGrad: false);

            var outputNode = model.Forward(graph, inputNode);

            outputNode.DataView
                .AsReadOnlySpan()
                .Slice(0, outputSize)
                .CopyTo(graphOutput);

            AssertClose(graphOutput, inferenceOutput, tolerance: 1e-5f);

            graph.Reset();
        }

        [Fact]
        public void Sequential_MultiLayer_Inference_Matches_GraphForward()
        {
            OverfitLicense.SuppressNotice = true;

            const int inputSize = 784;
            const int hiddenSize = 128;
            const int outputSize = 10;

            var input = new float[inputSize];
            var inferenceOutput = new float[outputSize];
            var graphOutput = new float[outputSize];

            FillDeterministic(input);

            using var model = new Sequential(
                new LinearLayer(inputSize, hiddenSize),
                new ReluActivation(),
                new LinearLayer(hiddenSize, outputSize));

            model.Eval();
            model.PrepareInference(maxIntermediateElements: inputSize + hiddenSize + outputSize + 1024);

            model.ForwardInference(input, inferenceOutput);

            using var graph = new ComputationGraph();

            using var inputStorage = new TensorStorage<float>(inputSize, clearMemory: false);
            input.CopyTo(inputStorage.AsSpan());

            using var inputNode = new AutogradNode(
                inputStorage,
                new TensorShape(1, inputSize),
                requiresGrad: false);

            var outputNode = model.Forward(graph, inputNode);

            outputNode.DataView
                .AsReadOnlySpan()
                .Slice(0, outputSize)
                .CopyTo(graphOutput);

            AssertClose(graphOutput, inferenceOutput, tolerance: 1e-5f);

            graph.Reset();
        }

        [Fact]
        public void Sequential_Cnn_Inference_Matches_GraphForward()
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

            const int inputSize = inputChannels * inputH * inputW;

            var input = new float[inputSize];
            var inferenceOutput = new float[outputClasses];
            var graphOutput = new float[outputClasses];

            FillDeterministic(input);

            using var model = new Sequential(
                new ConvLayer(inputChannels, convOutChannels, inputH, inputW, kernel),
                new ReluActivation(),
                new MaxPool2DLayer(convOutChannels, convOutH, convOutW, pool),
                new GlobalAveragePool2DLayer(convOutChannels, poolOutH, poolOutW),
                new LinearLayer(convOutChannels, outputClasses));

            model.Eval();
            model.PrepareInference(maxIntermediateElements: 64 * 1024);

            model.ForwardInference(input, inferenceOutput);

            using var graph = new ComputationGraph();

            using var inputStorage = new TensorStorage<float>(inputSize, clearMemory: false);
            input.CopyTo(inputStorage.AsSpan());

            using var inputNode = new AutogradNode(
                inputStorage,
                new TensorShape(1, inputChannels, inputH, inputW),
                requiresGrad: false);

            var outputNode = model.Forward(graph, inputNode);

            outputNode.DataView
                .AsReadOnlySpan()
                .Slice(0, outputClasses)
                .CopyTo(graphOutput);

            AssertClose(graphOutput, inferenceOutput, tolerance: 1e-4f);

            graph.Reset();
        }

        private static void AssertClose(
            ReadOnlySpan<float> expected,
            ReadOnlySpan<float> actual,
            float tolerance)
        {
            Assert.Equal(expected.Length, actual.Length);

            for (var i = 0; i < expected.Length; i++)
            {
                var diff = MathF.Abs(expected[i] - actual[i]);

                Assert.True(
                    diff <= tolerance,
                    $"Mismatch at index {i}: expected {expected[i]}, actual {actual[i]}, diff {diff}, tolerance {tolerance}");
            }
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
    }
}