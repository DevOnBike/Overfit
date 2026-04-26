// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Licensing;
using Microsoft.ML.OnnxRuntime;

namespace Benchmarks
{
    [MemoryDiagnoser]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [RankColumn]
    [Config(typeof(Config))]
    public class OnnxCnnInferenceBenchmarks : IDisposable
    {
        private const int InputChannels = 1;
        private const int InputH = 28;
        private const int InputW = 28;

        private const int ConvOutChannels = 8;
        private const int Kernel = 3;

        private const int ConvOutH = InputH - Kernel + 1;
        private const int ConvOutW = InputW - Kernel + 1;

        private const int Pool = 2;
        private const int PoolOutH = ConvOutH / Pool;
        private const int PoolOutW = ConvOutW / Pool;

        private const int OutputClasses = 10;

        private const int InputSize = InputChannels * InputH * InputW;
        private const int OutputSize = OutputClasses;

        private const int OperationsPerInvoke = 32_768;

        private Sequential _overfit = null!;
        private ConvLayer _conv = null!;
        private LinearLayer _linear = null!;

        private float[] _input = null!;
        private float[] _overfitOutput = null!;
        private float[] _onnxOutput = null!;

        private string _onnxModelPath = null!;

        private InferenceSession _onnxSession = null!;
        private RunOptions _onnxRunOptions = null!;

        private string[] _onnxInputNames = null!;
        private string[] _onnxOutputNames = null!;

        private OrtValue _onnxInputOrtValue = null!;
        private OrtValue _onnxOutputOrtValue = null!;

        private OrtValue[] _onnxInputs = null!;
        private OrtValue[] _onnxOutputs = null!;

        private class Config : ManualConfig
        {
            public Config()
            {
                AddJob(Job.Default
                    .WithWarmupCount(5)
                    .WithIterationCount(20)
                    .WithInvocationCount(1)
                    .WithUnrollFactor(1));

                AddDiagnoser(MemoryDiagnoser.Default);
            }
        }

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            _input = new float[InputSize];
            _overfitOutput = new float[OutputSize];
            _onnxOutput = new float[OutputSize];

            FillDeterministic(_input);

            _conv = new ConvLayer(
                InputChannels,
                ConvOutChannels,
                InputH,
                InputW,
                Kernel);

            _linear = new LinearLayer(
                ConvOutChannels,
                OutputClasses);

            _overfit = new Sequential(
                _conv,
                new ReluActivation(),
                new MaxPool2DLayer(ConvOutChannels, ConvOutH, ConvOutW, Pool),
                new GlobalAveragePool2DLayer(ConvOutChannels, PoolOutH, PoolOutW),
                _linear);

            _overfit.Eval();
            _overfit.PrepareInference(maxIntermediateElements: 64 * 1024);

            _overfit.ForwardInference(_input, _overfitOutput);

            _onnxModelPath = Path.Combine(
                Path.GetTempPath(),
                $"overfit-cnn-{Guid.NewGuid():N}.onnx");

            var convKernels = _conv.Kernels.DataView
                .AsReadOnlySpan()
                .ToArray();

            var linearWeights = _linear.Weights.DataView
                .AsReadOnlySpan()
                .ToArray();

            var linearBias = _linear.Bias.DataView
                .AsReadOnlySpan()
                .ToArray();

            OnnxCnnModelWriter.WriteConvReluPoolGapLinearModel(
                _onnxModelPath,
                InputChannels,
                InputH,
                InputW,
                ConvOutChannels,
                Kernel,
                Pool,
                OutputClasses,
                convKernels,
                linearWeights,
                linearBias);

            var sessionOptions = new SessionOptions
            {
                EnableCpuMemArena = true,
                EnableMemoryPattern = true,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                InterOpNumThreads = 1,
                IntraOpNumThreads = 1
            };

            _onnxSession = new InferenceSession(_onnxModelPath, sessionOptions);

            _onnxInputNames = [_onnxSession.InputMetadata.Keys.First()];
            _onnxOutputNames = [_onnxSession.OutputMetadata.Keys.First()];

            _onnxRunOptions = new RunOptions();

            _onnxInputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _input.AsMemory(),
                [1, InputChannels, InputH, InputW]);

            _onnxOutputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _onnxOutput.AsMemory(),
                [1, OutputClasses]);

            _onnxInputs = [_onnxInputOrtValue];
            _onnxOutputs = [_onnxOutputOrtValue];

            _onnxSession.Run(
                _onnxRunOptions,
                _onnxInputNames,
                _onnxInputs,
                _onnxOutputNames,
                _onnxOutputs);

            _overfit.ForwardInference(_input, _overfitOutput);

            AssertClose(_overfitOutput, _onnxOutput, tolerance: 1e-4f);

            // Extra warmup outside measured region.
            for (var i = 0; i < 256; i++)
            {
                _overfit.ForwardInference(_input, _overfitOutput);

                _onnxSession.Run(
                    _onnxRunOptions,
                    _onnxInputNames,
                    _onnxInputs,
                    _onnxOutputNames,
                    _onnxOutputs);
            }
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_Cnn_ZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _overfit.ForwardInference(_input, _overfitOutput);
                checksum += _overfitOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float OnnxRuntime_Cnn_TrueZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _onnxSession.Run(
                    _onnxRunOptions,
                    _onnxInputNames,
                    _onnxInputs,
                    _onnxOutputNames,
                    _onnxOutputs);

                checksum += _onnxOutput[0];
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxInputOrtValue?.Dispose();
            _onnxOutputOrtValue?.Dispose();
            _onnxRunOptions?.Dispose();
            _onnxSession?.Dispose();

            _overfit?.Dispose();

            if (!string.IsNullOrWhiteSpace(_onnxModelPath) &&
                File.Exists(_onnxModelPath))
            {
                try
                {
                    File.Delete(_onnxModelPath);
                }
                catch
                {
                    // Benchmark cleanup only.
                }
            }
        }

        public void Dispose()
        {
            Cleanup();
        }

        private static void AssertClose(
            ReadOnlySpan<float> expected,
            ReadOnlySpan<float> actual,
            float tolerance)
        {
            if (expected.Length != actual.Length)
            {
                throw new InvalidOperationException(
                    $"Output length mismatch: expected {expected.Length}, actual {actual.Length}");
            }

            for (var i = 0; i < expected.Length; i++)
            {
                var diff = MathF.Abs(expected[i] - actual[i]);

                if (diff > tolerance)
                {
                    throw new InvalidOperationException(
                        $"Output mismatch at {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}, tolerance={tolerance}");
                }
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