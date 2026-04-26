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
    public class OnnxLinearInferenceBenchmarks : IDisposable
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private Sequential _overfit = null!;
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
                    .WithWarmupCount(10)
                    .WithIterationCount(50)
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

            _linear = new LinearLayer(InputSize, OutputSize);
            _overfit = new Sequential(_linear);

            _overfit.Eval();
            _overfit.PrepareInference(maxIntermediateElements: InputSize + OutputSize + 1024);

            _overfit.ForwardInference(_input, _overfitOutput);

            _onnxModelPath = Path.Combine(
                Path.GetTempPath(),
                $"overfit-linear-{Guid.NewGuid():N}.onnx");

            var weights = _linear.Weights.DataView
                .AsReadOnlySpan()
                .ToArray();

            var bias = _linear.Bias.DataView
                .AsReadOnlySpan()
                .ToArray();

            OnnxLinearModelWriter.WriteLinearGemmModel(
                _onnxModelPath,
                InputSize,
                OutputSize,
                weights,
                bias);

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
                [1, InputSize]);

            _onnxOutputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _onnxOutput.AsMemory(),
                [1, OutputSize]);

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
        }

        [Benchmark]
        public float Overfit_Linear_ZeroAlloc()
        {
            _overfit.ForwardInference(_input, _overfitOutput);
            return _overfitOutput[0];
        }

        [Benchmark]
        public float OnnxRuntime_Linear_TrueZeroAlloc()
        {
            _onnxSession.Run(
                _onnxRunOptions,
                _onnxInputNames,
                _onnxInputs,
                _onnxOutputNames,
                _onnxOutputs);

            return _onnxOutput[0];
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
                        $"Output mismatch at {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}");
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