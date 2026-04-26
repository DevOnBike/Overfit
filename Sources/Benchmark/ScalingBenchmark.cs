// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Inference.Contracts;
using DevOnBike.Overfit.Licensing;
using Microsoft.ML.OnnxRuntime;

namespace Benchmarks
{
    /// <summary>
    /// Evaluates single-layer inference scaling across input sizes:
    ///
    ///     Linear(64, 10)
    ///     Linear(784, 10)
    ///     Linear(4096, 10)
    ///
    /// This benchmark verifies the current inference architecture:
    ///
    /// - Overfit uses InferenceEngine.Run(...)
    /// - ONNX Runtime uses preallocated OrtValue input/output buffers
    /// - no AutogradNode / ComputationGraph / model.Forward(...) path is used
    ///
    /// Required files:
    ///
    /// - benchmark_small.bin / benchmark_small.onnx
    /// - benchmark_medium.bin / benchmark_medium.onnx
    /// - benchmark_large.bin / benchmark_large.onnx
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class ScalingBenchmark : IDisposable
    {
        private const int OutputSize = 10;

        private const int SmallInputSize = 64;
        private const int MediumInputSize = 784;
        private const int LargeInputSize = 4096;

        private const string SmallBinPath = "benchmark_small.bin";
        private const string MediumBinPath = "benchmark_medium.bin";
        private const string LargeBinPath = "benchmark_large.bin";

        private const string SmallOnnxPath = "benchmark_small.onnx";
        private const string MediumOnnxPath = "benchmark_medium.onnx";
        private const string LargeOnnxPath = "benchmark_large.onnx";

        // Tuned to keep BenchmarkDotNet iteration time around or above 100 ms.
        private const int OverfitSmallOperations = 2_097_152;
        private const int OverfitMediumOperations = 524_288;
        private const int OverfitLargeOperations = 131_072;

        private const int OnnxSmallOperations = 131_072;
        private const int OnnxMediumOperations = 65_536;
        private const int OnnxLargeOperations = 65_536;

        private BenchmarkCase _small = null!;
        private BenchmarkCase _medium = null!;
        private BenchmarkCase _large = null!;

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            EnsureFileExists(SmallBinPath);
            EnsureFileExists(MediumBinPath);
            EnsureFileExists(LargeBinPath);

            EnsureFileExists(SmallOnnxPath);
            EnsureFileExists(MediumOnnxPath);
            EnsureFileExists(LargeOnnxPath);

            _small = new BenchmarkCase(
                inputSize: SmallInputSize,
                outputSize: OutputSize,
                binPath: SmallBinPath,
                onnxPath: SmallOnnxPath,
                seedOffset: 1);

            _medium = new BenchmarkCase(
                inputSize: MediumInputSize,
                outputSize: OutputSize,
                binPath: MediumBinPath,
                onnxPath: MediumOnnxPath,
                seedOffset: 2);

            _large = new BenchmarkCase(
                inputSize: LargeInputSize,
                outputSize: OutputSize,
                binPath: LargeBinPath,
                onnxPath: LargeOnnxPath,
                seedOffset: 3);

            WarmupAndVerify(_small);
            WarmupAndVerify(_medium);
            WarmupAndVerify(_large);
        }

        private static void WarmupAndVerify(
            BenchmarkCase benchmarkCase)
        {
            for (var i = 0; i < 512; i++)
            {
                benchmarkCase.RunOverfitOnce();
                benchmarkCase.RunOnnxOnce();
            }

            AssertClose(
                benchmarkCase.OnnxOutput,
                benchmarkCase.OverfitOutput,
                tolerance: 1e-3f);
        }

        [Benchmark(OperationsPerInvoke = OverfitSmallOperations)]
        public float Overfit_64()
        {
            return RunOverfit(
                _small,
                OverfitSmallOperations);
        }

        [Benchmark(OperationsPerInvoke = OnnxSmallOperations)]
        public float Onnx_64()
        {
            return RunOnnx(
                _small,
                OnnxSmallOperations);
        }

        [Benchmark(OperationsPerInvoke = OverfitMediumOperations)]
        public float Overfit_784()
        {
            return RunOverfit(
                _medium,
                OverfitMediumOperations);
        }

        [Benchmark(OperationsPerInvoke = OnnxMediumOperations)]
        public float Onnx_784()
        {
            return RunOnnx(
                _medium,
                OnnxMediumOperations);
        }

        [Benchmark(OperationsPerInvoke = OverfitLargeOperations)]
        public float Overfit_4096()
        {
            return RunOverfit(
                _large,
                OverfitLargeOperations);
        }

        [Benchmark(OperationsPerInvoke = OnnxLargeOperations)]
        public float Onnx_4096()
        {
            return RunOnnx(
                _large,
                OnnxLargeOperations);
        }

        private static float RunOverfit(
            BenchmarkCase benchmarkCase,
            int iterations)
        {
            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                benchmarkCase.RunOverfitOnce();
                checksum += benchmarkCase.OverfitOutput[0];
            }

            return checksum;
        }

        private static float RunOnnx(
            BenchmarkCase benchmarkCase,
            int iterations)
        {
            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                benchmarkCase.RunOnnxOnce();
                checksum += benchmarkCase.OnnxOutput[0];
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _small?.Dispose();
            _medium?.Dispose();
            _large?.Dispose();
        }

        public void Dispose()
        {
            Cleanup();
        }

        private static void EnsureFileExists(
            string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Missing benchmark file: {path}",
                    path);
            }
        }

        private static void AssertClose(
            ReadOnlySpan<float> expected,
            ReadOnlySpan<float> actual,
            float tolerance)
        {
            if (expected.Length != actual.Length)
            {
                throw new InvalidOperationException(
                    $"Output length mismatch: expected={expected.Length}, actual={actual.Length}");
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

        private static void FillDeterministic(
            float[] data,
            int seedOffset)
        {
            var seed = 0x12345678u + (uint)(seedOffset * 7919);

            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;

                var normalized = (seed & 0x00FFFFFF) / 16777216f;
                data[i] = normalized * 2f - 1f;
            }
        }

        private sealed class BenchmarkCase : IDisposable
        {
            private readonly Sequential _overfitModel;
            private readonly InferenceEngine _overfitEngine;

            private readonly InferenceSession _onnxSession;
            private readonly OrtValue _onnxInputValue;
            private readonly OrtValue _onnxOutputValue;
            private readonly RunOptions _onnxRunOptions;

            private readonly string[] _inputNames;
            private readonly string[] _outputNames;
            private readonly OrtValue[] _inputValues;
            private readonly OrtValue[] _outputValues;

            public BenchmarkCase(
                int inputSize,
                int outputSize,
                string binPath,
                string onnxPath,
                int seedOffset)
            {
                InputSize = inputSize;
                OutputSize = outputSize;

                Input = new float[inputSize];
                OverfitOutput = new float[outputSize];
                OnnxOutput = new float[outputSize];

                FillDeterministic(
                    Input,
                    seedOffset);

                _overfitModel = new Sequential(
                    new LinearLayer(inputSize, outputSize));

                _overfitModel.Load(binPath);
                _overfitModel.Eval();

                _overfitEngine = InferenceEngine.FromSequential(
                    _overfitModel,
                    inputSize: inputSize,
                    outputSize: outputSize,
                    new InferenceEngineOptions
                    {
                        WarmupIterations = 32,
                        MaxIntermediateElements = 64 * 1024,
                        ValidateFiniteInput = false,
                        DisposeModelWithEngine = false
                    });

                var sessionOptions = new SessionOptions
                {
                    EnableCpuMemArena = true,
                    EnableMemoryPattern = true,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    InterOpNumThreads = 1,
                    IntraOpNumThreads = 1
                };

                _onnxSession = new InferenceSession(
                    onnxPath,
                    sessionOptions);

                _onnxInputValue = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance,
                    Input.AsMemory(),
                    new long[] { 1, inputSize });

                _onnxOutputValue = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance,
                    OnnxOutput.AsMemory(),
                    new long[] { 1, outputSize });

                _onnxRunOptions = new RunOptions();

                _inputNames = new[] { "input" };
                _outputNames = new[] { "output" };

                _inputValues = new[] { _onnxInputValue };
                _outputValues = new[] { _onnxOutputValue };

                RunOverfitOnce();
                RunOnnxOnce();
            }

            public int InputSize { get; }

            public int OutputSize { get; }

            public float[] Input { get; }

            public float[] OverfitOutput { get; }

            public float[] OnnxOutput { get; }

            public void RunOverfitOnce()
            {
                _overfitEngine.Run(
                    Input,
                    OverfitOutput);
            }

            public void RunOnnxOnce()
            {
                _onnxSession.Run(
                    _onnxRunOptions,
                    _inputNames,
                    _inputValues,
                    _outputNames,
                    _outputValues);
            }

            public void Dispose()
            {
                _onnxInputValue.Dispose();
                _onnxOutputValue.Dispose();
                _onnxRunOptions.Dispose();
                _onnxSession.Dispose();

                _overfitEngine.Dispose();
                _overfitModel.Dispose();
            }
        }
    }
}