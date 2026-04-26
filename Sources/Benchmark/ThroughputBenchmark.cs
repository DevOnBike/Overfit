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
    /// Single-sample throughput benchmark.
    ///
    /// This benchmark verifies the current inference architecture:
    ///
    /// - Overfit uses InferenceEngine.Run(...)
    /// - ONNX Runtime uses preallocated OrtValue input/output buffers
    /// - no AutogradNode / ComputationGraph / model.Forward(...) path is used
    ///
    /// Required files:
    ///
    /// - benchmark_model.bin
    /// - benchmark_model.onnx
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class ThroughputBenchmark : IDisposable
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        // One logical throughput block.
        private const int IterationsPerRound = 10_000;

        // BDN needs enough total work per invocation.
        // 10_000 * 64 gives stable iteration time while still preserving the 10k-throughput scenario.
        private const int RoundsPerInvoke = 64;
        private const int OperationsPerInvoke = IterationsPerRound * RoundsPerInvoke;

        private const string BinPath = "benchmark_model.bin";
        private const string OnnxPath = "benchmark_model.onnx";

        private float[] _inputData = null!;

        // Overfit
        private Sequential _overfitModel = null!;
        private InferenceEngine _overfitEngine = null!;
        private float[] _overfitOutput = null!;

        // ONNX Runtime
        private InferenceSession _onnxSession = null!;
        private float[] _onnxOutputData = null!;
        private OrtValue _onnxInputValue = null!;
        private OrtValue _onnxOutputValue = null!;
        private RunOptions _onnxRunOptions = null!;
        private string[] _inputNames = null!;
        private string[] _outputNames = null!;
        private OrtValue[] _ortInputValues = null!;
        private OrtValue[] _ortOutputValues = null!;

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            if (!File.Exists(BinPath) || !File.Exists(OnnxPath))
            {
                throw new InvalidOperationException(
                    $"Missing benchmark files: {BinPath} and/or {OnnxPath}.");
            }

            _inputData = new float[InputSize];
            _overfitOutput = new float[OutputSize];
            _onnxOutputData = new float[OutputSize];

            FillDeterministic(_inputData);

            SetupOverfit();
            SetupOnnxRuntime();

            for (var i = 0; i < 512; i++)
            {
                _overfitEngine.Run(
                    _inputData,
                    _overfitOutput);

                RunOnnxOnce();
            }

            AssertClose(
                _onnxOutputData,
                _overfitOutput,
                tolerance: 1e-3f);
        }

        private void SetupOverfit()
        {
            _overfitModel = new Sequential(
                new LinearLayer(InputSize, OutputSize));

            _overfitModel.Load(BinPath);
            _overfitModel.Eval();

            _overfitEngine = InferenceEngine.FromSequential(
                _overfitModel,
                inputSize: InputSize,
                outputSize: OutputSize,
                new InferenceEngineOptions
                {
                    WarmupIterations = 32,
                    MaxIntermediateElements = 64 * 1024,
                    ValidateFiniteInput = false,
                    DisposeModelWithEngine = false
                });

            _overfitEngine.Run(
                _inputData,
                _overfitOutput);
        }

        private void SetupOnnxRuntime()
        {
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
                OnnxPath,
                sessionOptions);

            _onnxInputValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _inputData.AsMemory(),
                new long[] { 1, InputSize });

            _onnxOutputValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _onnxOutputData.AsMemory(),
                new long[] { 1, OutputSize });

            _onnxRunOptions = new RunOptions();

            _inputNames = new[] { "input" };
            _outputNames = new[] { "output" };

            _ortInputValues = new[] { _onnxInputValue };
            _ortOutputValues = new[] { _onnxOutputValue };

            RunOnnxOnce();
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = OperationsPerInvoke)]
        public float OnnxRuntime_10k_PreAllocated()
        {
            var checksum = 0f;

            for (var round = 0; round < RoundsPerInvoke; round++)
            {
                for (var i = 0; i < IterationsPerRound; i++)
                {
                    RunOnnxOnce();
                    checksum += _onnxOutputData[0];
                }
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_10k_InferenceEngine_ZeroAlloc()
        {
            var checksum = 0f;

            for (var round = 0; round < RoundsPerInvoke; round++)
            {
                for (var i = 0; i < IterationsPerRound; i++)
                {
                    _overfitEngine.Run(
                        _inputData,
                        _overfitOutput);

                    checksum += _overfitOutput[0];
                }
            }

            return checksum;
        }

        private void RunOnnxOnce()
        {
            _onnxSession.Run(
                _onnxRunOptions,
                _inputNames,
                _ortInputValues,
                _outputNames,
                _ortOutputValues);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxInputValue?.Dispose();
            _onnxOutputValue?.Dispose();
            _onnxRunOptions?.Dispose();
            _onnxSession?.Dispose();

            _overfitEngine?.Dispose();
            _overfitModel?.Dispose();
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
            float[] data)
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