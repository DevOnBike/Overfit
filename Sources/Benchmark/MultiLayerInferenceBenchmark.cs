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
    /// 3-layer MLP inference benchmark:
    ///
    ///     Linear(784, 256)
    ///     ReLU
    ///     Linear(256, 128)
    ///     ReLU
    ///     Linear(128, 10)
    ///
    /// This benchmark verifies the current inference path:
    ///
    /// - Overfit uses InferenceEngine.Run(...)
    /// - ONNX Runtime uses preallocated OrtValue input/output buffers
    /// - no AutogradNode / ComputationGraph / model.Forward(...) path is used
    ///
    /// Required files:
    ///
    /// - benchmark_mlp3.bin
    /// - benchmark_mlp3.onnx
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class MultiLayerInferenceBenchmark : IDisposable
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private const string OnnxPath = "benchmark_mlp3.onnx";
        private const string BinPath = "benchmark_mlp3.bin";

        // ~8-12 us * 16k = >100 ms per BDN iteration.
        private const int OperationsPerInvoke = 16_384;

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

            if (!File.Exists(OnnxPath) || !File.Exists(BinPath))
            {
                throw new InvalidOperationException(
                    $"Missing {OnnxPath} or {BinPath}. Generate benchmark_mlp3 files first.");
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
                new LinearLayer(InputSize, 256),
                new ReluActivation(),
                new LinearLayer(256, 128),
                new ReluActivation(),
                new LinearLayer(128, OutputSize));

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
                [1, InputSize]);

            _onnxOutputValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _onnxOutputData.AsMemory(),
                [1, OutputSize]);

            _onnxRunOptions = new RunOptions();

            _inputNames = ["input"];
            _outputNames = ["output"];

            _ortInputValues = [_onnxInputValue];
            _ortOutputValues = [_onnxOutputValue];

            RunOnnxOnce();
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = OperationsPerInvoke)]
        public float OnnxRuntime_3Layer_PreAllocated()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                RunOnnxOnce();
                checksum += _onnxOutputData[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_3Layer_InferenceEngine_ZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _overfitEngine.Run(
                    _inputData,
                    _overfitOutput);

                checksum += _overfitOutput[0];
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