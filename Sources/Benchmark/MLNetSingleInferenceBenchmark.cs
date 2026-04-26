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
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;

namespace Benchmarks
{
    /// <summary>
    /// Single-inference latency comparison across:
    ///
    /// - Overfit InferenceEngine
    /// - ONNX Runtime direct preallocated OrtValue path
    /// - ML.NET PredictionEngine
    ///
    /// Model:
    ///
    ///     Linear(784, 256)
    ///     ReLU
    ///     Linear(256, 128)
    ///     ReLU
    ///     Linear(128, 10)
    ///
    /// Files:
    ///
    ///     benchmark_mlp3.bin
    ///     benchmark_mlp3.onnx
    ///
    /// The Overfit path must use InferenceEngine.Run(...), not model.Forward(...).
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class MLNetSingleInferenceBenchmark : IDisposable
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private const string OnnxPath = "benchmark_mlp3.onnx";
        private const string BinPath = "benchmark_mlp3.bin";

        // ~8k * several-us methods gives stable BDN iteration time.
        private const int OperationsPerInvoke = 32_768;

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

        // ML.NET
        private PredictionEngine<OnnxInput, OnnxOutput> _mlNetEngine = null!;
        private OnnxInput _mlNetReusableInput = null!;

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            if (!File.Exists(OnnxPath) || !File.Exists(BinPath))
            {
                throw new InvalidOperationException(
                    $"Missing {OnnxPath} or {BinPath}. Generate the model files first.");
            }

            _inputData = new float[InputSize];
            FillDeterministic(_inputData);

            SetupOverfit();
            SetupOnnxRuntime();
            SetupMLNet();

            // Warm up outside measured region.
            for (var i = 0; i < 512; i++)
            {
                _overfitEngine.Run(_inputData, _overfitOutput);

                _onnxSession.Run(
                    _onnxRunOptions,
                    _inputNames,
                    _ortInputValues,
                    _outputNames,
                    _ortOutputValues);

                _ = _mlNetEngine.Predict(_mlNetReusableInput);
            }

            AssertClose(_onnxOutputData, _overfitOutput, 1e-3f);
        }

        private void SetupOverfit()
        {
            _overfitOutput = new float[OutputSize];

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

            _overfitEngine.Run(_inputData, _overfitOutput);
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

            _onnxSession = new InferenceSession(OnnxPath, sessionOptions);

            _onnxOutputData = new float[OutputSize];

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

            _onnxSession.Run(
                _onnxRunOptions,
                _inputNames,
                _ortInputValues,
                _outputNames,
                _ortOutputValues);
        }

        private void SetupMLNet()
        {
            var ml = new MLContext(seed: 42);

            var emptyData = ml.Data.LoadFromEnumerable(Array.Empty<OnnxInput>());

            var pipeline = ml.Transforms.ApplyOnnxModel(
                modelFile: OnnxPath,
                outputColumnNames: ["output"],
                inputColumnNames: ["input"]);

            var transformer = pipeline.Fit(emptyData);

            _mlNetEngine = ml.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(
                transformer);

            _mlNetReusableInput = new OnnxInput
            {
                Input = _inputData
            };
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = OperationsPerInvoke)]
        public float OnnxRuntime_PreAllocated()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _onnxSession.Run(
                    _onnxRunOptions,
                    _inputNames,
                    _ortInputValues,
                    _outputNames,
                    _ortOutputValues);

                checksum += _onnxOutputData[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_InferenceEngine_ZeroAlloc()
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

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float MLNet_PredictionEngine_ReusedInput()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                var output = _mlNetEngine.Predict(_mlNetReusableInput);
                checksum += output.Output[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float MLNet_PredictionEngine_FreshInput()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                var input = new OnnxInput
                {
                    Input = _inputData
                };

                var output = _mlNetEngine.Predict(input);
                checksum += output.Output[0];
            }

            return checksum;
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

            _mlNetEngine?.Dispose();
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

        public sealed class OnnxInput
        {
            [VectorType(InputSize)]
            [ColumnName("input")]
            public float[] Input { get; set; } = null!;
        }

        public sealed class OnnxOutput
        {
            [VectorType(OutputSize)]
            [ColumnName("output")]
            public float[] Output { get; set; } = null!;
        }
    }
}