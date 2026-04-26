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
    /// Batch scaling benchmark for a single Linear(784, 10) model.
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
    public class BatchScalingBenchmark : IDisposable
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private const string BinPath = "benchmark_model.bin";
        private const string OnnxPath = "benchmark_model.onnx";

        // Tuned to keep BenchmarkDotNet iteration time around or above 100 ms.
        private const int OverfitBatch1Operations = 524_288;
        private const int OverfitBatch16Operations = 32_768;
        private const int OverfitBatch64Operations = 8_192;
        private const int OverfitBatch256Operations = 4_096;

        private const int OnnxBatch1Operations = 65_536;
        private const int OnnxBatch16Operations = 65_536;
        private const int OnnxBatch64Operations = 16_384;
        private const int OnnxBatch256Operations = 8_192;

        private Sequential _overfitModel = null!;
        private InferenceEngine _overfitEngine = null!;
        private InferenceSession _onnxSession = null!;

        private BatchCase _batch1 = null!;
        private BatchCase _batch16 = null!;
        private BatchCase _batch64 = null!;
        private BatchCase _batch256 = null!;

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            if (!File.Exists(BinPath) || !File.Exists(OnnxPath))
            {
                throw new InvalidOperationException(
                    $"Missing benchmark files: {BinPath} and/or {OnnxPath}.");
            }

            SetupOverfit();
            SetupOnnxRuntime();

            _batch1 = CreateCase(1);
            _batch16 = CreateCase(16);
            _batch64 = CreateCase(64);
            _batch256 = CreateCase(256);

            WarmupAndVerify(_batch1);
            WarmupAndVerify(_batch16);
            WarmupAndVerify(_batch64);
            WarmupAndVerify(_batch256);
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
        }

        private BatchCase CreateCase(
            int batchSize)
        {
            var input = new float[batchSize * InputSize];
            var overfitOutput = new float[batchSize * OutputSize];
            var onnxOutput = new float[batchSize * OutputSize];

            FillDeterministic(input);

            var onnxInputValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                input.AsMemory(),
                new long[] { batchSize, InputSize });

            var onnxOutputValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                onnxOutput.AsMemory(),
                new long[] { batchSize, OutputSize });

            return new BatchCase(
                batchSize,
                input,
                overfitOutput,
                onnxOutput,
                onnxInputValue,
                onnxOutputValue);
        }

        private void WarmupAndVerify(
            BatchCase batchCase)
        {
            for (var i = 0; i < 128; i++)
            {
                _overfitEngine.Run(
                    batchCase.Input,
                    batchCase.OverfitOutput);

                RunOnnxOnce(batchCase);
            }

            AssertClose(
                batchCase.OnnxOutput,
                batchCase.OverfitOutput,
                tolerance: 1e-3f);
        }

        [Benchmark(OperationsPerInvoke = OverfitBatch1Operations)]
        public float Overfit_Batch1()
        {
            return RunOverfit(
                _batch1,
                OverfitBatch1Operations);
        }

        [Benchmark(OperationsPerInvoke = OnnxBatch1Operations)]
        public float OnnxRuntime_Batch1()
        {
            return RunOnnx(
                _batch1,
                OnnxBatch1Operations);
        }

        [Benchmark(OperationsPerInvoke = OverfitBatch16Operations)]
        public float Overfit_Batch16()
        {
            return RunOverfit(
                _batch16,
                OverfitBatch16Operations);
        }

        [Benchmark(OperationsPerInvoke = OnnxBatch16Operations)]
        public float OnnxRuntime_Batch16()
        {
            return RunOnnx(
                _batch16,
                OnnxBatch16Operations);
        }

        [Benchmark(OperationsPerInvoke = OverfitBatch64Operations)]
        public float Overfit_Batch64()
        {
            return RunOverfit(
                _batch64,
                OverfitBatch64Operations);
        }

        [Benchmark(OperationsPerInvoke = OnnxBatch64Operations)]
        public float OnnxRuntime_Batch64()
        {
            return RunOnnx(
                _batch64,
                OnnxBatch64Operations);
        }

        [Benchmark(OperationsPerInvoke = OverfitBatch256Operations)]
        public float Overfit_Batch256()
        {
            return RunOverfit(
                _batch256,
                OverfitBatch256Operations);
        }

        [Benchmark(OperationsPerInvoke = OnnxBatch256Operations)]
        public float OnnxRuntime_Batch256()
        {
            return RunOnnx(
                _batch256,
                OnnxBatch256Operations);
        }

        private float RunOverfit(
            BatchCase batchCase,
            int iterations)
        {
            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                _overfitEngine.Run(
                    batchCase.Input,
                    batchCase.OverfitOutput);

                checksum += batchCase.OverfitOutput[0];
            }

            return checksum;
        }

        private float RunOnnx(
            BatchCase batchCase,
            int iterations)
        {
            var checksum = 0f;

            for (var i = 0; i < iterations; i++)
            {
                RunOnnxOnce(batchCase);
                checksum += batchCase.OnnxOutput[0];
            }

            return checksum;
        }

        private void RunOnnxOnce(
            BatchCase batchCase)
        {
            _onnxSession.Run(
                batchCase.RunOptions,
                batchCase.InputNames,
                batchCase.InputValues,
                batchCase.OutputNames,
                batchCase.OutputValues);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _batch1?.Dispose();
            _batch16?.Dispose();
            _batch64?.Dispose();
            _batch256?.Dispose();

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

        private sealed class BatchCase : IDisposable
        {
            public BatchCase(
                int batchSize,
                float[] input,
                float[] overfitOutput,
                float[] onnxOutput,
                OrtValue onnxInputValue,
                OrtValue onnxOutputValue)
            {
                BatchSize = batchSize;
                Input = input;
                OverfitOutput = overfitOutput;
                OnnxOutput = onnxOutput;

                InputNames = new[] { "input" };
                OutputNames = new[] { "output" };

                InputValues = new[] { onnxInputValue };
                OutputValues = new[] { onnxOutputValue };

                RunOptions = new RunOptions();
            }

            public int BatchSize { get; }

            public float[] Input { get; }

            public float[] OverfitOutput { get; }

            public float[] OnnxOutput { get; }

            public string[] InputNames { get; }

            public string[] OutputNames { get; }

            public OrtValue[] InputValues { get; }

            public OrtValue[] OutputValues { get; }

            public RunOptions RunOptions { get; }

            public void Dispose()
            {
                InputValues[0].Dispose();
                OutputValues[0].Dispose();
                RunOptions.Dispose();
            }
        }
    }
}