// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
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
    /// Analyzes single-inference tail latency distribution.
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
    ///
    /// BenchmarkDotNet reports time per inference because OperationsPerInvoke = TotalCalls.
    /// Tail latency percentiles are printed in GlobalCleanup from the last measured run.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class TailLatencyBenchmark : IDisposable
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private const string BinPath = "benchmark_model.bin";
        private const string OnnxPath = "benchmark_model.onnx";

        // Large enough to avoid BenchmarkDotNet MinIterationTime warnings even for
        // sub-microsecond Overfit single-linear inference.
        private const int TotalCalls = 500_000;

        private float[] _inputData = null!;

        // Overfit
        private Sequential _overfitModel = null!;
        private InferenceEngine _overfitEngine = null!;
        private float[] _overfitOutput = null!;
        private long[] _overfitLatencies = null!;

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
        private long[] _onnxLatencies = null!;

        private float _overfitChecksum;
        private float _onnxChecksum;

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

            _overfitLatencies = new long[TotalCalls];
            _onnxLatencies = new long[TotalCalls];

            FillDeterministic(_inputData);

            SetupOverfit();
            SetupOnnxRuntime();

            for (var i = 0; i < 10_000; i++)
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

            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
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

        [Benchmark(Baseline = true, OperationsPerInvoke = TotalCalls)]
        public float OnnxRuntime_LatencyProfile()
        {
            var checksum = 0f;

            for (var i = 0; i < TotalCalls; i++)
            {
                var start = Stopwatch.GetTimestamp();

                RunOnnxOnce();

                _onnxLatencies[i] = Stopwatch.GetTimestamp() - start;
                checksum += _onnxOutputData[0];
            }

            _onnxChecksum = checksum;

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = TotalCalls)]
        public float Overfit_LatencyProfile()
        {
            var checksum = 0f;

            for (var i = 0; i < TotalCalls; i++)
            {
                var start = Stopwatch.GetTimestamp();

                _overfitEngine.Run(
                    _inputData,
                    _overfitOutput);

                _overfitLatencies[i] = Stopwatch.GetTimestamp() - start;
                checksum += _overfitOutput[0];
            }

            _overfitChecksum = checksum;

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
            if (_onnxLatencies is not null)
            {
                PrintLatencyReport(
                    "ONNX Runtime",
                    _onnxLatencies,
                    _onnxChecksum);
            }

            if (_overfitLatencies is not null)
            {
                PrintLatencyReport(
                    "Overfit",
                    _overfitLatencies,
                    _overfitChecksum);
            }

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

        private static void PrintLatencyReport(
            string name,
            long[] latencies,
            float checksum)
        {
            var sorted = new long[latencies.Length];
            latencies.AsSpan().CopyTo(sorted);

            Array.Sort(sorted);

            var p50 = TicksToMicroseconds(sorted[(int)(TotalCalls * 0.50)]);
            var p90 = TicksToMicroseconds(sorted[(int)(TotalCalls * 0.90)]);
            var p95 = TicksToMicroseconds(sorted[(int)(TotalCalls * 0.95)]);
            var p99 = TicksToMicroseconds(sorted[(int)(TotalCalls * 0.99)]);
            var p999 = TicksToMicroseconds(sorted[(int)(TotalCalls * 0.999)]);
            var max = TicksToMicroseconds(sorted[TotalCalls - 1]);

            var jitter = p999 / Math.Max(p50, 0.01);

            Console.WriteLine();
            Console.WriteLine("  +------------------------------------------------+");
            Console.WriteLine($"  |  {name,-44}  |");
            Console.WriteLine("  +------------------------------------------------+");
            Console.WriteLine($"  |  P50:       {p50,10:F2} us                       |");
            Console.WriteLine($"  |  P90:       {p90,10:F2} us                       |");
            Console.WriteLine($"  |  P95:       {p95,10:F2} us                       |");
            Console.WriteLine($"  |  P99:       {p99,10:F2} us                       |");
            Console.WriteLine($"  |  P99.9:     {p999,10:F2} us                       |");
            Console.WriteLine($"  |  Max:       {max,10:F2} us                       |");
            Console.WriteLine($"  |  Jitter:    {jitter,10:F2}x (P99.9/P50)           |");
            Console.WriteLine($"  |  Checksum:  {checksum,10:F2}                      |");
            Console.WriteLine("  +------------------------------------------------+");
        }

        private static double TicksToMicroseconds(
            long ticks)
        {
            return ticks * 1_000_000.0 / Stopwatch.Frequency;
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