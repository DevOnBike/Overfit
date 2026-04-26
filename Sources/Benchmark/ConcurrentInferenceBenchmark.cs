// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Threading;
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
    /// Concurrent single-sample inference benchmark.
    ///
    /// This benchmark intentionally does not use Task.Run or Parallel.For in the
    /// measured region. Worker threads are created once in GlobalSetup.
    ///
    /// Overfit path:
    /// - one Sequential + InferenceEngine per worker
    /// - one input/output buffer per worker
    /// - InferenceEngine.Run(...)
    ///
    /// ONNX path:
    /// - one shared InferenceSession
    /// - one preallocated OrtValue input/output pair per worker
    ///
    /// Required files:
    /// - benchmark_mlp3.bin
    /// - benchmark_mlp3.onnx
    ///
    /// Note:
    /// BenchmarkDotNet reports the time for one full concurrent round:
    /// WorkerCount workers * InnerIterations inferences per worker.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class ConcurrentInferenceBenchmark : IDisposable
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private const string OnnxPath = "benchmark_mlp3.onnx";
        private const string BinPath = "benchmark_mlp3.bin";

        // If per-inference is ~8 us, 16k iterations per worker gives enough
        // measured time even when all workers run concurrently.
        private const int InnerIterations = 16_384;

        private int _workerCount;

        private InferenceSession _onnxSession = null!;
        private WorkerContext[] _contexts = null!;
        private WorkerHarness _harness = null!;

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            if (!File.Exists(OnnxPath) || !File.Exists(BinPath))
            {
                throw new InvalidOperationException(
                    $"Missing {OnnxPath} or {BinPath}. Generate benchmark_mlp3 files first.");
            }

            _workerCount = Environment.ProcessorCount;

            SetupOnnxRuntime();

            _contexts = new WorkerContext[_workerCount];

            for (var i = 0; i < _contexts.Length; i++)
            {
                _contexts[i] = new WorkerContext(
                    workerIndex: i,
                    onnxSession: _onnxSession,
                    binPath: BinPath);
            }

            // Verify correctness before worker threads start.
            _contexts[0].RunOverfitOnce();
            _contexts[0].RunOnnxOnce();

            AssertClose(
                _contexts[0].OnnxOutput,
                _contexts[0].OverfitOutput,
                tolerance: 1e-3f);

            _harness = new WorkerHarness(
                _contexts,
                InnerIterations);

            _harness.Start();

            // Warmup outside measured region.
            for (var i = 0; i < 8; i++)
            {
                _harness.RunOverfitRound();
                _harness.RunOnnxRound();
            }
        }

        private void SetupOnnxRuntime()
        {
            var sessionOptions = new SessionOptions
            {
                EnableCpuMemArena = true,
                EnableMemoryPattern = true,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,

                // We benchmark external concurrency here, so each ONNX Run should
                // not spawn its own large worker pool.
                InterOpNumThreads = 1,
                IntraOpNumThreads = 1
            };

            _onnxSession = new InferenceSession(
                OnnxPath,
                sessionOptions);
        }

        [Benchmark(Baseline = true)]
        public double OnnxRuntime_Concurrent()
        {
            return _harness.RunOnnxRound();
        }

        [Benchmark]
        public double Overfit_Concurrent_ZeroContention()
        {
            return _harness.RunOverfitRound();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _harness?.Dispose();

            if (_contexts is not null)
            {
                for (var i = 0; i < _contexts.Length; i++)
                {
                    _contexts[i]?.Dispose();
                }
            }

            _onnxSession?.Dispose();
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
            float[] data,
            int workerIndex)
        {
            var seed = 0x12345678u + (uint)(workerIndex * 7919);

            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;

                var normalized = (seed & 0x00FFFFFF) / 16777216f;
                data[i] = normalized * 2f - 1f;
            }
        }

        private class WorkerContext : IDisposable
        {
            private readonly InferenceSession _onnxSession;

            private readonly Sequential _overfitModel;
            private readonly InferenceEngine _overfitEngine;

            private readonly OrtValue _onnxInputValue;
            private readonly OrtValue _onnxOutputValue;
            private readonly RunOptions _onnxRunOptions;

            private readonly string[] _inputNames;
            private readonly string[] _outputNames;
            private readonly OrtValue[] _inputValues;
            private readonly OrtValue[] _outputValues;

            public WorkerContext(
                int workerIndex,
                InferenceSession onnxSession,
                string binPath)
            {
                _onnxSession = onnxSession;

                Input = new float[InputSize];
                OverfitOutput = new float[OutputSize];
                OnnxOutput = new float[OutputSize];

                FillDeterministic(
                    Input,
                    workerIndex);

                _overfitModel = new Sequential(
                    new LinearLayer(InputSize, 256),
                    new ReluActivation(),
                    new LinearLayer(256, 128),
                    new ReluActivation(),
                    new LinearLayer(128, OutputSize));

                _overfitModel.Load(binPath);
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

                _onnxInputValue = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance,
                    Input.AsMemory(),
                    new long[] { 1, InputSize });

                _onnxOutputValue = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance,
                    OnnxOutput.AsMemory(),
                    new long[] { 1, OutputSize });

                _onnxRunOptions = new RunOptions();

                _inputNames = new[] { "input" };
                _outputNames = new[] { "output" };

                _inputValues = new[] { _onnxInputValue };
                _outputValues = new[] { _onnxOutputValue };

                RunOverfitOnce();
                RunOnnxOnce();
            }

            public float[] Input { get; }

            public float[] OverfitOutput { get; }

            public float[] OnnxOutput { get; }

            public double Checksum { get; set; }

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

            public double RunOverfitLoop(
                int iterations)
            {
                var checksum = 0.0;

                for (var i = 0; i < iterations; i++)
                {
                    _overfitEngine.Run(
                        Input,
                        OverfitOutput);

                    checksum += OverfitOutput[0];
                }

                Checksum = checksum;
                return checksum;
            }

            public double RunOnnxLoop(
                int iterations)
            {
                var checksum = 0.0;

                for (var i = 0; i < iterations; i++)
                {
                    _onnxSession.Run(
                        _onnxRunOptions,
                        _inputNames,
                        _inputValues,
                        _outputNames,
                        _outputValues);

                    checksum += OnnxOutput[0];
                }

                Checksum = checksum;
                return checksum;
            }

            public void Dispose()
            {
                _onnxInputValue.Dispose();
                _onnxOutputValue.Dispose();
                _onnxRunOptions.Dispose();

                _overfitEngine.Dispose();
                _overfitModel.Dispose();
            }
        }

        private class WorkerHarness : IDisposable
        {
            private const int ModeNone = 0;
            private const int ModeOverfit = 1;
            private const int ModeOnnx = 2;

            private readonly WorkerContext[] _contexts;
            private readonly Thread[] _threads;
            private readonly int _innerIterations;

            private volatile bool _stop;
            private int _phase;
            private int _mode;
            private int _remaining;
            private Exception? _workerException;

            public WorkerHarness(
                WorkerContext[] contexts,
                int innerIterations)
            {
                _contexts = contexts;
                _innerIterations = innerIterations;
                _threads = new Thread[contexts.Length];

                for (var i = 0; i < _threads.Length; i++)
                {
                    var workerIndex = i;

                    _threads[i] = new Thread(
                        () => WorkerLoop(workerIndex))
                    {
                        IsBackground = true,
                        Name = $"OverfitConcurrentBenchmarkWorker-{workerIndex}"
                    };
                }
            }

            public void Start()
            {
                for (var i = 0; i < _threads.Length; i++)
                {
                    _threads[i].Start();
                }
            }

            public double RunOverfitRound()
            {
                return RunRound(ModeOverfit);
            }

            public double RunOnnxRound()
            {
                return RunRound(ModeOnnx);
            }

            private double RunRound(
                int mode)
            {
                var exception = Volatile.Read(ref _workerException);

                if (exception is not null)
                {
                    throw new InvalidOperationException(
                        "Worker failed in previous round.",
                        exception);
                }

                Volatile.Write(ref _remaining, _contexts.Length);
                Volatile.Write(ref _mode, mode);

                Interlocked.Increment(ref _phase);

                var spinner = new SpinWait();

                while (Volatile.Read(ref _remaining) != 0)
                {
                    spinner.SpinOnce();
                }

                exception = Volatile.Read(ref _workerException);

                if (exception is not null)
                {
                    throw new InvalidOperationException(
                        "Worker failed.",
                        exception);
                }

                var checksum = 0.0;

                for (var i = 0; i < _contexts.Length; i++)
                {
                    checksum += _contexts[i].Checksum;
                }

                return checksum;
            }

            private void WorkerLoop(
                int workerIndex)
            {
                var observedPhase = Volatile.Read(ref _phase);
                var context = _contexts[workerIndex];

                while (!Volatile.Read(ref _stop))
                {
                    var currentPhase = Volatile.Read(ref _phase);

                    if (currentPhase == observedPhase)
                    {
                        Thread.Yield();
                        continue;
                    }

                    observedPhase = currentPhase;

                    try
                    {
                        var mode = Volatile.Read(ref _mode);

                        if (mode == ModeOverfit)
                        {
                            context.RunOverfitLoop(_innerIterations);
                        }
                        else if (mode == ModeOnnx)
                        {
                            context.RunOnnxLoop(_innerIterations);
                        }
                        else if (mode == ModeNone)
                        {
                            context.Checksum = 0.0;
                        }
                    }
                    catch (Exception ex)
                    {
                        Interlocked.CompareExchange(
                            ref _workerException,
                            ex,
                            null);
                    }
                    finally
                    {
                        Interlocked.Decrement(ref _remaining);
                    }
                }
            }

            public void Dispose()
            {
                Volatile.Write(ref _stop, true);
                Volatile.Write(ref _mode, ModeNone);

                Interlocked.Increment(ref _phase);

                for (var i = 0; i < _threads.Length; i++)
                {
                    _threads[i].Join();
                }
            }
        }
    }
}