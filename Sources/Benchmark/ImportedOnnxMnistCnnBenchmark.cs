// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Inference.Contracts;
using DevOnBike.Overfit.Licensing;
using DevOnBike.Overfit.Onnx;
using Microsoft.ML.OnnxRuntime;

namespace Benchmarks
{
    /// <summary>
    /// Benchmark for the PyTorch-exported MNIST CNN fixture imported through Overfit's ONNX importer.
    ///
    /// Measures:
    /// - Overfit: OnnxImporter.Load(...) -> Sequential -> InferenceEngine.Run(...)
    /// - ONNX Runtime: same .onnx fixture through preallocated OrtValue input/output buffers
    ///
    /// PyTorch comparison should be run with tools/benchmarks/benchmark_pytorch_mnist_cnn.py.
    /// Cross-process Python timings are intentionally not mixed into BenchmarkDotNet.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class ImportedOnnxMnistCnnBenchmark : IDisposable
    {
        private const int InputSize = 1 * 28 * 28;
        private const int OutputSize = 10;
        private const int OperationsPerInvoke = 32_768;

        private string _fixtureDir = null!;
        private string _modelPath = null!;

        private float[] _input = null!;
        private float[] _overfitOutput = null!;
        private float[] _onnxOutput = null!;

        private Sequential _overfitModel = null!;
        private InferenceEngine _overfitEngine = null!;

        private InferenceSession _onnxSession = null!;
        private RunOptions _onnxRunOptions = null!;
        private OrtValue _onnxInputValue = null!;
        private OrtValue _onnxOutputValue = null!;
        private string[] _onnxInputNames = null!;
        private string[] _onnxOutputNames = null!;
        private OrtValue[] _onnxInputs = null!;
        private OrtValue[] _onnxOutputs = null!;

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            _fixtureDir = FindFixtureDirectory();
            _modelPath = Path.Combine(
                _fixtureDir,
                "mnist_cnn.onnx");

            _input = LoadFloatBin(
                Path.Combine(
                    _fixtureDir,
                    "mnist_input.bin"));

            _overfitOutput = new float[OutputSize];
            _onnxOutput = new float[OutputSize];

            SetupOverfit();
            SetupOnnxRuntime();

            for (var i = 0; i < 512; i++)
            {
                _overfitEngine.Run(
                    _input,
                    _overfitOutput);

                RunOnnxOnce();
            }

            AssertClose(
                _onnxOutput,
                _overfitOutput,
                tolerance: 1e-4f);
        }

        private void SetupOverfit()
        {
            _overfitModel = OnnxImporter.Load(_modelPath);
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
                _input,
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
                _modelPath,
                sessionOptions);

            _onnxInputNames = new[]
            {
                _onnxSession.InputMetadata.Keys.First()
            };

            _onnxOutputNames = new[]
            {
                _onnxSession.OutputMetadata.Keys.First()
            };

            _onnxRunOptions = new RunOptions();

            _onnxInputValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _input.AsMemory(),
                new long[] { 1, 1, 28, 28 });

            _onnxOutputValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _onnxOutput.AsMemory(),
                new long[] { 1, OutputSize });

            _onnxInputs = new[]
            {
                _onnxInputValue
            };

            _onnxOutputs = new[]
            {
                _onnxOutputValue
            };

            RunOnnxOnce();
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_ImportedOnnxMnistCnn_ZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _overfitEngine.Run(
                    _input,
                    _overfitOutput);

                checksum += _overfitOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float OnnxRuntime_MnistCnn_PreAllocated()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                RunOnnxOnce();
                checksum += _onnxOutput[0];
            }

            return checksum;
        }

        private void RunOnnxOnce()
        {
            _onnxSession.Run(
                _onnxRunOptions,
                _onnxInputNames,
                _onnxInputs,
                _onnxOutputNames,
                _onnxOutputs);
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

        private static string FindFixtureDirectory()
        {
            var current = new DirectoryInfo(AppContext.BaseDirectory);

            while (current != null)
            {
                var candidate = Path.Combine(
                    current.FullName,
                    "Tests",
                    "test_fixtures");

                if (File.Exists(Path.Combine(candidate, "mnist_cnn.onnx")))
                {
                    return candidate;
                }

                candidate = Path.Combine(
                    current.FullName,
                    "test_fixtures");

                if (File.Exists(Path.Combine(candidate, "mnist_cnn.onnx")))
                {
                    return candidate;
                }

                current = current.Parent;
            }

            throw new DirectoryNotFoundException(
                "Could not locate test_fixtures directory containing mnist_cnn.onnx.");
        }

        private static float[] LoadFloatBin(
            string path)
        {
            var bytes = File.ReadAllBytes(path);

            if (bytes.Length % sizeof(float) != 0)
            {
                throw new InvalidDataException(
                    $"Float fixture length must be divisible by 4: {path}");
            }

            var result = new float[bytes.Length / sizeof(float)];

            for (var i = 0; i < result.Length; i++)
            {
                var bits = BinaryPrimitives.ReadUInt32LittleEndian(
                    bytes.AsSpan(
                        i * sizeof(float),
                        sizeof(float)));

                result[i] = BitConverter.UInt32BitsToSingle(bits);
            }

            return result;
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
    }
}
