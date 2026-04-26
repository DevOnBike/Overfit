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
    /// CNN inference benchmark:
    ///
    ///     Conv2D(1 -> 8, 3x3)
    ///     ReLU
    ///     MaxPool2D(2x2)
    ///     GlobalAveragePool2D
    ///     Linear(8, 10)
    ///
    /// This benchmark verifies:
    ///
    /// - Overfit uses InferenceEngine.Run(...)
    /// - ONNX Runtime uses preallocated OrtValue input/output buffers
    /// - no AutogradNode / ComputationGraph / model.Forward(...) path is used
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
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

        private Sequential _overfitModel = null!;
        private ConvLayer _conv = null!;
        private LinearLayer _linear = null!;
        private InferenceEngine _overfitEngine = null!;

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

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            _input = new float[InputSize];
            _overfitOutput = new float[OutputSize];
            _onnxOutput = new float[OutputSize];

            FillDeterministic(_input);

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
                _overfitOutput,
                _onnxOutput,
                tolerance: 1e-4f);
        }

        private void SetupOverfit()
        {
            _conv = new ConvLayer(
                InputChannels,
                ConvOutChannels,
                InputH,
                InputW,
                Kernel);

            _linear = new LinearLayer(
                ConvOutChannels,
                OutputClasses);

            _overfitModel = new Sequential(
                _conv,
                new ReluActivation(),
                new MaxPool2DLayer(
                    ConvOutChannels,
                    ConvOutH,
                    ConvOutW,
                    Pool),
                new GlobalAveragePool2DLayer(
                    ConvOutChannels,
                    PoolOutH,
                    PoolOutW),
                _linear);

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

            _onnxSession = new InferenceSession(
                _onnxModelPath,
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

            _onnxInputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _input.AsMemory(),
                new long[] { 1, InputChannels, InputH, InputW });

            _onnxOutputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                OrtMemoryInfo.DefaultInstance,
                _onnxOutput.AsMemory(),
                new long[] { 1, OutputClasses });

            _onnxInputs = new[]
            {
                _onnxInputOrtValue
            };

            _onnxOutputs = new[]
            {
                _onnxOutputOrtValue
            };

            RunOnnxOnce();
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_Cnn_InferenceEngine_ZeroAlloc()
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
        public float OnnxRuntime_Cnn_PreAllocated()
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
            _onnxInputOrtValue?.Dispose();
            _onnxOutputOrtValue?.Dispose();
            _onnxRunOptions?.Dispose();
            _onnxSession?.Dispose();

            _overfitEngine?.Dispose();
            _overfitModel?.Dispose();

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