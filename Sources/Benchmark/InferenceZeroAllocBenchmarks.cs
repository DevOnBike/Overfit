using System.Numerics;
using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Licensing;

namespace Benchmarks
{
    [MemoryDiagnoser]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [RankColumn]
    [Config(typeof(Config))]
    public class InferenceZeroAllocBenchmarks : IDisposable
    {
        private const int InputSize = 784;
        private const int HiddenSize = 128;
        private const int OutputSize = 10;

        private const int SingleLinearOperationsPerInvoke = 524_288;
        private const int MultiLayerOperationsPerInvoke = 32_768;

        private Sequential _singleLinearModel = null!;
        private LinearLayer _singleLinear = null!;

        private Sequential _multiLayerModel = null!;
        private LinearLayer _multiLinear1 = null!;
        private LinearLayer _multiLinear2 = null!;

        private float[] _input = null!;

        private float[] _overfitSingleOutput = null!;
        private float[] _overfitMultiOutput = null!;

        private float[] _manualSingleOutput = null!;
        private float[] _manualMultiOutput = null!;
        private float[] _manualHidden = null!;

        private float[] _singleWeightsT = null!;
        private float[] _singleBias = null!;

        private float[] _multiWeights1 = null!;
        private float[] _multiBias1 = null!;
        private float[] _multiWeights2T = null!;
        private float[] _multiBias2 = null!;

        private class Config : ManualConfig
        {
            public Config()
            {
                AddJob(Job.Default
                    .WithWarmupCount(5)
                    .WithIterationCount(20)
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

            _overfitSingleOutput = new float[OutputSize];
            _overfitMultiOutput = new float[OutputSize];

            _manualSingleOutput = new float[OutputSize];
            _manualMultiOutput = new float[OutputSize];
            _manualHidden = new float[HiddenSize];

            _singleWeightsT = new float[OutputSize * InputSize];
            _singleBias = new float[OutputSize];

            _multiWeights1 = new float[InputSize * HiddenSize];
            _multiBias1 = new float[HiddenSize];
            _multiWeights2T = new float[OutputSize * HiddenSize];
            _multiBias2 = new float[OutputSize];

            FillDeterministic(_input);

            _singleLinear = new LinearLayer(InputSize, OutputSize);

            _singleLinearModel = new Sequential(
                _singleLinear);

            _multiLinear1 = new LinearLayer(InputSize, HiddenSize);
            _multiLinear2 = new LinearLayer(HiddenSize, OutputSize);

            _multiLayerModel = new Sequential(
                _multiLinear1,
                new ReluActivation(),
                _multiLinear2);

            _singleLinearModel.Eval();
            _singleLinearModel.PrepareInference(64 * 1024);

            _multiLayerModel.Eval();
            _multiLayerModel.PrepareInference(64 * 1024);

            BuildManualCaches();

            _singleLinearModel.ForwardInference(_input, _overfitSingleOutput);
            ManualSingleLinearForward();

            _multiLayerModel.ForwardInference(_input, _overfitMultiOutput);
            ManualMultiLayerForward();

            AssertClose(_overfitSingleOutput, _manualSingleOutput, 1e-4f);
            AssertClose(_overfitMultiOutput, _manualMultiOutput, 1e-3f);

            for (var i = 0; i < 512; i++)
            {
                _singleLinearModel.ForwardInference(_input, _overfitSingleOutput);
                _multiLayerModel.ForwardInference(_input, _overfitMultiOutput);

                ManualSingleLinearForward();
                ManualMultiLayerForward();
            }
        }

        [Benchmark(OperationsPerInvoke = SingleLinearOperationsPerInvoke)]
        public float Overfit_SingleLinear_ZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < SingleLinearOperationsPerInvoke; i++)
            {
                _singleLinearModel.ForwardInference(_input, _overfitSingleOutput);
                checksum += _overfitSingleOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = SingleLinearOperationsPerInvoke)]
        public float Manual_SingleLinear_TrueZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < SingleLinearOperationsPerInvoke; i++)
            {
                ManualSingleLinearForward();
                checksum += _manualSingleOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MultiLayerOperationsPerInvoke)]
        public float Overfit_MultiLayer_ZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < MultiLayerOperationsPerInvoke; i++)
            {
                _multiLayerModel.ForwardInference(_input, _overfitMultiOutput);
                checksum += _overfitMultiOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MultiLayerOperationsPerInvoke)]
        public float Manual_MultiLayer_TrueZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < MultiLayerOperationsPerInvoke; i++)
            {
                ManualMultiLayerForward();
                checksum += _manualMultiOutput[0];
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _singleLinearModel?.Dispose();
            _multiLayerModel?.Dispose();
        }

        public void Dispose()
        {
            Cleanup();
        }

        private void BuildManualCaches()
        {
            var singleWeights = _singleLinear.Weights.DataView.AsReadOnlySpan();
            var singleBias = _singleLinear.Bias.DataView.AsReadOnlySpan();

            TransposeInputOutputToOutputInput(
                singleWeights,
                _singleWeightsT,
                InputSize,
                OutputSize);

            singleBias.CopyTo(_singleBias);

            var multiWeights1 = _multiLinear1.Weights.DataView.AsReadOnlySpan();
            var multiBias1 = _multiLinear1.Bias.DataView.AsReadOnlySpan();

            multiWeights1.CopyTo(_multiWeights1);
            multiBias1.CopyTo(_multiBias1);

            var multiWeights2 = _multiLinear2.Weights.DataView.AsReadOnlySpan();
            var multiBias2 = _multiLinear2.Bias.DataView.AsReadOnlySpan();

            TransposeInputOutputToOutputInput(
                multiWeights2,
                _multiWeights2T,
                HiddenSize,
                OutputSize);

            multiBias2.CopyTo(_multiBias2);
        }

        private void ManualSingleLinearForward()
        {
            LinearOutputMajorDot(
                _input,
                _singleWeightsT,
                _singleBias,
                _manualSingleOutput);
        }

        private void ManualMultiLayerForward()
        {
            LinearInputMajorVector4(
                _input,
                _multiWeights1,
                _multiBias1,
                _manualHidden,
                InputSize,
                HiddenSize);

            ReluInPlace(_manualHidden);

            LinearOutputMajorDot(
                _manualHidden,
                _multiWeights2T,
                _multiBias2,
                _manualMultiOutput);
        }

        private static void LinearOutputMajorDot(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsOutputInput,
            ReadOnlySpan<float> bias,
            Span<float> output)
        {
            var inputSize = input.Length;
            var outputSize = output.Length;

            for (var j = 0; j < outputSize; j++)
            {
                output[j] =
                    TensorPrimitives.Dot(
                        input,
                        weightsOutputInput.Slice(j * inputSize, inputSize)) +
                    bias[j];
            }
        }

        private static void LinearInputMajorVector4(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (!Vector.IsHardwareAccelerated ||
                outputSize < Vector<float>.Count * 4)
            {
                LinearInputMajorScalar(
                    input,
                    weightsInputOutput,
                    bias,
                    output,
                    inputSize,
                    outputSize);

                return;
            }

            var vectorWidth = Vector<float>.Count;
            var blockWidth = vectorWidth * 4;

            var j = 0;

            for (; j <= outputSize - blockWidth; j += blockWidth)
            {
                var acc0 = new Vector<float>(bias.Slice(j, vectorWidth));
                var acc1 = new Vector<float>(bias.Slice(j + vectorWidth, vectorWidth));
                var acc2 = new Vector<float>(bias.Slice(j + vectorWidth * 2, vectorWidth));
                var acc3 = new Vector<float>(bias.Slice(j + vectorWidth * 3, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    var x = new Vector<float>(input[i]);
                    var rowBase = i * outputSize + j;

                    acc0 += x * new Vector<float>(weightsInputOutput.Slice(rowBase, vectorWidth));
                    acc1 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth, vectorWidth));
                    acc2 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth * 2, vectorWidth));
                    acc3 += x * new Vector<float>(weightsInputOutput.Slice(rowBase + vectorWidth * 3, vectorWidth));
                }

                acc0.CopyTo(output.Slice(j, vectorWidth));
                acc1.CopyTo(output.Slice(j + vectorWidth, vectorWidth));
                acc2.CopyTo(output.Slice(j + vectorWidth * 2, vectorWidth));
                acc3.CopyTo(output.Slice(j + vectorWidth * 3, vectorWidth));
            }

            for (; j <= outputSize - vectorWidth; j += vectorWidth)
            {
                var acc = new Vector<float>(bias.Slice(j, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    acc += new Vector<float>(input[i]) *
                           new Vector<float>(weightsInputOutput.Slice(i * outputSize + j, vectorWidth));
                }

                acc.CopyTo(output.Slice(j, vectorWidth));
            }

            for (; j < outputSize; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputOutput[i * outputSize + j];
                }

                output[j] = sum;
            }
        }

        private static void LinearInputMajorScalar(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsInputOutput,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            bias.Slice(0, outputSize).CopyTo(output);

            for (var i = 0; i < inputSize; i++)
            {
                var x = input[i];
                var wBase = i * outputSize;

                for (var j = 0; j < outputSize; j++)
                {
                    output[j] += x * weightsInputOutput[wBase + j];
                }
            }
        }

        private static void ReluInPlace(
            Span<float> values)
        {
            for (var i = 0; i < values.Length; i++)
            {
                if (values[i] < 0f)
                {
                    values[i] = 0f;
                }
            }
        }

        private static void TransposeInputOutputToOutputInput(
            ReadOnlySpan<float> sourceInputOutput,
            Span<float> destinationOutputInput,
            int inputSize,
            int outputSize)
        {
            for (var i = 0; i < inputSize; i++)
            {
                var srcBase = i * outputSize;

                for (var j = 0; j < outputSize; j++)
                {
                    destinationOutputInput[j * inputSize + i] = sourceInputOutput[srcBase + j];
                }
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
                        $"Mismatch at {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}, tolerance={tolerance}");
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