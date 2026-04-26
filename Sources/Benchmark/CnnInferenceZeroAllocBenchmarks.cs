using System.Numerics;
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
    public class CnnInferenceZeroAllocBenchmarks : IDisposable
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
        private const int ConvOutputSize = ConvOutChannels * ConvOutH * ConvOutW;
        private const int PoolOutputSize = ConvOutChannels * PoolOutH * PoolOutW;
        private const int GapOutputSize = ConvOutChannels;
        private const int OutputSize = OutputClasses;

        private const int OperationsPerInvoke = 32_768;

        private Sequential _overfit = null!;
        private ConvLayer _conv = null!;
        private LinearLayer _linear = null!;

        private float[] _input = null!;
        private float[] _overfitOutput = null!;

        private float[] _manualConvKernels = null!;
        private float[] _manualLinearWeightsT = null!;
        private float[] _manualLinearBias = null!;

        private float[] _manualConvOutput = null!;
        private float[] _manualPoolOutput = null!;
        private float[] _manualGapOutput = null!;
        private float[] _manualOutput = null!;

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
            _overfitOutput = new float[OutputSize];

            _manualConvKernels = new float[ConvOutChannels * InputChannels * Kernel * Kernel];
            _manualLinearWeightsT = new float[GapOutputSize * OutputClasses];
            _manualLinearBias = new float[OutputClasses];

            _manualConvOutput = new float[ConvOutputSize];
            _manualPoolOutput = new float[PoolOutputSize];
            _manualGapOutput = new float[GapOutputSize];
            _manualOutput = new float[OutputSize];

            FillDeterministic(_input);

            _conv = new ConvLayer(
                InputChannels,
                ConvOutChannels,
                InputH,
                InputW,
                Kernel);

            _linear = new LinearLayer(
                GapOutputSize,
                OutputClasses);

            _overfit = new Sequential(
                _conv,
                new ReluActivation(),
                new MaxPool2DLayer(ConvOutChannels, ConvOutH, ConvOutW, Pool),
                new GlobalAveragePool2DLayer(ConvOutChannels, PoolOutH, PoolOutW),
                _linear);

            _overfit.Eval();
            _overfit.PrepareInference(maxIntermediateElements: 64 * 1024);

            BuildManualCaches();

            _overfit.ForwardInference(_input, _overfitOutput);
            ManualCnnForward();

            AssertClose(_overfitOutput, _manualOutput, tolerance: 1e-4f);

            for (var i = 0; i < 256; i++)
            {
                _overfit.ForwardInference(_input, _overfitOutput);
                ManualCnnForward();
            }
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_Cnn_ZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _overfit.ForwardInference(_input, _overfitOutput);
                checksum += _overfitOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Manual_Cnn_FastZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                ManualCnnForward();
                checksum += _manualOutput[0];
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _overfit?.Dispose();
        }

        public void Dispose()
        {
            Cleanup();
        }

        private void BuildManualCaches()
        {
            _conv.Kernels.DataView
                .AsReadOnlySpan()
                .CopyTo(_manualConvKernels);

            var weights = _linear.Weights.DataView.AsReadOnlySpan();
            var bias = _linear.Bias.DataView.AsReadOnlySpan();

            bias.CopyTo(_manualLinearBias);

            // Linear weights layout: [input, output]
            // Manual cache layout:  [output, input]
            for (var i = 0; i < GapOutputSize; i++)
            {
                var srcBase = i * OutputClasses;

                for (var j = 0; j < OutputClasses; j++)
                {
                    _manualLinearWeightsT[j * GapOutputSize + i] = weights[srcBase + j];
                }
            }
        }

        private void ManualCnnForward()
        {
            ManualConv2D_1x3x3_Vectorized(
                _input,
                _manualConvKernels,
                _manualConvOutput);

            ManualReluInPlace(_manualConvOutput);

            ManualMaxPool2D(
                _manualConvOutput,
                _manualPoolOutput);

            ManualGlobalAveragePool2D(
                _manualPoolOutput,
                _manualGapOutput);

            ManualLinearOutputMajorDot(
                _manualGapOutput,
                _manualLinearWeightsT,
                _manualLinearBias,
                _manualOutput);
        }

        private static void ManualConv2D_1x3x3_Vectorized(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output)
        {
            if (!Vector.IsHardwareAccelerated || ConvOutW < Vector<float>.Count)
            {
                ManualConv2D_1x3x3_Scalar(input, kernels, output);
                return;
            }

            var vectorWidth = Vector<float>.Count;

            for (var oc = 0; oc < ConvOutChannels; oc++)
            {
                var kernelBase = oc * 9;
                var outputChannelBase = oc * ConvOutH * ConvOutW;

                var k00 = new Vector<float>(kernels[kernelBase + 0]);
                var k01 = new Vector<float>(kernels[kernelBase + 1]);
                var k02 = new Vector<float>(kernels[kernelBase + 2]);
                var k10 = new Vector<float>(kernels[kernelBase + 3]);
                var k11 = new Vector<float>(kernels[kernelBase + 4]);
                var k12 = new Vector<float>(kernels[kernelBase + 5]);
                var k20 = new Vector<float>(kernels[kernelBase + 6]);
                var k21 = new Vector<float>(kernels[kernelBase + 7]);
                var k22 = new Vector<float>(kernels[kernelBase + 8]);

                for (var oy = 0; oy < ConvOutH; oy++)
                {
                    var inputRow0 = oy * InputW;
                    var inputRow1 = (oy + 1) * InputW;
                    var inputRow2 = (oy + 2) * InputW;
                    var outputRow = outputChannelBase + oy * ConvOutW;

                    var ox = 0;

                    for (; ox <= ConvOutW - vectorWidth; ox += vectorWidth)
                    {
                        var acc =
                            new Vector<float>(input.Slice(inputRow0 + ox, vectorWidth)) * k00 +
                            new Vector<float>(input.Slice(inputRow0 + ox + 1, vectorWidth)) * k01 +
                            new Vector<float>(input.Slice(inputRow0 + ox + 2, vectorWidth)) * k02 +
                            new Vector<float>(input.Slice(inputRow1 + ox, vectorWidth)) * k10 +
                            new Vector<float>(input.Slice(inputRow1 + ox + 1, vectorWidth)) * k11 +
                            new Vector<float>(input.Slice(inputRow1 + ox + 2, vectorWidth)) * k12 +
                            new Vector<float>(input.Slice(inputRow2 + ox, vectorWidth)) * k20 +
                            new Vector<float>(input.Slice(inputRow2 + ox + 1, vectorWidth)) * k21 +
                            new Vector<float>(input.Slice(inputRow2 + ox + 2, vectorWidth)) * k22;

                        acc.CopyTo(output.Slice(outputRow + ox, vectorWidth));
                    }

                    for (; ox < ConvOutW; ox++)
                    {
                        output[outputRow + ox] =
                            input[inputRow0 + ox] * kernels[kernelBase + 0] +
                            input[inputRow0 + ox + 1] * kernels[kernelBase + 1] +
                            input[inputRow0 + ox + 2] * kernels[kernelBase + 2] +
                            input[inputRow1 + ox] * kernels[kernelBase + 3] +
                            input[inputRow1 + ox + 1] * kernels[kernelBase + 4] +
                            input[inputRow1 + ox + 2] * kernels[kernelBase + 5] +
                            input[inputRow2 + ox] * kernels[kernelBase + 6] +
                            input[inputRow2 + ox + 1] * kernels[kernelBase + 7] +
                            input[inputRow2 + ox + 2] * kernels[kernelBase + 8];
                    }
                }
            }
        }

        private static void ManualConv2D_1x3x3_Scalar(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output)
        {
            for (var oc = 0; oc < ConvOutChannels; oc++)
            {
                var kernelBase = oc * 9;
                var outputChannelBase = oc * ConvOutH * ConvOutW;

                var k00 = kernels[kernelBase + 0];
                var k01 = kernels[kernelBase + 1];
                var k02 = kernels[kernelBase + 2];
                var k10 = kernels[kernelBase + 3];
                var k11 = kernels[kernelBase + 4];
                var k12 = kernels[kernelBase + 5];
                var k20 = kernels[kernelBase + 6];
                var k21 = kernels[kernelBase + 7];
                var k22 = kernels[kernelBase + 8];

                for (var oy = 0; oy < ConvOutH; oy++)
                {
                    var inputRow0 = oy * InputW;
                    var inputRow1 = (oy + 1) * InputW;
                    var inputRow2 = (oy + 2) * InputW;
                    var outputRow = outputChannelBase + oy * ConvOutW;

                    for (var ox = 0; ox < ConvOutW; ox++)
                    {
                        output[outputRow + ox] =
                            input[inputRow0 + ox] * k00 +
                            input[inputRow0 + ox + 1] * k01 +
                            input[inputRow0 + ox + 2] * k02 +
                            input[inputRow1 + ox] * k10 +
                            input[inputRow1 + ox + 1] * k11 +
                            input[inputRow1 + ox + 2] * k12 +
                            input[inputRow2 + ox] * k20 +
                            input[inputRow2 + ox + 1] * k21 +
                            input[inputRow2 + ox + 2] * k22;
                    }
                }
            }
        }

        private static void ManualReluInPlace(Span<float> values)
        {
            for (var i = 0; i < values.Length; i++)
            {
                if (values[i] < 0f)
                {
                    values[i] = 0f;
                }
            }
        }

        private static void ManualMaxPool2D(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            for (var c = 0; c < ConvOutChannels; c++)
            {
                var inputChannelBase = c * ConvOutH * ConvOutW;
                var outputChannelBase = c * PoolOutH * PoolOutW;

                for (var oh = 0; oh < PoolOutH; oh++)
                {
                    for (var ow = 0; ow < PoolOutW; ow++)
                    {
                        var max = float.MinValue;

                        for (var ph = 0; ph < Pool; ph++)
                        {
                            var iy = oh * Pool + ph;
                            var inputRowBase = inputChannelBase + iy * ConvOutW + ow * Pool;

                            for (var pw = 0; pw < Pool; pw++)
                            {
                                var value = input[inputRowBase + pw];

                                if (value > max)
                                {
                                    max = value;
                                }
                            }
                        }

                        output[outputChannelBase + oh * PoolOutW + ow] = max;
                    }
                }
            }
        }

        private static void ManualGlobalAveragePool2D(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            const int spatialSize = PoolOutH * PoolOutW;
            const float scale = 1f / spatialSize;

            for (var c = 0; c < ConvOutChannels; c++)
            {
                var baseIndex = c * spatialSize;
                var sum = 0f;

                for (var i = 0; i < spatialSize; i++)
                {
                    sum += input[baseIndex + i];
                }

                output[c] = sum * scale;
            }
        }

        private static void ManualLinearOutputMajorDot(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsT,
            ReadOnlySpan<float> bias,
            Span<float> output)
        {
            for (var j = 0; j < output.Length; j++)
            {
                var wBase = j * input.Length;
                var sum = bias[j];

                for (var i = 0; i < input.Length; i++)
                {
                    sum += input[i] * weightsT[wBase + i];
                }

                output[j] = sum;
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
                        $"Mismatch at {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}");
                }
            }
        }

        private static void FillDeterministic(float[] data)
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