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
                    .WithWarmupCount(10)
                    .WithIterationCount(50)
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

            // Warmup: JIT + caches outside measured region.
            _overfit.ForwardInference(_input, _overfitOutput);
            ManualCnnForward();
        }

        [Benchmark]
        public float Overfit_Cnn_ZeroAlloc()
        {
            _overfit.ForwardInference(_input, _overfitOutput);
            return _overfitOutput[0];
        }

        [Benchmark]
        public float Manual_Cnn_TrueZeroAlloc()
        {
            ManualCnnForward();
            return _manualOutput[0];
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
            _conv.Kernels.DataView.AsReadOnlySpan().CopyTo(_manualConvKernels);

            var weights = _linear.Weights.DataView.AsReadOnlySpan();
            var bias = _linear.Bias.DataView.AsReadOnlySpan();

            bias.CopyTo(_manualLinearBias);

            // LinearLayer weights layout: [input, output]
            // Manual cache layout: [output, input]
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
            ManualConv2D(
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

            ManualLinear(
                _manualGapOutput,
                _manualLinearWeightsT,
                _manualLinearBias,
                _manualOutput);
        }

        private static void ManualConv2D(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> kernels,
            Span<float> output)
        {
            const int kernelSizePerOutput = InputChannels * Kernel * Kernel;

            for (var oc = 0; oc < ConvOutChannels; oc++)
            {
                var kernelBase = oc * kernelSizePerOutput;
                var outputChannelBase = oc * ConvOutH * ConvOutW;

                for (var oy = 0; oy < ConvOutH; oy++)
                {
                    for (var ox = 0; ox < ConvOutW; ox++)
                    {
                        var sum = 0f;

                        for (var ic = 0; ic < InputChannels; ic++)
                        {
                            var inputChannelBase = ic * InputH * InputW;
                            var kernelChannelBase = kernelBase + ic * Kernel * Kernel;

                            for (var ky = 0; ky < Kernel; ky++)
                            {
                                var inputRowBase = inputChannelBase + (oy + ky) * InputW + ox;
                                var kernelRowBase = kernelChannelBase + ky * Kernel;

                                for (var kx = 0; kx < Kernel; kx++)
                                {
                                    sum += input[inputRowBase + kx] *
                                           kernels[kernelRowBase + kx];
                                }
                            }
                        }

                        output[outputChannelBase + oy * ConvOutW + ox] = sum;
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

        private static void ManualLinear(
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