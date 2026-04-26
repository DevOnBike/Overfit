// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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

        private Sequential _singleLayerModel = null!;
        private Sequential _multiLayerModel = null!;

        private LinearLayer _singleLinear = null!;
        private LinearLayer _multiLinear1 = null!;
        private LinearLayer _multiLinear2 = null!;

        private float[] _input = null!;

        private float[] _singleOutput = null!;
        private float[] _multiOutput = null!;

        private float[] _manualSingleWeightsT = null!;
        private float[] _manualSingleBias = null!;
        private float[] _manualSingleOutput = null!;

        private float[] _manualMultiWeights1T = null!;
        private float[] _manualMultiBias1 = null!;
        private float[] _manualMultiHidden = null!;

        private float[] _manualMultiWeights2T = null!;
        private float[] _manualMultiBias2 = null!;
        private float[] _manualMultiOutput = null!;

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

            _singleOutput = new float[OutputSize];
            _multiOutput = new float[OutputSize];

            _manualSingleOutput = new float[OutputSize];

            _manualMultiHidden = new float[HiddenSize];
            _manualMultiOutput = new float[OutputSize];

            FillDeterministic(_input);

            _singleLinear = new LinearLayer(InputSize, OutputSize);
            _singleLayerModel = new Sequential(_singleLinear);

            _multiLinear1 = new LinearLayer(InputSize, HiddenSize);
            _multiLinear2 = new LinearLayer(HiddenSize, OutputSize);

            _multiLayerModel = new Sequential(
                _multiLinear1,
                new ReluActivation(),
                _multiLinear2);

            _singleLayerModel.Eval();
            _singleLayerModel.PrepareInference(maxIntermediateElements: InputSize + HiddenSize + OutputSize + 1024);

            _multiLayerModel.Eval();
            _multiLayerModel.PrepareInference(maxIntermediateElements: InputSize + HiddenSize + OutputSize + 1024);

            _manualSingleWeightsT = new float[InputSize * OutputSize];
            _manualSingleBias = new float[OutputSize];

            _manualMultiWeights1T = new float[InputSize * HiddenSize];
            _manualMultiBias1 = new float[HiddenSize];

            _manualMultiWeights2T = new float[HiddenSize * OutputSize];
            _manualMultiBias2 = new float[OutputSize];

            BuildManualLinearCache(
                _singleLinear,
                InputSize,
                OutputSize,
                _manualSingleWeightsT,
                _manualSingleBias);

            BuildManualLinearCache(
                _multiLinear1,
                InputSize,
                HiddenSize,
                _manualMultiWeights1T,
                _manualMultiBias1);

            BuildManualLinearCache(
                _multiLinear2,
                HiddenSize,
                OutputSize,
                _manualMultiWeights2T,
                _manualMultiBias2);

            // Warmup: JIT + inference cache poza pomiarem.
            _singleLayerModel.ForwardInference(_input, _singleOutput);
            _multiLayerModel.ForwardInference(_input, _multiOutput);

            ManualLinear(_input, _manualSingleWeightsT, _manualSingleBias, _manualSingleOutput);

            ManualLinear(_input, _manualMultiWeights1T, _manualMultiBias1, _manualMultiHidden);
            ManualReluInPlace(_manualMultiHidden);
            ManualLinear(_manualMultiHidden, _manualMultiWeights2T, _manualMultiBias2, _manualMultiOutput);
        }

        [Benchmark]
        public float Overfit_SingleLinear_ZeroAlloc()
        {
            _singleLayerModel.ForwardInference(_input, _singleOutput);
            return _singleOutput[0];
        }

        [Benchmark]
        public float Manual_SingleLinear_TrueZeroAlloc()
        {
            ManualLinear(_input, _manualSingleWeightsT, _manualSingleBias, _manualSingleOutput);
            return _manualSingleOutput[0];
        }

        [Benchmark]
        public float Overfit_MultiLayer_ZeroAlloc()
        {
            _multiLayerModel.ForwardInference(_input, _multiOutput);
            return _multiOutput[0];
        }

        [Benchmark]
        public float Manual_MultiLayer_TrueZeroAlloc()
        {
            ManualLinear(_input, _manualMultiWeights1T, _manualMultiBias1, _manualMultiHidden);
            ManualReluInPlace(_manualMultiHidden);
            ManualLinear(_manualMultiHidden, _manualMultiWeights2T, _manualMultiBias2, _manualMultiOutput);

            return _manualMultiOutput[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _singleLayerModel?.Dispose();
            _multiLayerModel?.Dispose();
        }

        public void Dispose()
        {
            Cleanup();
        }

        private static void BuildManualLinearCache(
            LinearLayer layer,
            int inputSize,
            int outputSize,
            float[] weightsT,
            float[] bias)
        {
            var weights = layer.Weights.DataView.AsReadOnlySpan();
            var biasSpan = layer.Bias.DataView.AsReadOnlySpan();

            biasSpan.CopyTo(bias);

            // Layer weights layout: [input, output]
            // Manual cache layout: [output, input]
            for (var i = 0; i < inputSize; i++)
            {
                var srcBase = i * outputSize;

                for (var j = 0; j < outputSize; j++)
                {
                    weightsT[j * inputSize + i] = weights[srcBase + j];
                }
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
                var wRow = weightsT.Slice(j * input.Length, input.Length);
                output[j] = TensorPrimitives.Dot(input, wRow) + bias[j];
            }
        }

        private static void ManualReluInPlace(Span<float> values)
        {
            TensorPrimitives.Max(values, 0f, values);
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