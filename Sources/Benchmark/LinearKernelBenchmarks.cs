// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using Benchmarks.Helpers;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class LinearKernelBenchmarks
    {
        private const int BigInputSize = 784;
        private const int BigOutputSize = 128;

        private const int SmallInputSize = 128;
        private const int SmallOutputSize = 10;

        private const int OperationsPerInvoke = 32_768;

        private float[] _bigInput = null!;
        private float[] _bigWeights = null!;
        private float[] _bigWeightsT = null!;
        private float[] _bigBias = null!;
        private float[] _bigOutput = null!;

        private float[] _smallInput = null!;
        private float[] _smallWeights = null!;
        private float[] _smallWeightsT = null!;
        private float[] _smallBias = null!;
        private float[] _smallOutput = null!;

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
            _bigInput = new float[BigInputSize];
            _bigWeights = new float[BigInputSize * BigOutputSize];
            _bigWeightsT = new float[BigInputSize * BigOutputSize];
            _bigBias = new float[BigOutputSize];
            _bigOutput = new float[BigOutputSize];

            _smallInput = new float[SmallInputSize];
            _smallWeights = new float[SmallInputSize * SmallOutputSize];
            _smallWeightsT = new float[SmallInputSize * SmallOutputSize];
            _smallBias = new float[SmallOutputSize];
            _smallOutput = new float[SmallOutputSize];

            FillDeterministic(_bigInput, 1);
            FillDeterministic(_bigWeights, 2);
            FillDeterministic(_bigBias, 3);
            Transpose(_bigWeights, _bigWeightsT, BigInputSize, BigOutputSize);

            FillDeterministic(_smallInput, 4);
            FillDeterministic(_smallWeights, 5);
            FillDeterministic(_smallBias, 6);
            Transpose(_smallWeights, _smallWeightsT, SmallInputSize, SmallOutputSize);

            // Warmup all kernels.
            Big_OutputMajorDot();
            Big_InputMajorScalar();
            Big_InputMajorVector1();
            Big_InputMajorVector2();
            Big_InputMajorVector4();

            Small_OutputMajorDot();
            Small_InputMajorScalar();
            Small_InputMajorVector1();
        }

        // -----------------------------
        // Big Linear: 784 -> 128
        // -----------------------------

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Big_OutputMajorDot()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                OutputMajorDot(
                    _bigInput,
                    _bigWeightsT,
                    _bigBias,
                    _bigOutput);

                checksum += _bigOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Big_InputMajorScalar()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                InputMajorScalar(
                    _bigInput,
                    _bigWeights,
                    _bigBias,
                    _bigOutput,
                    BigInputSize,
                    BigOutputSize);

                checksum += _bigOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Big_InputMajorVector1()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                InputMajorVector1(
                    _bigInput,
                    _bigWeights,
                    _bigBias,
                    _bigOutput,
                    BigInputSize,
                    BigOutputSize);

                checksum += _bigOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Big_InputMajorVector2()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                InputMajorVector2(
                    _bigInput,
                    _bigWeights,
                    _bigBias,
                    _bigOutput,
                    BigInputSize,
                    BigOutputSize);

                checksum += _bigOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Big_InputMajorVector4()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                InputMajorVector4(
                    _bigInput,
                    _bigWeights,
                    _bigBias,
                    _bigOutput,
                    BigInputSize,
                    BigOutputSize);

                checksum += _bigOutput[0];
            }

            return checksum;
        }

        // -----------------------------
        // Small Linear: 128 -> 10
        // -----------------------------

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Small_OutputMajorDot()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                OutputMajorDot(
                    _smallInput,
                    _smallWeightsT,
                    _smallBias,
                    _smallOutput);

                checksum += _smallOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Small_InputMajorScalar()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                InputMajorScalar(
                    _smallInput,
                    _smallWeights,
                    _smallBias,
                    _smallOutput,
                    SmallInputSize,
                    SmallOutputSize);

                checksum += _smallOutput[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Small_InputMajorVector1()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                InputMajorVector1(
                    _smallInput,
                    _smallWeights,
                    _smallBias,
                    _smallOutput,
                    SmallInputSize,
                    SmallOutputSize);

                checksum += _smallOutput[0];
            }

            return checksum;
        }

        // -----------------------------
        // Kernels
        // -----------------------------

        private static void OutputMajorDot(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsT,
            ReadOnlySpan<float> bias,
            Span<float> output)
        {
            var inputSize = input.Length;
            var outputSize = output.Length;

            for (var j = 0; j < outputSize; j++)
            {
                output[j] = TensorPrimitives.Dot(
                    input,
                    weightsT.Slice(j * inputSize, inputSize)) + bias[j];
            }
        }

        private static void InputMajorScalar(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weights,
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
                    output[j] += x * weights[wBase + j];
                }
            }
        }

        private static void InputMajorVector1(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weights,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (!Vector.IsHardwareAccelerated || outputSize < Vector<float>.Count)
            {
                InputMajorScalar(input, weights, bias, output, inputSize, outputSize);
                return;
            }

            var vectorWidth = Vector<float>.Count;
            var j = 0;

            for (; j <= outputSize - vectorWidth; j += vectorWidth)
            {
                var acc = new Vector<float>(bias.Slice(j, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    var x = new Vector<float>(input[i]);
                    var w = new Vector<float>(weights.Slice(i * outputSize + j, vectorWidth));

                    acc += x * w;
                }

                acc.CopyTo(output.Slice(j, vectorWidth));
            }

            for (; j < outputSize; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weights[i * outputSize + j];
                }

                output[j] = sum;
            }
        }

        private static void InputMajorVector2(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weights,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (!Vector.IsHardwareAccelerated || outputSize < Vector<float>.Count * 2)
            {
                InputMajorVector1(input, weights, bias, output, inputSize, outputSize);
                return;
            }

            var vectorWidth = Vector<float>.Count;
            var blockWidth = vectorWidth * 2;
            var j = 0;

            for (; j <= outputSize - blockWidth; j += blockWidth)
            {
                var acc0 = new Vector<float>(bias.Slice(j, vectorWidth));
                var acc1 = new Vector<float>(bias.Slice(j + vectorWidth, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    var x = new Vector<float>(input[i]);
                    var rowBase = i * outputSize + j;

                    acc0 += x * new Vector<float>(weights.Slice(rowBase, vectorWidth));
                    acc1 += x * new Vector<float>(weights.Slice(rowBase + vectorWidth, vectorWidth));
                }

                acc0.CopyTo(output.Slice(j, vectorWidth));
                acc1.CopyTo(output.Slice(j + vectorWidth, vectorWidth));
            }

            for (; j <= outputSize - vectorWidth; j += vectorWidth)
            {
                var acc = new Vector<float>(bias.Slice(j, vectorWidth));

                for (var i = 0; i < inputSize; i++)
                {
                    acc += new Vector<float>(input[i]) *
                           new Vector<float>(weights.Slice(i * outputSize + j, vectorWidth));
                }

                acc.CopyTo(output.Slice(j, vectorWidth));
            }

            for (; j < outputSize; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weights[i * outputSize + j];
                }

                output[j] = sum;
            }
        }

        private static void InputMajorVector4(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weights,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (!Vector.IsHardwareAccelerated || outputSize < Vector<float>.Count * 4)
            {
                InputMajorVector2(input, weights, bias, output, inputSize, outputSize);
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

                    acc0 += x * new Vector<float>(weights.Slice(rowBase, vectorWidth));
                    acc1 += x * new Vector<float>(weights.Slice(rowBase + vectorWidth, vectorWidth));
                    acc2 += x * new Vector<float>(weights.Slice(rowBase + vectorWidth * 2, vectorWidth));
                    acc3 += x * new Vector<float>(weights.Slice(rowBase + vectorWidth * 3, vectorWidth));
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
                           new Vector<float>(weights.Slice(i * outputSize + j, vectorWidth));
                }

                acc.CopyTo(output.Slice(j, vectorWidth));
            }

            for (; j < outputSize; j++)
            {
                var sum = bias[j];

                for (var i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weights[i * outputSize + j];
                }

                output[j] = sum;
            }
        }

        // -----------------------------
        // Setup helpers
        // -----------------------------

        private static void Transpose(
            ReadOnlySpan<float> src,
            Span<float> dst,
            int inputSize,
            int outputSize)
        {
            for (var i = 0; i < inputSize; i++)
            {
                var srcBase = i * outputSize;

                for (var j = 0; j < outputSize; j++)
                {
                    dst[j * inputSize + i] = src[srcBase + j];
                }
            }
        }

        private static void FillDeterministic(float[] data, uint seed)
        {
            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;

                var normalized = (seed & 0x00FFFFFF) / 16777216f;
                data[i] = normalized * 2f - 1f;
            }
        }
    }
}