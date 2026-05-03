// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// Microbenchmarks for the SLM runtime building blocks used by future KV-cache decode.
    ///
    /// These benchmarks do not call GPT1Model and do not use ComputationGraph.
    /// They isolate the kernels we will use inside cached GenerateNextToken:
    ///
    /// - single-token projection / matvec,
    /// - cached single-head attention,
    /// - KeyValueCache read/write spans.
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*SlmRuntimeKernel*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class SlmRuntimeKernelProjectionBenchmark
    {
        private const int OperationsPerInvoke = 4_096;

        private float[] _input = null!;
        private float[] _weights = null!;
        private float[] _bias = null!;
        private float[] _output = null!;

        private int _inputSize;
        private int _outputSize;
        private float _checksum;

        [Params(64, 768)]
        public int InputSize { get; set; }

        [Params(64, 768)]
        public int OutputSize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _inputSize = InputSize;
            _outputSize = OutputSize;

            _input = new float[_inputSize];
            _weights = new float[_inputSize * _outputSize];
            _bias = new float[_outputSize];
            _output = new float[_outputSize];

            FillDeterministic(_input, seed: 123);
            FillDeterministic(_weights, seed: 456);
            FillDeterministic(_bias, seed: 789);

            SingleTokenProjectionKernel.Project(
                _input,
                _weights,
                _bias,
                _output,
                _inputSize,
                _outputSize);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Project_WithBias()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                SingleTokenProjectionKernel.Project(
                    _input,
                    _weights,
                    _bias,
                    _output,
                    _inputSize,
                    _outputSize);

                checksum += _output[i & (_output.Length - 1)];
            }

            _checksum = checksum;
            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Project_WithoutBias()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                SingleTokenProjectionKernel.ProjectWithoutBias(
                    _input,
                    _weights,
                    _output,
                    _inputSize,
                    _outputSize);

                checksum += _output[i & (_output.Length - 1)];
            }

            _checksum = checksum;
            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            GC.KeepAlive(_checksum);
        }

        private static void FillDeterministic(float[] data, int seed)
        {
            var rng = new Random(seed);

            for (var i = 0; i < data.Length; i++)
            {
                data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }
    }

}
