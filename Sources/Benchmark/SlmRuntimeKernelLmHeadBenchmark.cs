using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class SlmRuntimeKernelLmHeadBenchmark
    {
        private const int OperationsPerInvoke = 64;

        private const int DModel = 768;
        private const int VocabSize = 40_478;

        private float[] _hidden = null!;
        private float[] _lmHeadWeights = null!;
        private float[] _logits = null!;
        private float _checksum;

        [GlobalSetup]
        public void Setup()
        {
            _hidden = new float[DModel];
            _lmHeadWeights = new float[DModel * VocabSize];
            _logits = new float[VocabSize];

            FillDeterministic(_hidden, seed: 111);
            FillDeterministic(_lmHeadWeights, seed: 222);

            SingleTokenProjectionKernel.ProjectWithoutBias(
            _hidden,
            _lmHeadWeights,
            _logits,
            DModel,
            VocabSize);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Project_LmHead_768_To_40478()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                SingleTokenProjectionKernel.ProjectWithoutBias(
                _hidden,
                _lmHeadWeights,
                _logits,
                DModel,
                VocabSize);

                checksum += _logits[i % VocabSize];
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