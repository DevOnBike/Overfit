using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class SlmRuntimeKernelCachedAttentionBenchmark
    {
        private const int OperationsPerInvoke = 8_192;

        private float[] _query = null!;
        private float[] _keys = null!;
        private float[] _values = null!;
        private float[] _output = null!;
        private float[] _scoreScratch = null!;
        private float _scale;
        private float _checksum;

        [Params(16, 64, 256, 512)]
        public int SequenceLength { get; set; }

        [Params(64)]
        public int HeadDimension { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _query = new float[HeadDimension];
            _keys = new float[SequenceLength * HeadDimension];
            _values = new float[SequenceLength * HeadDimension];
            _output = new float[HeadDimension];
            _scoreScratch = new float[SequenceLength];
            _scale = 1f / MathF.Sqrt(HeadDimension);

            FillDeterministic(_query, seed: 123);
            FillDeterministic(_keys, seed: 456);
            FillDeterministic(_values, seed: 789);

            CachedAttentionKernel.ComputeSingleHead(
            _query,
            _keys,
            _values,
            _output,
            _scoreScratch,
            SequenceLength,
            HeadDimension,
            _scale);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float CachedAttention_SingleHead()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                CachedAttentionKernel.ComputeSingleHead(
                _query,
                _keys,
                _values,
                _output,
                _scoreScratch,
                SequenceLength,
                HeadDimension,
                _scale);

                checksum += _output[i & (HeadDimension - 1)];
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