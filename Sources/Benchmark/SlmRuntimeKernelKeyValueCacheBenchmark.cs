using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class SlmRuntimeKernelKeyValueCacheBenchmark : IDisposable
    {
        private const int OperationsPerInvoke = 16_384;

        private KeyValueCache _cache = null!;
        private float[] _source = null!;
        private float _checksum;
        private bool _disposed;

        [Params(12)]
        public int LayerCount { get; set; }

        [Params(12)]
        public int HeadCount { get; set; }

        [Params(64, 256, 512)]
        public int MaxSequenceLength { get; set; }

        [Params(64)]
        public int HeadDimension { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _cache = KeyValueCache.Create(
            LayerCount,
            HeadCount,
            MaxSequenceLength,
            HeadDimension);

            _source = new float[HeadDimension];
            FillDeterministic(_source, seed: 123);

            var key = _cache.GetKeyWriteSpan(0, 0, 0);
            _source.CopyTo(key);

            var value = _cache.GetValueWriteSpan(0, 0, 0);
            _source.CopyTo(value);

            _cache.Advance();
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float WriteKeyValue_OnePosition()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                var position = i % MaxSequenceLength;
                var layer = i % LayerCount;
                var head = i % HeadCount;

                _source.CopyTo(_cache.GetKeyWriteSpan(layer, head, position));
                _source.CopyTo(_cache.GetValueWriteSpan(layer, head, position));

                checksum += _source[0];
            }

            _checksum = checksum;
            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float ReadKeyValue_OnePosition()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                var position = 0;
                var layer = i % LayerCount;
                var head = i % HeadCount;

                var key = _cache.GetKeyReadSpan(layer, head, position, length: 1);
                var value = _cache.GetValueReadSpan(layer, head, position, length: 1);

                checksum += key[0] + value[0];
            }

            _checksum = checksum;
            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _cache.Dispose();

            GC.KeepAlive(_checksum);
        }

        public void Dispose()
        {
            Cleanup();
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