// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Diagnostics.Contracts;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Runtime;
using DevOnBike.Overfit.Evolutionary.Storage;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class MapElitesBenchmarks
    {
        private GridEliteArchive _archive = null!;
        private MapElites<DummyContext> _map = null!;
        private DummyContext _context;
        private SimpleDescriptorEvaluator _evaluator = null!;

        [Params(64, 256)]
        public int ParameterCount { get; set; }

        [Params(64, 256)]
        public int BatchSize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _archive = new GridEliteArchive(
                parameterCount: ParameterCount,
                binsPerDimension: [32, 32],
                descriptorMin: [0f, 0f],
                descriptorMax: [1f, 1f]);

            _evaluator = new SimpleDescriptorEvaluator();

            _map = new MapElites<DummyContext>(
                parameterCount: ParameterCount,
                batchSize: BatchSize,
                archive: _archive,
                evaluator: _evaluator,
                seed: 12345,
                mutationSigma: 0.02f,
                initialMin: -1f,
                initialMax: 1f,
                randomInjectionProbability: 0.05f);

            _context = new DummyContext();

            // Warm the archive so the benchmark reflects steady-state mutate+insert behavior.
            for (var i = 0; i < 8; i++)
            {
                _map.RunIteration(ref _context);
            }
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _map.Dispose();
            _archive.Dispose();
        }

        [Benchmark]
        public MapElitesIterationMetrics RunIteration()
        {
            return _map.RunIteration(ref _context);
        }

        private readonly struct DummyContext
        {
        }

        private sealed class SimpleDescriptorEvaluator : IBehaviorDescriptorEvaluator<DummyContext>
        {
            public float Evaluate(
                ReadOnlySpan<float> parameters,
                ref DummyContext context,
                Span<float> descriptor)
            {
                var sumSq = 0f;
                for (var i = 0; i < parameters.Length; i++)
                {
                    sumSq += parameters[i] * parameters[i];
                }

                descriptor[0] = Clamp01(0.5f + 0.25f * parameters[0]);
                descriptor[1] = Clamp01(0.5f + 0.25f * parameters[1]);

                return -sumSq;
            }

            private static float Clamp01(float value)
            {
                if (value < 0f) return 0f;
                if (value > 1f) return 1f;
                return value;
            }
        }
    }
}