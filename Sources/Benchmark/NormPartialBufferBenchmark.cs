// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace Benchmarks
{
    /// <summary>
    /// The one thing the LayerNorm / RmsNorm change actually altered: where the per-worker partial-gradient
    /// buffer lives. Everything else in those kernels is untouched, so this isolates the lever rather than
    /// measuring a whole backward pass around it.
    ///
    /// <para>Shape copied from the real code: one contiguous <c>workerCount × C</c> buffer, zeroed, each
    /// worker accumulating into its own slot, then the slots summed into the final gradient. The stack arm is
    /// what shipped before (up to 512 KB of frame at C = 4096 on a 32-worker box — the reason for the change);
    /// the pooled arm is what ships now.</para>
    ///
    /// <para>Deliberately NOT on the shared <c>BenchmarkConfig</c>. That job pins
    /// <c>InvocationCount=1 / UnrollFactor=1</c>, and the existing <c>LayerNormBenchmark</c> run under it came
    /// back with 6–13% standard deviation — wide enough to hide anything this change could plausibly do.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class NormPartialBufferBenchmark
    {
        private float[] _final = [];
        private int _workerCount;

        /// <summary>Feature width. 4096 is where the stack arm reached ~512 KB per buffer.</summary>
        [Params(128, 512, 4096)]
        public int C
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            _workerCount = OverfitParallel.WorkerCount;
            _final = new float[C];
        }

        [Benchmark(Baseline = true)]
        public float Stackalloc()
        {
            var slots = _workerCount * C;

            // What shipped before: the whole per-worker workspace on the caller's stack.
            Span<float> partial = stackalloc float[slots];
            partial.Clear();

            Accumulate(partial, _workerCount, C);

            return Merge(partial, _final, _workerCount, C);
        }

        [Benchmark]
        public float Pooled()
        {
            var slots = _workerCount * C;

            using var buffer = new PooledBuffer<float>(slots, clearMemory: true);
            var partial = buffer.Span;

            Accumulate(partial, _workerCount, C);

            return Merge(partial, _final, _workerCount, C);
        }

        /// <summary>Each worker writes only its own slot — the race-free pattern the real kernels use.</summary>
        private static void Accumulate(Span<float> partial, int workerCount, int c)
        {
            for (var w = 0; w < workerCount; w++)
            {
                var slot = partial.Slice(w * c, c);
                for (var i = 0; i < c; i++)
                {
                    slot[i] += i * 0.5f;
                }
            }
        }

        private static float Merge(Span<float> partial, Span<float> final, int workerCount, int c)
        {
            final.Clear();
            for (var w = 0; w < workerCount; w++)
            {
                TensorPrimitives.Add(final, partial.Slice(w * c, c), final);
            }

            return final[0];
        }
    }
}
