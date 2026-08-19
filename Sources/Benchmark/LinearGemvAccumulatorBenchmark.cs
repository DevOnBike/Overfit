// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace Benchmarks
{
    /// <summary>
    /// VGG-16's <c>fc1</c> at its real shape — <c>25088 -> 4096</c>, batch 1, a 411 MB weight matrix.
    ///
    /// <para><b>Why this layer.</b> Measured 2026-08-19 on the finished tree, VGG-16 costs 27.30 ms across 16
    /// cores and <b>fc1 alone is 6.97 ms of it — 25.5% of the whole model in one layer</b>, more than any
    /// convolution. It also scales worst of anything in the model: 12.1 ms on one core against 8.58 ms for
    /// all three dense layers on sixteen, i.e. <b>1.41x</b>.</para>
    ///
    /// <para><b>The premise this benchmark exists to test, stated before it runs.</b> 411 MB in 6.97 ms is
    /// <b>59.0 GB/s</b>, against a 90.8 GB/s DRAM read ceiling measured elsewhere in this repository — 65%.
    /// If DRAM were the limit the layer would sit at the ceiling, so something else binds. Reading
    /// <c>LinearKernels.AccumulateRowsAvx512</c> gives a candidate: it loads four accumulator vectors and
    /// stores four back <b>for every input row</b>, so each four weight loads carry eight extra memory
    /// operations. Those hit L1 and are individually cheap, but they occupy load and store ports.</para>
    ///
    /// <para><b>What would refute it.</b> <see cref="StreamCeiling"/> runs the identical worker and slab
    /// structure over the identical 411 MB and does nothing but accumulate into registers. If the current
    /// kernel is already close to that number, the accumulator traffic is not the constraint and the blocked
    /// arms below will return nothing — which is the answer, and cheaper than shipping the rewrite to find
    /// out.</para>
    ///
    /// <para><b>Its own job, because the shared one cannot measure this.</b> <c>BenchmarkConfig</c> uses
    /// <c>WarmupCount(5)</c> with <c>InvocationCount(1)</c>, and five warmups do not reliably reach tier-1
    /// on a call this long — the same trap read VGG-16 at 5.929 ms and then 1.482 ms once the warmup was
    /// raised.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*LinearGemvAccumulator*"
    /// </summary>
    [Config(typeof(Config))]
    [MemoryDiagnoser]
    public unsafe class LinearGemvAccumulatorBenchmark
    {
        private const int InputSize = 25088;
        private const int OutputSize = 4096;

        private float[] _input = null!;
        private float[] _weights = null!;
        private float[] _bias = null!;
        private float[] _output = null!;

        private sealed class Config : ManualConfig
        {
            public Config()
            {
                AddJob(Job.Default
                    .WithWarmupCount(25)
                    .WithIterationCount(15)
                    .WithInvocationCount(1)
                    .WithUnrollFactor(1));
            }
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260819);

            _input = new float[InputSize];
            _weights = new float[(long)InputSize * OutputSize];
            _bias = new float[OutputSize];
            _output = new float[OutputSize];

            for (var i = 0; i < InputSize; i++)
            {
                _input[i] = (float)rng.NextDouble();
            }

            // Real values rather than zeros: a denormal or a NaN through an FMA chain costs on some parts,
            // and uninitialised pool memory has produced exactly that contamination here before.
            for (long i = 0; i < _weights.LongLength; i++)
            {
                _weights[i] = (float)(rng.NextDouble() - 0.5);
            }
        }

        /// <summary>The shipped kernel: one accumulator load and store per weight vector, per input row.</summary>
        [Benchmark(Baseline = true)]
        public float Current()
        {
            LinearKernels.ForwardRowsParallel(_input, _weights, _bias, _output, InputSize, OutputSize);

            return _output[0];
        }

        /// <summary>
        /// The ceiling for this shape: the same workers over the same slabs, reading every weight once and
        /// accumulating into registers only. No accumulator array, no FMA against the input vector.
        /// </summary>
        [Benchmark]
        public float StreamCeiling()
        {
            var workers = Math.Min(OverfitParallel.MaxDegreeOfParallelism, InputSize);

            using var partials = new PooledBuffer<float>(workers, clearMemory: true);

            fixed (float* weights = _weights, sink = partials.Span)
            {
                var context = new SweepContext(null, weights, sink, InputSize, OutputSize, workers, 0);

                OverfitParallel.For(0, workers, 1, &StreamWorker, &context);

                var total = 0f;

                for (var w = 0; w < workers; w++)
                {
                    total += sink[w];
                }

                return total;
            }
        }

        /// <summary>Accumulator held across two input rows, so its traffic halves.</summary>
        [Benchmark]
        public float Blocked2() => Blocked(2);

        /// <summary>Accumulator held across four input rows.</summary>
        [Benchmark]
        public float Blocked4() => Blocked(4);

        /// <summary>Accumulator held across eight input rows.</summary>
        [Benchmark]
        public float Blocked8() => Blocked(8);

        private float Blocked(int rowBlock)
        {
            var workers = Math.Min(OverfitParallel.MaxDegreeOfParallelism, InputSize);

            using var partials = new PooledBuffer<float>(workers * OutputSize, clearMemory: true);

            fixed (float* input = _input, weights = _weights, sink = partials.Span, output = _output)
            {
                var context = new SweepContext(input, weights, sink, InputSize, OutputSize, workers, rowBlock);

                OverfitParallel.For(0, workers, 1, &BlockedWorker, &context);

                for (var j = 0; j < OutputSize; j++)
                {
                    var sum = 0f;

                    for (var w = 0; w < workers; w++)
                    {
                        sum += sink[((long)w * OutputSize) + j];
                    }

                    output[j] = sum;
                }

                return output[0];
            }
        }

        private static void StreamWorker(int workerStart, int workerEnd, void* contextPtr)
        {
            ref readonly var context = ref Unsafe.AsRef<SweepContext>(contextPtr);

            for (var w = workerStart; w < workerEnd; w++)
            {
                var rowStart = (int)((long)context.InputSize * w / context.Workers);
                var rowEnd = (int)((long)context.InputSize * (w + 1) / context.Workers);
                var outputSize = context.OutputSize;

                var s0 = Vector512<float>.Zero;
                var s1 = Vector512<float>.Zero;
                var s2 = Vector512<float>.Zero;
                var s3 = Vector512<float>.Zero;

                for (var i = rowStart; i < rowEnd; i++)
                {
                    var row = context.Weights + ((long)i * outputSize);

                    for (var j = 0; j <= outputSize - 64; j += 64)
                    {
                        s0 += Vector512.Load(row + j);
                        s1 += Vector512.Load(row + j + 16);
                        s2 += Vector512.Load(row + j + 32);
                        s3 += Vector512.Load(row + j + 48);
                    }
                }

                context.Partials[w] = Vector512.Sum(s0 + s1 + s2 + s3);
            }
        }

        /// <summary>
        /// The candidate: <c>RowBlock</c> input rows share one accumulator load and store, so the
        /// accumulator's L1 traffic falls by that factor while the weight stream is unchanged.
        /// </summary>
        private static void BlockedWorker(int workerStart, int workerEnd, void* contextPtr)
        {
            ref readonly var context = ref Unsafe.AsRef<SweepContext>(contextPtr);

            for (var w = workerStart; w < workerEnd; w++)
            {
                var rowStart = (int)((long)context.InputSize * w / context.Workers);
                var rowEnd = (int)((long)context.InputSize * (w + 1) / context.Workers);
                var outputSize = context.OutputSize;
                var block = context.RowBlock;
                var accumulator = context.Partials + ((long)w * outputSize);

                var i = rowStart;

                for (; i <= rowEnd - block; i += block)
                {
                    for (var j = 0; j <= outputSize - 32; j += 32)
                    {
                        var a0 = Vector512.Load(accumulator + j);
                        var a1 = Vector512.Load(accumulator + j + 16);

                        for (var b = 0; b < block; b++)
                        {
                            var x = Vector512.Create(context.Input[i + b]);
                            var row = context.Weights + ((long)(i + b) * outputSize) + j;

                            a0 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(row), a0);
                            a1 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(row + 16), a1);
                        }

                        Vector512.Store(a0, accumulator + j);
                        Vector512.Store(a1, accumulator + j + 16);
                    }
                }

                for (; i < rowEnd; i++)
                {
                    var x = context.Input[i];
                    var row = context.Weights + ((long)i * outputSize);

                    for (var j = 0; j < outputSize; j++)
                    {
                        accumulator[j] += x * row[j];
                    }
                }
            }
        }

        private readonly struct SweepContext
        {
            public readonly float* Input;
            public readonly float* Weights;
            public readonly float* Partials;
            public readonly int InputSize;
            public readonly int OutputSize;
            public readonly int Workers;
            public readonly int RowBlock;

            public SweepContext(
                float* input,
                float* weights,
                float* partials,
                int inputSize,
                int outputSize,
                int workers,
                int rowBlock)
            {
                Input = input;
                Weights = weights;
                Partials = partials;
                InputSize = inputSize;
                OutputSize = outputSize;
                Workers = workers;
                RowBlock = rowBlock;
            }
        }
    }
}
