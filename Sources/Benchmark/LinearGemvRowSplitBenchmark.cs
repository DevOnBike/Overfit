// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    /// <summary>
    /// Prices two ways of splitting a batch-1 dense layer across workers, before either is written into
    /// <c>LinearKernels</c> — the ceiling-first step, on the shape that actually costs something.
    ///
    /// <para><b>The subject.</b> VGG-16's first fully-connected layer is <c>Linear(25088 -> 4096)</c>. At batch
    /// 1 it performs 205 MFLOP and reads <b>411.0 MB of weights</b>, so it is a streaming problem and its unit
    /// is GB/s, not GFLOP/s. Measured inside VGG-16 it takes <b>9.99 ms = 41.1 GB/s</b>, against this box's
    /// measured read ceiling of <b>90.8 GB/s</b> (<see cref="MachineRooflineBenchmark"/>). It is 16% of the
    /// model.</para>
    ///
    /// <para><b>The hypothesis, and what would refute it.</b> Today the parallel dispatch gives each worker a
    /// range of output COLUMNS and each worker walks every input row, so it reads 64 contiguous floats (256 B)
    /// and then skips <c>outputSize * 4 = 16,384 B</c> to the next row, 25,088 times. With 4 KB pages every
    /// one of those reads lands on a different page, so the hardware prefetcher has nothing to follow and the
    /// dTLB cannot hold the working set. <b>If splitting by input ROW instead — which makes each worker read
    /// one contiguous slab — does not raise the achieved GB/s, then the access pattern is not the limit and
    /// this hypothesis is wrong.</b></para>
    ///
    /// <para><b>Why the row split is not obviously free.</b> Register accumulators require fixing a column
    /// block and looping rows, which is exactly the strided pattern. Reading sequentially requires fixing a row
    /// and sweeping all columns, which puts the accumulator in memory. It fits: 4096 floats is 16 KB against
    /// this core's 48 KB L1d, so the accumulator stays in L1 for the whole sweep and only the weight stream
    /// reaches DRAM. That is the trade being measured.</para>
    ///
    /// <para><b>Both arms use the same parallel primitive</b>, <see cref="OverfitParallel"/>, so the only lever
    /// between them is the decomposition. An arm on <c>Parallel.For</c> would have moved two things at once.
    /// The third arm is single-threaded, which prices the per-thread pattern with the dispatch removed.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*LinearGemvRowSplit*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public unsafe class LinearGemvRowSplitBenchmark : IDisposable
    {
        /// <summary>
        /// Shapes to price, as "inputSize x outputSize". The first three are VGG-16's own fully-connected
        /// stack; the fourth is a layer whose whole weight matrix fits in L2, where the strided read costs
        /// nothing and the row split should therefore NOT win. A gate calibrated only on the shape that
        /// motivated it is a gate calibrated on one point.
        /// </summary>
        [Params("4608x4096", "5120x4096", "5632x4096")]
        public string Shape { get; set; } = "25088x4096";

        private const int BatchSize = 1;

        private int InputSize;
        private int OutputSize;

        private TensorStorage<float> _weights = null!;
        private TensorStorage<float> _input = null!;
        private TensorStorage<float> _bias = null!;
        private TensorStorage<float> _output = null!;
        private TensorStorage<float> _partials = null!;
        private TensorStorage<float> _weightsTransposed = null!;

        private int _workers;

        [GlobalSetup]
        public void Setup()
        {
            var parts = Shape.Split('x');

            InputSize = int.Parse(parts[0]);
            OutputSize = int.Parse(parts[1]);
            _workers = Math.Min(OverfitParallel.MaxDegreeOfParallelism, InputSize);

            _weights = new TensorStorage<float>(InputSize * OutputSize, clearMemory: false);
            _input = new TensorStorage<float>(BatchSize * InputSize, clearMemory: false);
            _bias = new TensorStorage<float>(OutputSize, clearMemory: false);
            _output = new TensorStorage<float>(BatchSize * OutputSize, clearMemory: true);
            _partials = new TensorStorage<float>(_workers * OutputSize, clearMemory: true);

            // Production allocates this second, output-major copy in LinearLayer's constructor and passes
            // it on every call, so the baseline arm gets it too. Arm A must be what production runs.
            _weightsTransposed = new TensorStorage<float>(InputSize * OutputSize, clearMemory: false);

            Fill(_weights.AsSpan(), seed: 17);
            Fill(_input.AsSpan(), seed: 23);
            Fill(_bias.AsSpan(), seed: 31);

            var source = _weights.AsSpan();
            var transposed = _weightsTransposed.AsSpan();

            for (var i = 0; i < InputSize; i++)
            {
                for (var j = 0; j < OutputSize; j++)
                {
                    transposed[(j * InputSize) + i] = source[(i * OutputSize) + j];
                }
            }

            // A one-time correctness check, not a benchmark: the two decompositions must agree before either
            // timing means anything. Cheap enough to run in setup, and it fails the whole run if they do not.
            var reference = new float[BatchSize * OutputSize];
            var candidate = new float[BatchSize * OutputSize];

            LinearKernels.Forward(
                _input.AsSpan(),
                _weights.AsSpan(),
                _weightsTransposed.AsReadOnlySpan(),
                _bias.AsSpan(),
                reference,
                InputSize,
                OutputSize);

            RowSplit(candidate);

            for (var i = 0; i < reference.Length; i++)
            {
                // The two orders sum the same 25,088 products differently, so they cannot be bit-identical.
                // The bound is the textbook one for a K-term dot product: K * u * sum|a_i b_i|, and |a|,|b|
                // are below 1 here, so K * u * K is a safe over-estimate of the sum of absolute products.
                var bound = 4f * InputSize * 5.9604645e-8f * MathF.Sqrt(InputSize);

                if (MathF.Abs(reference[i] - candidate[i]) > bound)
                {
                    throw new InvalidOperationException(
                        $"row split disagrees at {i}: column split {reference[i]}, row split {candidate[i]}, "
                        + $"bound {bound}");
                }
            }
        }

        /// <summary>Today's shape: one worker owns a range of output columns and strides the weight matrix.</summary>
        [Benchmark(Baseline = true)]
        public void ColumnSplit()
        {
            LinearKernels.Forward(
                _input.AsSpan(),
                _weights.AsSpan(),
                _weightsTransposed.AsReadOnlySpan(),
                _bias.AsSpan(),
                _output.AsSpan(),
                InputSize,
                OutputSize);
        }

        /// <summary>The candidate: one worker owns a range of input rows and reads one contiguous slab.</summary>
        [Benchmark]
        public void RowSplit()
        {
            RowSplit(_output.AsSpan());
        }

        /// <summary>The candidate's per-thread pattern with the dispatch removed, for attribution.</summary>
        [Benchmark]
        public void RowSplitSingleThread()
        {
            fixed (float* weights = _weights.AsSpan(),
                   input = _input.AsSpan(),
                   bias = _bias.AsSpan(),
                   output = _output.AsSpan(),
                   partials = _partials.AsSpan())
            {
                new Span<float>(partials, OutputSize).Clear();

                AccumulateRows(input, weights, partials, 0, InputSize, OutputSize);

                for (var j = 0; j < OutputSize; j++)
                {
                    output[j] = bias[j] + partials[j];
                }
            }
        }

        private void RowSplit(Span<float> output)
        {
            fixed (float* weights = _weights.AsSpan(),
                   input = _input.AsSpan(),
                   bias = _bias.AsSpan(),
                   outputBase = output,
                   partials = _partials.AsSpan())
            {
                _partials.AsSpan().Clear();

                var ctx = new RowSplitContext(input, weights, partials, InputSize, OutputSize, _workers);

                OverfitParallel.For(0, _workers, 1, &RowSplitWorker, &ctx);

                // Reduction: workers x 4096 adds, negligible against 411 MB of weight traffic.
                for (var j = 0; j < OutputSize; j++)
                {
                    var sum = bias[j];

                    for (var w = 0; w < _workers; w++)
                    {
                        sum += partials[(w * OutputSize) + j];
                    }

                    outputBase[j] = sum;
                }
            }
        }

        private static void RowSplitWorker(int workerStart, int workerEnd, void* ctxPtr)
        {
            ref readonly var ctx = ref Unsafe.AsRef<RowSplitContext>(ctxPtr);

            for (var w = workerStart; w < workerEnd; w++)
            {
                var rowStart = (int)((long)ctx.InputSize * w / ctx.Workers);
                var rowEnd = (int)((long)ctx.InputSize * (w + 1) / ctx.Workers);

                AccumulateRows(
                    ctx.Input,
                    ctx.Weights,
                    ctx.Partials + ((long)w * ctx.OutputSize),
                    rowStart,
                    rowEnd,
                    ctx.OutputSize);
            }
        }

        /// <summary>
        /// Sweeps whole weight rows into an accumulator that stays in L1, so the weight stream is sequential.
        /// </summary>
        private static void AccumulateRows(
            float* input,
            float* weights,
            float* accumulator,
            int rowStart,
            int rowEnd,
            int outputSize)
        {
            const int Width = 16;
            const int BlockWidth = Width * 4;

            for (var i = rowStart; i < rowEnd; i++)
            {
                var x = Vector512.Create(input[i]);
                var row = weights + ((long)i * outputSize);
                var j = 0;

                for (; j <= outputSize - BlockWidth; j += BlockWidth)
                {
                    var a0 = Vector512.Load(accumulator + j);
                    var a1 = Vector512.Load(accumulator + j + Width);
                    var a2 = Vector512.Load(accumulator + j + (Width * 2));
                    var a3 = Vector512.Load(accumulator + j + (Width * 3));

                    a0 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(row + j), a0);
                    a1 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(row + j + Width), a1);
                    a2 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(row + j + (Width * 2)), a2);
                    a3 = Avx512F.FusedMultiplyAdd(x, Vector512.Load(row + j + (Width * 3)), a3);

                    Vector512.Store(a0, accumulator + j);
                    Vector512.Store(a1, accumulator + j + Width);
                    Vector512.Store(a2, accumulator + j + (Width * 2));
                    Vector512.Store(a3, accumulator + j + (Width * 3));
                }

                for (; j < outputSize; j++)
                {
                    accumulator[j] += input[i] * row[j];
                }
            }
        }

        [GlobalCleanup]
        public void Dispose()
        {
            _weights?.Dispose();
            _input?.Dispose();
            _bias?.Dispose();
            _output?.Dispose();
            _partials?.Dispose();
            _weightsTransposed?.Dispose();
        }

        private static void Fill(Span<float> values, int seed)
        {
            var state = (uint)(0x9E3779B9 + seed);

            for (var i = 0; i < values.Length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                values[i] = (((state & 0x00FFFFFF) / 16777216f) * 2f) - 1f;
            }
        }

        private readonly struct RowSplitContext
        {
            public readonly float* Input;
            public readonly float* Weights;
            public readonly float* Partials;
            public readonly int InputSize;
            public readonly int OutputSize;
            public readonly int Workers;

            public RowSplitContext(
                float* input,
                float* weights,
                float* partials,
                int inputSize,
                int outputSize,
                int workers)
            {
                Input = input;
                Weights = weights;
                Partials = partials;
                InputSize = inputSize;
                OutputSize = outputSize;
                Workers = workers;
            }
        }
    }
}
