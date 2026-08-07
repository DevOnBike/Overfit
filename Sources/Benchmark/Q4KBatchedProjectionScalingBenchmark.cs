// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// How the batched Q4_K projection scales with ROW COUNT, from 1 to 128 — the sizes a speculative verify
    /// actually asks for, not the 512–672 of a prefill.
    ///
    /// <para><b>The question.</b> If batching amortised perfectly (DRAM-bound), time(rows=R) ≈ time(rows=1) and
    /// the per-row cost falls ~R×. If the kernel is compute-bound per row, time grows linearly and batching buys
    /// nothing. <see cref="Q4KDotKernel.ProjectBatched"/> reads each weight super-block from DRAM once and then
    /// <b>re-decodes</b> it — unpack scales/mins/nibbles, plus the F32 tail — once per row;
    /// <see cref="Q4KDotKernel.ProjectBatchedWeightStationary"/> decodes once and dots many. The ratio between
    /// them, as a function of rows, is what says whether the speculative verify's per-row cost is a fixable
    /// inefficiency or the shape of the problem.</para>
    ///
    /// <para><b>Why it is here and not in the test project.</b> It was
    /// <c>Tests/LanguageModels/Runtime/Parity/Q4KBatchedProjectionScalingBench.cs</c>, an xUnit test with
    /// <b>no assertions at all</b> — it could not fail, so it verified nothing and merely occupied the release
    /// gate. It also hand-rolled its own warm-up, best-of-8 minimum and a rows=1 canary, which is
    /// BenchmarkDotNet's job and which BenchmarkDotNet does with statistics this could not produce: it had no
    /// baseline ratio, no standard deviation, no outlier detection, and shared a process with whatever xUnit
    /// scheduled alongside it. Its own doc comment conceded the point — "flip to [Fact] and run on a COLD, idle
    /// box only".</para>
    ///
    /// <para><b>Job.</b> Not the shared <see cref="BenchmarkConfig"/>: at rows=1 this is a sub-millisecond
    /// operation, which is exactly the case CLAUDE.md warns about — <c>InvocationCount=1</c> leaves a
    /// microbenchmark measuring timer noise, and produced a phantom 1.61× regression once. Default invocation
    /// counts let BenchmarkDotNet pick a batch size per row count.</para>
    ///
    /// <para>Shape is 2048 → 8192, ~9.4 MB of Q4_K, deliberately larger than L2 so the DRAM behaviour is real.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*Q4KBatchedProjectionScaling*"
    /// </summary>
    [SimpleJob(warmupCount: 5, iterationCount: 15)]
    [MemoryDiagnoser]
    public class Q4KBatchedProjectionScalingBenchmark
    {
        private const int InputSize = 2048;
        private const int OutputSize = 8192;

        /// <summary>
        /// Row counts a speculative verify submits. 1 is the decode-equivalent floor and the canary: it does the
        /// least work, so if it moves between runs the box moved rather than the kernel.
        /// </summary>
        [Params(1, 2, 4, 8, 16, 64, 128)]
        public int Rows
        {
            get; set;
        }

        private Q4KWeight _weight = null!;
        private float[] _input = null!;
        private float[] _output = null!;
        private sbyte[] _quants = null!;
        private float[] _scales = null!;
        private short[] _bsums = null!;
        private int _superBlocksPerRow;

        /// <summary>One projection is <c>2 · rows · inputSize · outputSize</c> MACs, so the TFLOP/s column is
        /// directly comparable with the prefill benchmark's.</summary>
        public static WorkAmount GetWorkAmount(BenchmarkCase benchmarkCase)
        {
            return WorkAmount.Matmul((int)benchmarkCase.Parameters["Rows"], InputSize, OutputSize);
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260807);

            // A real quantization rather than random bytes. The original filled the block buffer with
            // `Random.NextBytes`, which is fine for a data-independent kernel but can decode to denormal or NaN
            // scales — and a denormal takes a different path in hardware, which is a poor thing to have inside
            // the measurement.
            var f32 = new float[(long)OutputSize * InputSize];

            for (var i = 0; i < f32.Length; i++)
            {
                f32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            _weight = new Q4KWeight(GgmlQuant.QuantizeQ4_K(f32, InputSize, OutputSize), InputSize, OutputSize);
            _superBlocksPerRow = _weight.SuperBlocksPerRow;

            _input = new float[(long)Rows * InputSize];

            for (var i = 0; i < _input.Length; i++)
            {
                _input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            _output = new float[(long)Rows * OutputSize];
            _quants = new sbyte[(long)Rows * InputSize];
            _scales = new float[(long)Rows * _superBlocksPerRow];
            _bsums = new short[(long)Rows * _superBlocksPerRow * Q4KDotKernel.GroupsPerSuperBlock];
        }

        /// <summary>Decode the weight super-block once per ROW. The baseline, because it is what the kernel
        /// docs' "~3×" claim is measured against.</summary>
        [Benchmark(Baseline = true)]
        public void ReDecodePerRow()
        {
            Q4KDotKernel.ProjectBatched(_input, Rows, _weight, ReadOnlySpan<float>.Empty, _output,
                _quants, _scales, _bsums);
        }

        /// <summary>Decode each super-block once and dot it against every row in the tile.</summary>
        [Benchmark]
        public void WeightStationary()
        {
            Q4KDotKernel.ProjectBatchedWeightStationary(_input, Rows, _weight, ReadOnlySpan<float>.Empty, _output,
                _quants, _scales, _bsums);
        }
    }
}
