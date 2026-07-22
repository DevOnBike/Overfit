// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Benchmarks.Helpers;

namespace Benchmarks
{
    /// <summary>
    /// Measures <b>this machine's ceilings</b> — sustained memory bandwidth and peak arithmetic rate — so that
    /// every other kernel number in this project can be read as a fraction of what the hardware can do,
    /// instead of as a bare figure that sounds fast or slow depending on the reader's mood.
    ///
    /// <para><b>The question that motivated it.</b> On 2026-07-22 our Q4_K prefill GEMM measured 1.70 TFLOP/s
    /// on llama.cpp's own benchmark shape, against their 1.56 — so our kernel is not the reason prefill is
    /// behind. But "1.70" only becomes actionable next to a ceiling: at 40% of peak there is a kernel left to
    /// write, at 85% there is not, and the only remaining lever is a wider instruction set. Without this
    /// benchmark that distinction is guesswork, and CLAUDE.md is explicit that a perf claim without a
    /// measurement is a guess however confident the reasoning sounds.</para>
    ///
    /// <para><b>Reading the results.</b></para>
    /// <list type="bullet">
    ///   <item><see cref="ReadBandwidth"/> / <see cref="CopyBandwidth"/> / <see cref="TriadBandwidth"/> — the
    ///     STREAM-style ceiling that bounds <i>decode</i>, which is memory-bound: per token it must stream the
    ///     whole weight file once, so decode tok/s can never exceed bandwidth ÷ model bytes. Buffers are far
    ///     larger than any L3, so these are DRAM figures, not cache figures.</item>
    ///   <item><see cref="PeakFmaFloat"/> — the classic dense-FMA roofline, purely register-resident.</item>
    ///   <item><see cref="PeakIntegerDot"/> — the ceiling that actually bounds <i>our</i> quantized kernels,
    ///     which reach their MACs through <c>vpmaddubsw</c>/<c>vpmaddwd</c> rather than <c>vfmadd</c>. This is
    ///     the number the Q4_K/Q6_K GEMMs should be judged against; comparing them to the float FMA peak
    ///     flatters or maligns them depending on how the two instruction paths happen to be provisioned.</item>
    /// </list>
    ///
    /// <para><b>Counting convention</b> is <see cref="WorkAmount"/>'s: a multiply-accumulate is 2 operations,
    /// matching llama.cpp's <c>test-backend-ops</c>, so the columns are directly comparable across projects.
    /// The integer benchmark is credited with the <i>logical</i> int8 MACs it performs, on the same footing a
    /// quantized matmul is credited with the MACs of the dense matmul it stands in for.</para>
    ///
    /// <para>All five saturate every core, because a single-threaded ceiling would not bound anything this
    /// project runs — both prefill and decode are parallel over the whole machine.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*MachineRoofline*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class MachineRooflineBenchmark
    {
        /// <summary>Per-buffer size. Must comfortably exceed L3 so the bandwidth figures are DRAM, not cache.</summary>
        public const long BufferBytes = 256L * 1024 * 1024;

        /// <summary>Independent accumulator chains per thread — enough to cover FMA latency and saturate the ports.</summary>
        public const int Chains = 12;

        /// <summary>Inner iterations per thread, sized so one invocation lasts several milliseconds.</summary>
        public const int Iterations = 2_000_000;

        private const int FloatCount = (int)(BufferBytes / sizeof(float));

        /// <summary>Lanes per 256-bit vector of <see cref="float"/>.</summary>
        private const int FloatLanes = 8;

        /// <summary>Lanes per 256-bit vector of <see cref="sbyte"/> — the logical MAC count of one <c>vpmaddubsw</c>.</summary>
        private const int ByteLanes = 32;

        private float[] _a = null!;
        private float[] _b = null!;
        private float[] _c = null!;
        private int _workers;

        /// <summary>Consumed so the JIT cannot eliminate the measured loops.</summary>
        public float FloatSink;

        /// <summary>Consumed so the JIT cannot eliminate the measured loops.</summary>
        public int IntSink;

        /// <summary>
        /// Declares to <see cref="ThroughputColumn"/> how much work each benchmark performs, so the rate columns
        /// are derived from code that sits next to the loop being measured.
        /// </summary>
        public static WorkAmount GetWorkAmount(BenchmarkCase benchmarkCase)
        {
            var workers = Environment.ProcessorCount;

            // One logical MAC per lane per chain per iteration per thread.
            var macs = (long)Chains * Iterations * workers;

            return benchmarkCase.Descriptor.WorkloadMethod.Name switch
            {
                // One pass over one buffer.
                nameof(ReadBandwidth) => WorkAmount.Memory(BufferBytes),

                // Read one buffer, write another.
                nameof(CopyBandwidth) => WorkAmount.Memory(2L * BufferBytes),

                // STREAM triad: two reads and one write.
                nameof(TriadBandwidth) => WorkAmount.Memory(3L * BufferBytes),

                nameof(PeakFmaFloat) => new WorkAmount(2L * macs * FloatLanes, 0L),

                nameof(PeakIntegerDot) => new WorkAmount(2L * macs * ByteLanes, 0L),

                _ => default,
            };
        }

        [GlobalSetup]
        public void Setup()
        {
            _workers = Environment.ProcessorCount;

            _a = new float[FloatCount];
            _b = new float[FloatCount];
            _c = new float[FloatCount];

            var rng = new Random(20260722);

            for (var i = 0; i < FloatCount; i++)
            {
                _a[i] = (float)rng.NextDouble();
                _b[i] = (float)rng.NextDouble();
                _c[i] = (float)rng.NextDouble();
            }
        }

        /// <summary>Sustained read bandwidth: one streaming pass over a buffer far larger than L3.</summary>
        [Benchmark(Baseline = true)]
        public void ReadBandwidth()
        {
            var source = _a;
            var partials = new float[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                var (start, end) = SliceFor(worker);
                var span = source.AsSpan(start, end - start);
                var acc = Vector256<float>.Zero;
                var i = 0;

                for (; i <= span.Length - FloatLanes; i += FloatLanes)
                {
                    acc += Vector256.Create(span.Slice(i, FloatLanes));
                }

                var sum = Vector256.Sum(acc);

                for (; i < span.Length; i++)
                {
                    sum += span[i];
                }

                partials[worker] = sum;
            });

            var total = 0f;

            for (var i = 0; i < partials.Length; i++)
            {
                total += partials[i];
            }

            FloatSink = total;
        }

        /// <summary>Sustained copy bandwidth: one read stream plus one write stream.</summary>
        [Benchmark]
        public void CopyBandwidth()
        {
            var source = _a;
            var destination = _b;

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                var (start, end) = SliceFor(worker);
                source.AsSpan(start, end - start).CopyTo(destination.AsSpan(start, end - start));
            });

            FloatSink = destination[0];
        }

        /// <summary>STREAM triad <c>a = b + s·c</c>: two read streams plus one write stream, with real arithmetic.</summary>
        [Benchmark]
        public void TriadBandwidth()
        {
            var a = _a;
            var b = _b;
            var c = _c;
            var scalar = Vector256.Create(3.0f);

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                var (start, end) = SliceFor(worker);
                var i = start;

                for (; i <= end - FloatLanes; i += FloatLanes)
                {
                    var vb = Vector256.Create(b.AsSpan(i, FloatLanes));
                    var vc = Vector256.Create(c.AsSpan(i, FloatLanes));

                    (vb + (scalar * vc)).CopyTo(a.AsSpan(i, FloatLanes));
                }

                for (; i < end; i++)
                {
                    a[i] = b[i] + (3.0f * c[i]);
                }
            });

            FloatSink = a[0];
        }

        /// <summary>
        /// Peak float FMA rate with no memory traffic at all: independent register-resident accumulator chains,
        /// so the result is bounded by issue width and latency rather than by cache or DRAM.
        /// </summary>
        [Benchmark]
        public void PeakFmaFloat()
        {
            if (!Fma.IsSupported)
            {
                return;
            }

            var partials = new float[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                partials[worker] = FmaChains();
            });

            var sum = 0f;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            FloatSink = sum;
        }

        /// <summary>
        /// Peak int8 dot-product rate through the exact instruction pair our quantized kernels use —
        /// <c>vpmaddubsw</c> then <c>vpmaddwd</c>, accumulated into 32-bit lanes. This, not
        /// <see cref="PeakFmaFloat"/>, is the ceiling <c>Q4KGemvKernel.GemmTiled</c> is actually racing.
        /// </summary>
        [Benchmark]
        public void PeakIntegerDot()
        {
            if (!Avx2.IsSupported)
            {
                return;
            }

            var partials = new int[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                partials[worker] = IntegerDotChains();
            });

            var sum = 0;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            IntSink = sum;
        }

        /// <summary>
        /// <see cref="Chains"/> independent FMA chains held in <b>named locals</b>, one per accumulator.
        ///
        /// <para>The accumulators must not live in a <c>stackalloc</c> span. A first version of this benchmark
        /// put them there and measured 0.79 TFLOP/s — <i>below</i> the 1.70 TFLOP/s our real Q4_K matmul
        /// achieves, which is impossible for a loop that touches no memory. The span forced a load and a store
        /// per accumulator per iteration, so it measured L1 round-trips rather than FMA issue rate. Constant
        /// indices are what keep a value in a register, and that is precisely the property under test here.</para>
        /// </summary>
        private static float FmaChains()
        {
            var multiplicand = Vector256.Create(1.000001f);
            var addend = Vector256.Create(0.000001f);

            var a0 = Vector256.Create(1f);
            var a1 = Vector256.Create(2f);
            var a2 = Vector256.Create(3f);
            var a3 = Vector256.Create(4f);
            var a4 = Vector256.Create(5f);
            var a5 = Vector256.Create(6f);
            var a6 = Vector256.Create(7f);
            var a7 = Vector256.Create(8f);
            var a8 = Vector256.Create(9f);
            var a9 = Vector256.Create(10f);
            var a10 = Vector256.Create(11f);
            var a11 = Vector256.Create(12f);

            for (var iteration = 0; iteration < Iterations; iteration++)
            {
                a0 = Fma.MultiplyAdd(a0, multiplicand, addend);
                a1 = Fma.MultiplyAdd(a1, multiplicand, addend);
                a2 = Fma.MultiplyAdd(a2, multiplicand, addend);
                a3 = Fma.MultiplyAdd(a3, multiplicand, addend);
                a4 = Fma.MultiplyAdd(a4, multiplicand, addend);
                a5 = Fma.MultiplyAdd(a5, multiplicand, addend);
                a6 = Fma.MultiplyAdd(a6, multiplicand, addend);
                a7 = Fma.MultiplyAdd(a7, multiplicand, addend);
                a8 = Fma.MultiplyAdd(a8, multiplicand, addend);
                a9 = Fma.MultiplyAdd(a9, multiplicand, addend);
                a10 = Fma.MultiplyAdd(a10, multiplicand, addend);
                a11 = Fma.MultiplyAdd(a11, multiplicand, addend);
            }

            return Vector256.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
        }

        /// <summary>
        /// <see cref="Chains"/> independent <c>vpmaddubsw</c>+<c>vpmaddwd</c> chains in named locals, for the
        /// same register-residency reason as <see cref="FmaChains"/>.
        /// </summary>
        private static int IntegerDotChains()
        {
            var weights = Vector256.Create((byte)3);
            var activations = Vector256.Create((sbyte)5);
            var ones = Vector256.Create((short)1);

            var a0 = Vector256.Create(1);
            var a1 = Vector256.Create(2);
            var a2 = Vector256.Create(3);
            var a3 = Vector256.Create(4);
            var a4 = Vector256.Create(5);
            var a5 = Vector256.Create(6);
            var a6 = Vector256.Create(7);
            var a7 = Vector256.Create(8);
            var a8 = Vector256.Create(9);
            var a9 = Vector256.Create(10);
            var a10 = Vector256.Create(11);
            var a11 = Vector256.Create(12);

            for (var iteration = 0; iteration < Iterations; iteration++)
            {
                a0 = Avx2.Add(a0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a1 = Avx2.Add(a1, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a2 = Avx2.Add(a2, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a3 = Avx2.Add(a3, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a4 = Avx2.Add(a4, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a5 = Avx2.Add(a5, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a6 = Avx2.Add(a6, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a7 = Avx2.Add(a7, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a8 = Avx2.Add(a8, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a9 = Avx2.Add(a9, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a10 = Avx2.Add(a10, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                a11 = Avx2.Add(a11, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
            }

            return Vector256.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
        }

        /// <summary>Splits the buffer into one contiguous, vector-aligned slice per worker.</summary>
        private (int Start, int End) SliceFor(int worker)
        {
            var perWorker = (FloatCount / _workers / FloatLanes) * FloatLanes;
            var start = worker * perWorker;
            var end = worker == _workers - 1 ? FloatCount : start + perWorker;

            return (start, end);
        }
    }
}
