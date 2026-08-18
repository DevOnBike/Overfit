// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

        /// <summary>Inner iterations for the kernel-shaped chains, which do far more work per iteration.</summary>
        public const int ShapeIterations = 200_000;

        /// <summary>Independent chains in the kernel-shaped benchmarks — fewer, because each holds more live vectors.</summary>
        public const int ShapeChains = 4;

        /// <summary>`Mul(Blend(...), Sh32(...))` statements per chain, mirroring the kernel's iacc0 block.</summary>
        public const int ShapeMuls = 8;

        /// <summary>Live accumulator chains in the wide variants — more than the 16 ymm registers AVX2 offers.</summary>
        public const int WideChains = 16;

        /// <summary>Iterations for the wide variants, scaled down by the extra chains so runtime stays comparable.</summary>
        public const int WideIterations = ShapeIterations / 4;

        /// <summary>Backing buffer for the streamed-weight probes.</summary>
        public const long StreamBytes = 256L * 1024 * 1024;

        /// <summary>Per-worker window that stays resident in its own L2.</summary>
        public const long L2WindowBytes = 512L * 1024;

        /// <summary>Per-worker window that overflows L2 but stays within the shared L3.</summary>
        public const long L3WindowBytes = 4L * 1024 * 1024;

        private const int FloatCount = (int)(BufferBytes / sizeof(float));

        /// <summary>Lanes per 256-bit vector of <see cref="float"/>.</summary>
        private const int FloatLanes = 8;

        /// <summary>Lanes per 256-bit vector of <see cref="sbyte"/> — the logical MAC count of one <c>vpmaddubsw</c>.</summary>
        private const int ByteLanes = 32;

        /// <summary>Lanes per 512-bit vector of <see cref="float"/>.</summary>
        private const int FloatLanes512 = 16;

        /// <summary>Lanes per 512-bit vector of <see cref="sbyte"/>.</summary>
        private const int ByteLanes512 = 64;

        private float[] _a = null!;
        private float[] _b = null!;
        private float[] _c = null!;
        private byte[] _stream = null!;
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

                // Report nothing rather than a fictitious rate when the silicon lacks AVX-512: the body
                // returns immediately, so crediting it with work would show an infinite throughput.
                nameof(PeakFmaFloat512) => Avx512F.IsSupported
                    ? new WorkAmount(2L * macs * FloatLanes512, 0L)
                    : default,

                nameof(PeakIntegerDot512) => Avx512BW.IsSupported
                    ? new WorkAmount(2L * macs * ByteLanes512, 0L)
                    : default,

                // Each of the ShapeMuls statements is one vpmaddubsw over a full vector of int8 pairs.
                nameof(PeakQ4KShape) => new WorkAmount(
                    2L * ShapeChains * ShapeMuls * ShapeIterations * workers * ByteLanes, 0L),

                nameof(PeakQ4KShape512) => Avx512BW.IsSupported
                    ? new WorkAmount(2L * ShapeChains * ShapeMuls * ShapeIterations * workers * ByteLanes512, 0L)
                    : default,

                nameof(PeakIntegerDotWide) => new WorkAmount(
                    2L * WideChains * Iterations * workers * ByteLanes, 0L),

                // Same instruction mix and MAC count as PeakQ4KShape — only the weight source differs.
                nameof(PeakQ4KShapeFromL2) or nameof(PeakQ4KShapeFromL3) or nameof(PeakQ4KShapeFromDram) =>
                    new WorkAmount(2L * ShapeChains * ShapeMuls * ShapeIterations * workers * ByteLanes, 0L),

                nameof(PeakIntegerDotWide512) => Avx512BW.IsSupported
                    ? new WorkAmount(2L * WideChains * Iterations * workers * ByteLanes512, 0L)
                    : default,

                _ => default,
            };
        }

        [GlobalSetup]
        public void Setup()
        {
            // OVERFIT_ROOFLINE_WORKERS overrides the worker count, so the SINGLE-CORE ceiling can be measured
            // with the same kernel, the same panels and the same process shape as the all-core one.
            //
            // It exists because a per-thread claim needs a per-thread denominator. Dividing a one-thread
            // result by the all-core ceiling understates it by the core count; correcting for that with a
            // datasheet boost clock is a guess, and this project does not put guessed numbers under measured
            // ones. Two runs of this benchmark give the ratio directly, and the ratio is what says how much of
            // an imperfect parallel speed-up is the silicon clocking down rather than the code.
            var requested = Environment.GetEnvironmentVariable("OVERFIT_ROOFLINE_WORKERS");

            _workers = int.TryParse(requested, out var parsed) && parsed > 0
                ? Math.Min(parsed, Environment.ProcessorCount)
                : Environment.ProcessorCount;

            _a = new float[FloatCount];
            _b = new float[FloatCount];
            _c = new float[FloatCount];

            _stream = new byte[StreamBytes];
            var rng = new Random(20260722);
            rng.NextBytes(_stream);

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

        /// <summary>
        /// The 512-bit counterpart of <see cref="PeakFmaFloat"/> — the go/no-go gate for porting the quantized
        /// GEMMs to AVX-512.
        ///
        /// <para><b>What decides it.</b> Our Q4_K GEMM already runs at 78% of the 256-bit float ceiling, so the
        /// only way to move FFN — 69% of prefill — is to raise the ceiling. llama.cpp's AVX-512 build measured
        /// 1.60× its AVX2 build <i>on this machine</i>, so the instruction set pays there. What is not yet known
        /// is whether it pays <i>here</i>: many parts run one 512-bit FMA unit rather than two, or drop clocks
        /// under 512-bit load, and then the honest headroom is ~1.1× and the port is not worth writing. If this
        /// benchmark does not beat <see cref="PeakFmaFloat"/> by a clear margin, the plan dies here rather than
        /// after a kernel rewrite.</para>
        ///
        /// <para>Note this measures a burst, not a sustained thermal steady state; a part that downclocks only
        /// after seconds of 512-bit work will look better here than in production.</para>
        /// </summary>
        [Benchmark]
        public void PeakFmaFloat512()
        {
            if (!Avx512F.IsSupported)
            {
                return;
            }

            var partials = new float[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                partials[worker] = FmaChains512();
            });

            var sum = 0f;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            FloatSink = sum;
        }

        /// <summary>The 512-bit counterpart of <see cref="PeakIntegerDot"/> — the ceiling the quantized inner
        /// loop would race after an AVX-512 port.</summary>
        [Benchmark]
        public void PeakIntegerDot512()
        {
            if (!Avx512BW.IsSupported)
            {
                return;
            }

            var partials = new int[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                partials[worker] = IntegerDotChains512();
            });

            var sum = 0;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            IntSink = sum;
        }

        /// <summary>Named locals for the same register-residency reason as <see cref="FmaChains"/>.</summary>
        private static float FmaChains512()
        {
            var multiplicand = Vector512.Create(1.000001f);
            var addend = Vector512.Create(0.000001f);

            var a0 = Vector512.Create(1f);
            var a1 = Vector512.Create(2f);
            var a2 = Vector512.Create(3f);
            var a3 = Vector512.Create(4f);
            var a4 = Vector512.Create(5f);
            var a5 = Vector512.Create(6f);
            var a6 = Vector512.Create(7f);
            var a7 = Vector512.Create(8f);
            var a8 = Vector512.Create(9f);
            var a9 = Vector512.Create(10f);
            var a10 = Vector512.Create(11f);
            var a11 = Vector512.Create(12f);

            for (var iteration = 0; iteration < Iterations; iteration++)
            {
                a0 = Avx512F.FusedMultiplyAdd(a0, multiplicand, addend);
                a1 = Avx512F.FusedMultiplyAdd(a1, multiplicand, addend);
                a2 = Avx512F.FusedMultiplyAdd(a2, multiplicand, addend);
                a3 = Avx512F.FusedMultiplyAdd(a3, multiplicand, addend);
                a4 = Avx512F.FusedMultiplyAdd(a4, multiplicand, addend);
                a5 = Avx512F.FusedMultiplyAdd(a5, multiplicand, addend);
                a6 = Avx512F.FusedMultiplyAdd(a6, multiplicand, addend);
                a7 = Avx512F.FusedMultiplyAdd(a7, multiplicand, addend);
                a8 = Avx512F.FusedMultiplyAdd(a8, multiplicand, addend);
                a9 = Avx512F.FusedMultiplyAdd(a9, multiplicand, addend);
                a10 = Avx512F.FusedMultiplyAdd(a10, multiplicand, addend);
                a11 = Avx512F.FusedMultiplyAdd(a11, multiplicand, addend);
            }

            return Vector512.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
        }

        /// <summary>Named locals for the same register-residency reason as <see cref="IntegerDotChains"/>.</summary>
        private static int IntegerDotChains512()
        {
            var weights = Vector512.Create((byte)3);
            var activations = Vector512.Create((sbyte)5);
            var ones = Vector512.Create((short)1);

            var a0 = Vector512.Create(1);
            var a1 = Vector512.Create(2);
            var a2 = Vector512.Create(3);
            var a3 = Vector512.Create(4);
            var a4 = Vector512.Create(5);
            var a5 = Vector512.Create(6);
            var a6 = Vector512.Create(7);
            var a7 = Vector512.Create(8);
            var a8 = Vector512.Create(9);
            var a9 = Vector512.Create(10);
            var a10 = Vector512.Create(11);
            var a11 = Vector512.Create(12);

            for (var iteration = 0; iteration < Iterations; iteration++)
            {
                a0 = Avx512F.Add(a0, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a1 = Avx512F.Add(a1, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a2 = Avx512F.Add(a2, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a3 = Avx512F.Add(a3, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a4 = Avx512F.Add(a4, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a5 = Avx512F.Add(a5, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a6 = Avx512F.Add(a6, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a7 = Avx512F.Add(a7, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a8 = Avx512F.Add(a8, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a9 = Avx512F.Add(a9, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a10 = Avx512F.Add(a10, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                a11 = Avx512F.Add(a11, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
            }

            return Vector512.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
        }

        /// <summary>
        /// The ceiling for the instruction mix <c>Q4KGemvKernel.GemmTiled</c> actually issues, rather than for
        /// an idealised dot-product chain — the number that predicts what an AVX-512 port of that kernel can buy.
        ///
        /// <para><b>Why the plain integer peak is the wrong reference.</b> <see cref="PeakIntegerDot"/> reaches
        /// 32 MACs in three instructions. The real kernel spends five per group — <c>Blend</c>, two lane
        /// shuffles, <c>vpmaddubsw</c>, <c>Add</c> — because the repacked <c>block_q4_Kx8</c> layout must be
        /// rearranged into position before it can be multiplied, and three of those five contend for the shuffle
        /// port rather than the vector-ALU ports. That mix, not the dot-product itself, is what the kernel is
        /// racing, so this benchmark reproduces the eight-statement <c>iacc0</c> block verbatim.</para>
        ///
        /// <para>Comparing this against <see cref="PeakQ4KShape512"/> gives the port's headroom directly:
        /// whatever ratio the two show is roughly what widening the column tile to 512 bits can deliver, before
        /// any memory or scheduling effects. Measuring it costs 40 lines instead of the ~250 an AVX-512 kernel
        /// rewrite would, and today has twice punished reasoning about mechanism ahead of measuring it.</para>
        /// </summary>
        [Benchmark]
        public void PeakQ4KShape()
        {
            var partials = new int[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                partials[worker] = Q4KShapeChains();
            });

            var sum = 0;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            IntSink = sum;
        }

        /// <summary>The same instruction mix at 512 bits — two activation columns per instruction.</summary>
        [Benchmark]
        public void PeakQ4KShape512()
        {
            if (!Avx512BW.IsSupported)
            {
                return;
            }

            var partials = new int[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                partials[worker] = Q4KShapeChains512();
            });

            var sum = 0;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            IntSink = sum;
        }

        private static int Q4KShapeChains()
        {
            // Four decoded weight-nibble vectors and one broadcast activation vector, exactly as the kernel
            // holds them across its eight iacc0 statements.
            var w0 = Vector256.Create((byte)3);
            var w1 = Vector256.Create((byte)5);
            var w2 = Vector256.Create((byte)7);
            var w3 = Vector256.Create((byte)9);
            var act = Vector256.Create((sbyte)2);

            var a0 = Vector256<short>.Zero;
            var a1 = Vector256<short>.Zero;
            var a2 = Vector256<short>.Zero;
            var a3 = Vector256<short>.Zero;

            for (var iteration = 0; iteration < ShapeIterations; iteration++)
            {
                a0 = Step(a0, w0, w1, act);
                a1 = Step(a1, w1, w2, act);
                a2 = Step(a2, w2, w3, act);
                a3 = Step(a3, w3, w0, act);
            }

            return Vector256.Sum(Avx2.Add(Avx2.Add(a0, a1), Avx2.Add(a2, a3)).AsInt16());

            // Two of the kernel's eight statements, repeated four times: Blend + two shuffles + maddubs + Add.
            static Vector256<short> Step(Vector256<short> acc, Vector256<byte> lo, Vector256<byte> hi, Vector256<sbyte> act)
            {
                for (var repeat = 0; repeat < ShapeMuls / 2; repeat++)
                {
                    acc = Avx2.Add(acc, Mul256(Blend256(lo, Sh256(hi, 177)), Sh32_256(act, 0)));
                    acc = Avx2.Add(acc, Mul256(Blend256(Sh256(lo, 177), hi), Sh32_256(act, 85)));
                }

                return acc;
            }
        }

        private static int Q4KShapeChains512()
        {
            var w0 = Vector512.Create((byte)3);
            var w1 = Vector512.Create((byte)5);
            var w2 = Vector512.Create((byte)7);
            var w3 = Vector512.Create((byte)9);
            var act = Vector512.Create((sbyte)2);

            // Blend mask: the 256-bit form used Avx2.Blend with imm 170 (odd int32 lanes from the right
            // operand); at 512 bits that is the same pattern repeated over 16 lanes.
            var blendMask = Vector512.Create(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);

            var a0 = Vector512<short>.Zero;
            var a1 = Vector512<short>.Zero;
            var a2 = Vector512<short>.Zero;
            var a3 = Vector512<short>.Zero;

            for (var iteration = 0; iteration < ShapeIterations; iteration++)
            {
                a0 = Step(a0, w0, w1, act, blendMask);
                a1 = Step(a1, w1, w2, act, blendMask);
                a2 = Step(a2, w2, w3, act, blendMask);
                a3 = Step(a3, w3, w0, act, blendMask);
            }

            return Vector512.Sum(Avx512BW.Add(Avx512BW.Add(a0, a1), Avx512BW.Add(a2, a3)).AsInt16());

            static Vector512<short> Step(
                Vector512<short> acc, Vector512<byte> lo, Vector512<byte> hi, Vector512<sbyte> act, Vector512<int> mask)
            {
                for (var repeat = 0; repeat < ShapeMuls / 2; repeat++)
                {
                    acc = Avx512BW.Add(acc, Mul512(Blend512(lo, Sh512(hi, 177), mask), Sh32_512(act, 0)));
                    acc = Avx512BW.Add(acc, Mul512(Blend512(Sh512(lo, 177), hi, mask), Sh32_512(act, 85)));
                }

                return acc;
            }
        }

        /// <summary>
        /// Register-pressure probe: the body of <see cref="IntegerDotChains"/> verbatim, with the chain count
        /// raised from 12 to <see cref="WideChains"/> and nothing else changed.
        ///
        /// <para><b>What it settles.</b> Our real GEMM reaches 1.70 TFLOP/s against its own instruction mix's
        /// 4.63, and the suspect is that its state does not fit the register file: at four activation columns
        /// <c>GemmTiled</c> keeps 8 float accumulators live across the block loop, 8 integer accumulators
        /// across the sub-block loop and 8 more inside it — 24 vectors before a single weight, against 16 ymm.
        /// Counting registers is reasoning about mechanism, so it is measured: 12 chains plus 3 constants fit
        /// in 16 ymm, 16 chains do not, and both fit comfortably in 512-bit's 32 zmm. If the cliff is real,
        /// 256-bit falls between the two and 512-bit does not.</para>
        ///
        /// <para>An earlier version of this probe routed each step through a helper taking five vector
        /// parameters and reported 512-bit as <i>worse</i> than 256-bit — impossible if zmm's larger file
        /// helps at all, and the giveaway that a non-inlined call was making it time the calling convention
        /// rather than the register file. Everything here is inline and every accumulator is a named local.</para>
        /// </summary>
        [Benchmark]
        public void PeakIntegerDotWide()
        {
            var partials = new int[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                partials[worker] = IntegerDotChainsWide();
            });

            var sum = 0;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            IntSink = sum;
        }

        /// <summary>The same probe at 512 bits, where 16 chains still fit the 32-register file.</summary>
        [Benchmark]
        public void PeakIntegerDotWide512()
        {
            if (!Avx512BW.IsSupported)
            {
                return;
            }

            var partials = new int[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                partials[worker] = IntegerDotChainsWide512();
            });

            var sum = 0;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            IntSink = sum;
        }

        private static int IntegerDotChainsWide()
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
            var a12 = Vector256.Create(13);
            var a13 = Vector256.Create(14);
            var a14 = Vector256.Create(15);
            var a15 = Vector256.Create(16);

            for (var iteration = 0; iteration < Iterations; iteration++)
            {
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
                    a12 = Avx2.Add(a12, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                    a13 = Avx2.Add(a13, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                    a14 = Avx2.Add(a14, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                    a15 = Avx2.Add(a15, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), ones));
                }
            }

            return Vector256.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15);
        }

        private static int IntegerDotChainsWide512()
        {
            var weights = Vector512.Create((byte)3);
            var activations = Vector512.Create((sbyte)5);
            var ones = Vector512.Create((short)1);

            var a0 = Vector512.Create(1);
            var a1 = Vector512.Create(2);
            var a2 = Vector512.Create(3);
            var a3 = Vector512.Create(4);
            var a4 = Vector512.Create(5);
            var a5 = Vector512.Create(6);
            var a6 = Vector512.Create(7);
            var a7 = Vector512.Create(8);
            var a8 = Vector512.Create(9);
            var a9 = Vector512.Create(10);
            var a10 = Vector512.Create(11);
            var a11 = Vector512.Create(12);
            var a12 = Vector512.Create(13);
            var a13 = Vector512.Create(14);
            var a14 = Vector512.Create(15);
            var a15 = Vector512.Create(16);

            for (var iteration = 0; iteration < Iterations; iteration++)
            {
                {
                    a0 = Avx512F.Add(a0, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a1 = Avx512F.Add(a1, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a2 = Avx512F.Add(a2, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a3 = Avx512F.Add(a3, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a4 = Avx512F.Add(a4, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a5 = Avx512F.Add(a5, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a6 = Avx512F.Add(a6, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a7 = Avx512F.Add(a7, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a8 = Avx512F.Add(a8, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a9 = Avx512F.Add(a9, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a10 = Avx512F.Add(a10, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a11 = Avx512F.Add(a11, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a12 = Avx512F.Add(a12, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a13 = Avx512F.Add(a13, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a14 = Avx512F.Add(a14, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                    a15 = Avx512F.Add(a15, Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(weights, activations), ones));
                }
            }

            return Vector512.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15);
        }

        /// <summary>
        /// <see cref="PeakQ4KShape"/>'s instruction mix with the weight vectors <b>loaded from memory</b>
        /// instead of held in registers — the one thing the register-resident mix benchmark did not model, and
        /// the last untested candidate for the kernel's unexplained 2.7× residual.
        ///
        /// <para><b>Why it matters before anything else.</b> Ablation showed the kernel's non-arithmetic work
        /// (F16 scale decode 12%, scalar scale unpack 3.5%, nibble unpack ~0%) bounds at ~15%, nowhere near the
        /// 63% of runtime the residual represents. The real kernel streams 8×32 B of weight per sub-block —
        /// 12.7 MB per projection. If these probes land near the kernel's measured 1.70–1.95 TFLOP/s, the
        /// kernel is load-bound, and <b>an AVX-512 port would widen compute against a memory wall</b> — the
        /// same mistake the reverted AVX-512 decode port already made in this repo.</para>
        ///
        /// <para>The byte-per-MAC ratio matches the kernel: 128 B loaded per 1024 MACs. Each worker walks its
        /// own window so the three variants really do sit in L2, in L3 and in DRAM respectively rather than
        /// all sharing one hot region.</para>
        /// </summary>
        [Benchmark]
        public void PeakQ4KShapeFromL2()
        {
            StreamProbe(L2WindowBytes);
        }

        /// <summary>Same mix and byte-per-MAC ratio, from a window sized to live in L3.</summary>
        [Benchmark]
        public void PeakQ4KShapeFromL3()
        {
            StreamProbe(L3WindowBytes);
        }

        /// <summary>Same mix and byte-per-MAC ratio, from a window far larger than any cache.</summary>
        [Benchmark]
        public void PeakQ4KShapeFromDram()
        {
            StreamProbe(StreamBytes);
        }

        private void StreamProbe(long window)
        {
            var stream = _stream;
            var partials = new int[_workers];

            Parallel.For(0, _workers, new ParallelOptions { MaxDegreeOfParallelism = _workers }, worker =>
            {
                // When the window IS the whole buffer there is no room to give workers distinct bases, so they
                // share one stream. (Dividing by the zero-sized remainder is what made this probe throw.)
                var slack = StreamBytes - window;
                var start = slack <= 0 ? 0L : (long)worker * window % slack;

                partials[worker] = Q4KStreamChains(stream, start, window);
            });

            var sum = 0;

            for (var i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }

            IntSink = sum;
        }

        /// <summary>
        /// The <see cref="Q4KShapeChains"/> body with the four weight vectors reloaded from
        /// <paramref name="stream"/> every outer iteration. Statements are inline — a helper taking vector
        /// parameters is what invalidated the first register-pressure probe.
        /// </summary>
        private static int Q4KStreamChains(byte[] stream, long start, long window)
        {
            ref var origin = ref MemoryMarshal.GetArrayDataReference(stream);

            var act = Vector256.Create((sbyte)2);
            var act0 = Sh32_256(act, 0);
            var act1 = Sh32_256(act, 85);

            var a0 = Vector256<short>.Zero;
            var a1 = Vector256<short>.Zero;
            var a2 = Vector256<short>.Zero;
            var a3 = Vector256<short>.Zero;

            long offset = 0;

            for (var iteration = 0; iteration < ShapeIterations; iteration++)
            {
                ref var p = ref Unsafe.Add(ref origin, (nint)(start + offset));

                var w0 = Unsafe.ReadUnaligned<Vector256<byte>>(ref p);
                var w1 = Unsafe.ReadUnaligned<Vector256<byte>>(ref Unsafe.Add(ref p, 32));
                var w2 = Unsafe.ReadUnaligned<Vector256<byte>>(ref Unsafe.Add(ref p, 64));
                var w3 = Unsafe.ReadUnaligned<Vector256<byte>>(ref Unsafe.Add(ref p, 96));

                offset += 128;

                if (offset >= window - 128)
                {
                    offset = 0;
                }

                for (var repeat = 0; repeat < ShapeMuls / 2; repeat++)
                {
                    a0 = Avx2.Add(a0, Mul256(Blend256(w0, Sh256(w1, 177)), act0));
                    a0 = Avx2.Add(a0, Mul256(Blend256(Sh256(w0, 177), w1), act1));
                    a1 = Avx2.Add(a1, Mul256(Blend256(w1, Sh256(w2, 177)), act0));
                    a1 = Avx2.Add(a1, Mul256(Blend256(Sh256(w1, 177), w2), act1));
                    a2 = Avx2.Add(a2, Mul256(Blend256(w2, Sh256(w3, 177)), act0));
                    a2 = Avx2.Add(a2, Mul256(Blend256(Sh256(w2, 177), w3), act1));
                    a3 = Avx2.Add(a3, Mul256(Blend256(w3, Sh256(w0, 177)), act0));
                    a3 = Avx2.Add(a3, Mul256(Blend256(Sh256(w3, 177), w0), act1));
                }
            }

            return Vector256.Sum(Avx2.Add(Avx2.Add(a0, a1), Avx2.Add(a2, a3)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Sh256(Vector256<byte> v, [ConstantExpected] byte imm)
        {
            return Avx2.Shuffle(v.AsInt32(), imm).AsByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<sbyte> Sh32_256(Vector256<sbyte> v, [ConstantExpected] byte imm)
        {
            return Avx2.Shuffle(v.AsInt32(), imm).AsSByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Blend256(Vector256<byte> a, Vector256<byte> b)
        {
            return Avx2.Blend(a.AsInt32(), b.AsInt32(), 170).AsByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> Mul256(Vector256<byte> rhs, Vector256<sbyte> lhs)
        {
            return Avx2.MultiplyAddAdjacent(rhs, lhs);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> Sh512(Vector512<byte> v, [ConstantExpected] byte imm)
        {
            return Avx512F.Shuffle(v.AsInt32(), imm).AsByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<sbyte> Sh32_512(Vector512<sbyte> v, [ConstantExpected] byte imm)
        {
            return Avx512F.Shuffle(v.AsInt32(), imm).AsSByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<byte> Blend512(Vector512<byte> a, Vector512<byte> b, Vector512<int> mask)
        {
            return Avx512F.BlendVariable(a.AsInt32(), b.AsInt32(), mask).AsByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<short> Mul512(Vector512<byte> rhs, Vector512<sbyte> lhs)
        {
            return Avx512BW.MultiplyAddAdjacent(rhs, lhs);
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
