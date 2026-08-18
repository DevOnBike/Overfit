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
    /// Prices the candidate SGEMM micro-kernel <b>tile shapes</b> before any of them is written into
    /// <c>Conv2DGemmKernels</c> — the ceiling-first step that made the LLM prefill track pay.
    ///
    /// <para><b>The question.</b> The conv GEMM runs at 566 GFLOP/s against a 2190 GFLOP/s machine ceiling
    /// (26%). Its micro-kernel is <c>Mr=8 × Nr=8</c>, which per k-step issues <b>8 FMAs against 9 loads</b>
    /// (one B vector + eight A broadcasts) — arithmetic that says it is limited by the load ports, not the FMA
    /// units. Widening the tile amortises the B loads over more FMAs. Whether that actually pays, and which
    /// shape pays most on this silicon, is measured here rather than argued.</para>
    ///
    /// <para><b>What each shape costs per k-step</b> (V = floats per vector: 8 for AVX2, 16 for AVX-512):</para>
    /// <list type="table">
    ///   <item><term>8×1V</term><description>8 accumulators · 1 B load + 8 broadcasts = 9 loads · 8 FMAs → 0.89 FMA/load <b>(today's shape)</b></description></item>
    ///   <item><term>6×2V</term><description>12 accumulators · 2 + 6 = 8 loads · 12 FMAs → 1.50</description></item>
    ///   <item><term>4×3V</term><description>12 accumulators · 3 + 4 = 7 loads · 12 FMAs → 1.71</description></item>
    ///   <item><term>8×2V</term><description>16 accumulators · 2 + 8 = 10 loads · 16 FMAs → 1.60 (needs AVX-512's 32 registers to hold)</description></item>
    /// </list>
    ///
    /// <para><b>Method notes that this project has paid to learn.</b> Accumulators are <b>named locals</b>, never
    /// a <c>stackalloc</c> span — a span forces an L1 round-trip per accumulator per iteration and measures
    /// cache latency instead of issue rate (it cost a 2.8× error in an earlier roofline). A and B panels are
    /// sized to sit in L1 so the result reflects instruction issue, not memory bandwidth; the real kernel's
    /// memory behaviour is a separate question from its tile shape.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*GemmMicroKernelShape*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class GemmMicroKernelShapeBenchmark
    {
        /// <summary>Contraction length per call — a typical VGG im2col K (inC 256 · 3 · 3 = 2304 rounds here).</summary>
        public const int K = 2304;

        /// <summary>Times the tile is swept, to give BenchmarkDotNet a multi-millisecond subject.</summary>
        public const int Sweeps = 2000;

        // 12, because the 12x32 shape below needs twelve A rows. MLAS and BLIS both hold 24
        // accumulators on AVX-512; this buffer has to be able to feed that.
        private const int MaxRows = 12;
        private const int MaxCols = 48;

        private float[] _a = null!;
        private float[] _b = null!;
        private float[] _c = null!;

        public float Sink;

        /// <summary>
        /// FLOPs for the shape a benchmark method encodes, read from its name suffix (rows × cols).
        /// A MAC counts as 2, matching <see cref="Throughput"/> and the rest of this project.
        /// </summary>
        public static WorkAmount GetWorkAmount(BenchmarkCase benchmarkCase)
        {
            var (rows, cols) = benchmarkCase.Descriptor.WorkloadMethod.Name switch
            {
                nameof(Avx2_8x8) => (8, 8),
                nameof(Avx2_6x16) => (6, 16),
                nameof(Avx2_4x24) => (4, 24),
                nameof(Avx512_8x16) => (8, 16),
                nameof(Avx512_8x32) => (8, 32),
                nameof(Avx512_8x32_SpanAccumulators) => (8, 32),
                nameof(Avx512_12x32) => (12, 32),
                nameof(Avx512_6x48) => (6, 48),
                _ => (0, 0),
            };

            return WorkAmount.Matmul(rows, K * (long)Sweeps, cols);
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260723);

            _a = new float[MaxRows * K];
            _b = new float[K * MaxCols];
            _c = new float[MaxRows * MaxCols];

            for (var i = 0; i < _a.Length; i++)
            {
                _a[i] = (float)((rng.NextDouble() * 2.0) - 1.0);
            }

            for (var i = 0; i < _b.Length; i++)
            {
                _b[i] = (float)((rng.NextDouble() * 2.0) - 1.0);
            }
        }

        /// <summary>Today's shape: 8 rows × one 256-bit vector. 9 loads per 8 FMAs.</summary>
        [Benchmark(Baseline = true)]
        public unsafe void Avx2_8x8()
        {
            fixed (float* a = _a, b = _b, c = _c)
            {
                for (var s = 0; s < Sweeps; s++)
                {
                    var a0 = Vector256<float>.Zero;
                    var a1 = Vector256<float>.Zero;
                    var a2 = Vector256<float>.Zero;
                    var a3 = Vector256<float>.Zero;
                    var a4 = Vector256<float>.Zero;
                    var a5 = Vector256<float>.Zero;
                    var a6 = Vector256<float>.Zero;
                    var a7 = Vector256<float>.Zero;

                    for (var k = 0; k < K; k++)
                    {
                        var bv = Vector256.Load(b + (k * 8));
                        var ak = a + k;

                        a0 = Fma.MultiplyAdd(Vector256.Create(ak[0 * K]), bv, a0);
                        a1 = Fma.MultiplyAdd(Vector256.Create(ak[1 * K]), bv, a1);
                        a2 = Fma.MultiplyAdd(Vector256.Create(ak[2 * K]), bv, a2);
                        a3 = Fma.MultiplyAdd(Vector256.Create(ak[3 * K]), bv, a3);
                        a4 = Fma.MultiplyAdd(Vector256.Create(ak[4 * K]), bv, a4);
                        a5 = Fma.MultiplyAdd(Vector256.Create(ak[5 * K]), bv, a5);
                        a6 = Fma.MultiplyAdd(Vector256.Create(ak[6 * K]), bv, a6);
                        a7 = Fma.MultiplyAdd(Vector256.Create(ak[7 * K]), bv, a7);
                    }

                    a0.Store(c);
                    a1.Store(c + 8);
                    a2.Store(c + 16);
                    a3.Store(c + 24);
                    a4.Store(c + 32);
                    a5.Store(c + 40);
                    a6.Store(c + 48);
                    a7.Store(c + 56);
                }
            }

            Sink = _c[0];
        }

        /// <summary>6 rows × two 256-bit vectors: 12 accumulators, 8 loads per 12 FMAs.</summary>
        [Benchmark]
        public unsafe void Avx2_6x16()
        {
            fixed (float* a = _a, b = _b, c = _c)
            {
                for (var s = 0; s < Sweeps; s++)
                {
                    Vector256<float> a00 = default, a01 = default, a10 = default, a11 = default;
                    Vector256<float> a20 = default, a21 = default, a30 = default, a31 = default;
                    Vector256<float> a40 = default, a41 = default, a50 = default, a51 = default;

                    for (var k = 0; k < K; k++)
                    {
                        var b0 = Vector256.Load(b + (k * 16));
                        var b1 = Vector256.Load(b + (k * 16) + 8);
                        var ak = a + k;

                        var r = Vector256.Create(ak[0 * K]);
                        a00 = Fma.MultiplyAdd(r, b0, a00);
                        a01 = Fma.MultiplyAdd(r, b1, a01);
                        r = Vector256.Create(ak[1 * K]);
                        a10 = Fma.MultiplyAdd(r, b0, a10);
                        a11 = Fma.MultiplyAdd(r, b1, a11);
                        r = Vector256.Create(ak[2 * K]);
                        a20 = Fma.MultiplyAdd(r, b0, a20);
                        a21 = Fma.MultiplyAdd(r, b1, a21);
                        r = Vector256.Create(ak[3 * K]);
                        a30 = Fma.MultiplyAdd(r, b0, a30);
                        a31 = Fma.MultiplyAdd(r, b1, a31);
                        r = Vector256.Create(ak[4 * K]);
                        a40 = Fma.MultiplyAdd(r, b0, a40);
                        a41 = Fma.MultiplyAdd(r, b1, a41);
                        r = Vector256.Create(ak[5 * K]);
                        a50 = Fma.MultiplyAdd(r, b0, a50);
                        a51 = Fma.MultiplyAdd(r, b1, a51);
                    }

                    a00.Store(c);
                    a01.Store(c + 8);
                    a10.Store(c + 16);
                    a11.Store(c + 24);
                    a20.Store(c + 32);
                    a21.Store(c + 40);
                    a30.Store(c + 48);
                    a31.Store(c + 56);
                    a40.Store(c + 64);
                    a41.Store(c + 72);
                    a50.Store(c + 80);
                    a51.Store(c + 88);
                }
            }

            Sink = _c[0];
        }

        /// <summary>4 rows × three 256-bit vectors: 12 accumulators, 7 loads per 12 FMAs — the best AVX2 ratio.</summary>
        [Benchmark]
        public unsafe void Avx2_4x24()
        {
            fixed (float* a = _a, b = _b, c = _c)
            {
                for (var s = 0; s < Sweeps; s++)
                {
                    Vector256<float> a00 = default, a01 = default, a02 = default;
                    Vector256<float> a10 = default, a11 = default, a12 = default;
                    Vector256<float> a20 = default, a21 = default, a22 = default;
                    Vector256<float> a30 = default, a31 = default, a32 = default;

                    for (var k = 0; k < K; k++)
                    {
                        var b0 = Vector256.Load(b + (k * 24));
                        var b1 = Vector256.Load(b + (k * 24) + 8);
                        var b2 = Vector256.Load(b + (k * 24) + 16);
                        var ak = a + k;

                        var r = Vector256.Create(ak[0 * K]);
                        a00 = Fma.MultiplyAdd(r, b0, a00);
                        a01 = Fma.MultiplyAdd(r, b1, a01);
                        a02 = Fma.MultiplyAdd(r, b2, a02);
                        r = Vector256.Create(ak[1 * K]);
                        a10 = Fma.MultiplyAdd(r, b0, a10);
                        a11 = Fma.MultiplyAdd(r, b1, a11);
                        a12 = Fma.MultiplyAdd(r, b2, a12);
                        r = Vector256.Create(ak[2 * K]);
                        a20 = Fma.MultiplyAdd(r, b0, a20);
                        a21 = Fma.MultiplyAdd(r, b1, a21);
                        a22 = Fma.MultiplyAdd(r, b2, a22);
                        r = Vector256.Create(ak[3 * K]);
                        a30 = Fma.MultiplyAdd(r, b0, a30);
                        a31 = Fma.MultiplyAdd(r, b1, a31);
                        a32 = Fma.MultiplyAdd(r, b2, a32);
                    }

                    a00.Store(c);
                    a01.Store(c + 8);
                    a02.Store(c + 16);
                    a10.Store(c + 24);
                    a11.Store(c + 32);
                    a12.Store(c + 40);
                    a20.Store(c + 48);
                    a21.Store(c + 56);
                    a22.Store(c + 64);
                    a30.Store(c + 72);
                    a31.Store(c + 80);
                    a32.Store(c + 88);
                }
            }

            Sink = _c[0];
        }

        /// <summary>AVX-512, 8 rows × one 512-bit vector: the direct widening of today's shape.</summary>
        [Benchmark]
        public unsafe void Avx512_8x16()
        {
            if (!Avx512F.IsSupported)
            {
                return;
            }

            fixed (float* a = _a, b = _b, c = _c)
            {
                for (var s = 0; s < Sweeps; s++)
                {
                    Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default;
                    Vector512<float> a4 = default, a5 = default, a6 = default, a7 = default;

                    for (var k = 0; k < K; k++)
                    {
                        var bv = Vector512.Load(b + (k * 16));
                        var ak = a + k;

                        a0 = Avx512F.FusedMultiplyAdd(Vector512.Create(ak[0 * K]), bv, a0);
                        a1 = Avx512F.FusedMultiplyAdd(Vector512.Create(ak[1 * K]), bv, a1);
                        a2 = Avx512F.FusedMultiplyAdd(Vector512.Create(ak[2 * K]), bv, a2);
                        a3 = Avx512F.FusedMultiplyAdd(Vector512.Create(ak[3 * K]), bv, a3);
                        a4 = Avx512F.FusedMultiplyAdd(Vector512.Create(ak[4 * K]), bv, a4);
                        a5 = Avx512F.FusedMultiplyAdd(Vector512.Create(ak[5 * K]), bv, a5);
                        a6 = Avx512F.FusedMultiplyAdd(Vector512.Create(ak[6 * K]), bv, a6);
                        a7 = Avx512F.FusedMultiplyAdd(Vector512.Create(ak[7 * K]), bv, a7);
                    }

                    a0.Store(c);
                    a1.Store(c + 16);
                    a2.Store(c + 32);
                    a3.Store(c + 48);
                    a4.Store(c + 64);
                    a5.Store(c + 80);
                    a6.Store(c + 96);
                    a7.Store(c + 112);
                }
            }

            Sink = _c[0];
        }

        /// <summary>AVX-512, 8 rows × two 512-bit vectors: 16 accumulators, 10 loads per 16 FMAs.</summary>
        [Benchmark]
        public unsafe void Avx512_8x32()
        {
            if (!Avx512F.IsSupported)
            {
                return;
            }

            fixed (float* a = _a, b = _b, c = _c)
            {
                for (var s = 0; s < Sweeps; s++)
                {
                    Vector512<float> a00 = default, a01 = default, a10 = default, a11 = default;
                    Vector512<float> a20 = default, a21 = default, a30 = default, a31 = default;
                    Vector512<float> a40 = default, a41 = default, a50 = default, a51 = default;
                    Vector512<float> a60 = default, a61 = default, a70 = default, a71 = default;

                    for (var k = 0; k < K; k++)
                    {
                        var b0 = Vector512.Load(b + (k * 32));
                        var b1 = Vector512.Load(b + (k * 32) + 16);
                        var ak = a + k;

                        var r = Vector512.Create(ak[0 * K]);
                        a00 = Avx512F.FusedMultiplyAdd(r, b0, a00);
                        a01 = Avx512F.FusedMultiplyAdd(r, b1, a01);
                        r = Vector512.Create(ak[1 * K]);
                        a10 = Avx512F.FusedMultiplyAdd(r, b0, a10);
                        a11 = Avx512F.FusedMultiplyAdd(r, b1, a11);
                        r = Vector512.Create(ak[2 * K]);
                        a20 = Avx512F.FusedMultiplyAdd(r, b0, a20);
                        a21 = Avx512F.FusedMultiplyAdd(r, b1, a21);
                        r = Vector512.Create(ak[3 * K]);
                        a30 = Avx512F.FusedMultiplyAdd(r, b0, a30);
                        a31 = Avx512F.FusedMultiplyAdd(r, b1, a31);
                        r = Vector512.Create(ak[4 * K]);
                        a40 = Avx512F.FusedMultiplyAdd(r, b0, a40);
                        a41 = Avx512F.FusedMultiplyAdd(r, b1, a41);
                        r = Vector512.Create(ak[5 * K]);
                        a50 = Avx512F.FusedMultiplyAdd(r, b0, a50);
                        a51 = Avx512F.FusedMultiplyAdd(r, b1, a51);
                        r = Vector512.Create(ak[6 * K]);
                        a60 = Avx512F.FusedMultiplyAdd(r, b0, a60);
                        a61 = Avx512F.FusedMultiplyAdd(r, b1, a61);
                        r = Vector512.Create(ak[7 * K]);
                        a70 = Avx512F.FusedMultiplyAdd(r, b0, a70);
                        a71 = Avx512F.FusedMultiplyAdd(r, b1, a71);
                    }

                    a00.Store(c);
                    a01.Store(c + 16);
                    a10.Store(c + 32);
                    a11.Store(c + 48);
                    a20.Store(c + 64);
                    a21.Store(c + 80);
                    a30.Store(c + 96);
                    a31.Store(c + 112);
                    a40.Store(c + 128);
                    a41.Store(c + 144);
                    a50.Store(c + 160);
                    a51.Store(c + 176);
                    a60.Store(c + 192);
                    a61.Store(c + 208);
                    a70.Store(c + 224);
                    a71.Store(c + 240);
                }
            }

            Sink = _c[0];
        }

        /// <summary>
        /// AVX-512, <b>12 rows x two 512-bit vectors: 24 accumulators</b> — the shape both MLAS and BLIS
        /// converge on, and eight more than this project's current kernel holds.
        ///
        /// <para><b>Read from the two libraries rather than reasoned about.</b> MLAS's
        /// <c>FgemmKernelAvx512FCommon.inc</c> declares <c>zmm4-zmm27</c> as its block accumulators — 12
        /// rows by two vectors — and BLIS's SKX configuration uses <c>MR=32, NR=12</c>, the same 24 with
        /// the roles swapped. Our 8x32 holds 16, leaving 13 of the 32 zmm registers unused.</para>
        ///
        /// <para><b>The arithmetic that makes it worth pricing.</b> Per k-step this shape issues 2 B loads
        /// (128 B) plus 12 A broadcasts (48 B) for 24 FMAs: <b>4.36 FLOP per byte fetched, against 3.20
        /// for 8x32</b> — 36% more arithmetic per operand. That is the direction every measurement in
        /// `XC-78` points: the kernel reaches 95% of single-core peak with L1-resident operands and a
        /// fraction of that in production, so what is short is operand delivery, not FMA throughput.</para>
        /// </summary>
        [Benchmark]
        public unsafe void Avx512_12x32()
        {
            if (!Avx512F.IsSupported)
            {
                return;
            }

            fixed (float* a = _a, b = _b, c = _c)
            {
                for (var s = 0; s < Sweeps; s++)
                {
                    Vector512<float> a00 = default, a01 = default;
                    Vector512<float> a10 = default, a11 = default;
                    Vector512<float> a20 = default, a21 = default;
                    Vector512<float> a30 = default, a31 = default;
                    Vector512<float> a40 = default, a41 = default;
                    Vector512<float> a50 = default, a51 = default;
                    Vector512<float> a60 = default, a61 = default;
                    Vector512<float> a70 = default, a71 = default;
                    Vector512<float> a80 = default, a81 = default;
                    Vector512<float> a90 = default, a91 = default;
                    Vector512<float> a100 = default, a101 = default;
                    Vector512<float> a110 = default, a111 = default;

                    for (var k = 0; k < K; k++)
                    {
                        var b0 = Vector512.Load(b + (k * 32));
                        var b1 = Vector512.Load(b + (k * 32) + 16);
                        var ak = a + k;

                        var r0 = Vector512.Create(ak[0 * K]);
                        a00 = Avx512F.FusedMultiplyAdd(r0, b0, a00);
                        a01 = Avx512F.FusedMultiplyAdd(r0, b1, a01);
                        var r1 = Vector512.Create(ak[1 * K]);
                        a10 = Avx512F.FusedMultiplyAdd(r1, b0, a10);
                        a11 = Avx512F.FusedMultiplyAdd(r1, b1, a11);
                        var r2 = Vector512.Create(ak[2 * K]);
                        a20 = Avx512F.FusedMultiplyAdd(r2, b0, a20);
                        a21 = Avx512F.FusedMultiplyAdd(r2, b1, a21);
                        var r3 = Vector512.Create(ak[3 * K]);
                        a30 = Avx512F.FusedMultiplyAdd(r3, b0, a30);
                        a31 = Avx512F.FusedMultiplyAdd(r3, b1, a31);
                        var r4 = Vector512.Create(ak[4 * K]);
                        a40 = Avx512F.FusedMultiplyAdd(r4, b0, a40);
                        a41 = Avx512F.FusedMultiplyAdd(r4, b1, a41);
                        var r5 = Vector512.Create(ak[5 * K]);
                        a50 = Avx512F.FusedMultiplyAdd(r5, b0, a50);
                        a51 = Avx512F.FusedMultiplyAdd(r5, b1, a51);
                        var r6 = Vector512.Create(ak[6 * K]);
                        a60 = Avx512F.FusedMultiplyAdd(r6, b0, a60);
                        a61 = Avx512F.FusedMultiplyAdd(r6, b1, a61);
                        var r7 = Vector512.Create(ak[7 * K]);
                        a70 = Avx512F.FusedMultiplyAdd(r7, b0, a70);
                        a71 = Avx512F.FusedMultiplyAdd(r7, b1, a71);
                        var r8 = Vector512.Create(ak[8 * K]);
                        a80 = Avx512F.FusedMultiplyAdd(r8, b0, a80);
                        a81 = Avx512F.FusedMultiplyAdd(r8, b1, a81);
                        var r9 = Vector512.Create(ak[9 * K]);
                        a90 = Avx512F.FusedMultiplyAdd(r9, b0, a90);
                        a91 = Avx512F.FusedMultiplyAdd(r9, b1, a91);
                        var r10 = Vector512.Create(ak[10 * K]);
                        a100 = Avx512F.FusedMultiplyAdd(r10, b0, a100);
                        a101 = Avx512F.FusedMultiplyAdd(r10, b1, a101);
                        var r11 = Vector512.Create(ak[11 * K]);
                        a110 = Avx512F.FusedMultiplyAdd(r11, b0, a110);
                        a111 = Avx512F.FusedMultiplyAdd(r11, b1, a111);
                    }

                    a00.Store(c + 0);
                    a01.Store(c + 16);
                    a10.Store(c + 32);
                    a11.Store(c + 48);
                    a20.Store(c + 64);
                    a21.Store(c + 80);
                    a30.Store(c + 96);
                    a31.Store(c + 112);
                    a40.Store(c + 128);
                    a41.Store(c + 144);
                    a50.Store(c + 160);
                    a51.Store(c + 176);
                    a60.Store(c + 192);
                    a61.Store(c + 208);
                    a70.Store(c + 224);
                    a71.Store(c + 240);
                    a80.Store(c + 256);
                    a81.Store(c + 272);
                    a90.Store(c + 288);
                    a91.Store(c + 304);
                    a100.Store(c + 320);
                    a101.Store(c + 336);
                    a110.Store(c + 352);
                    a111.Store(c + 368);
                }
            }

            Sink = _c[0];
        }

        /// <summary>
        /// The 8x32 shape again, with the sixteen accumulators held in a <c>stackalloc</c> span instead of
        /// named locals. <b>Everything else is identical</b> — same loads, same broadcasts, same FMAs, same
        /// order, same panels.
        ///
        /// <para><b>What this prices, and why it is not about this benchmark.</b> This file's own remarks
        /// already state the rule — <i>accumulators are named locals, never a <c>stackalloc</c> span, because
        /// a span forces an L1 round-trip per accumulator per iteration</i> — and record that breaking it once
        /// cost a 2.8x error in an earlier roofline. <b>`Q4KGemvKernel.GemmTiled512` breaks it</b>: five
        /// <c>stackalloc</c> spans of <c>Vector512</c> serve as its accumulators. That kernel is the
        /// quantised prefill path, which is compute-bound rather than bandwidth-bound, so it is where the
        /// pattern would hurt.</para>
        ///
        /// <para><b>This arm answers the cheap question first.</b> Reimplementing Q4_K's accumulation to
        /// compare layouts is a day's work; pricing the layout itself against an already-measured shape is
        /// one arm. <b>If the span costs nothing here the JIT promotes it and the hypothesis dies; if it
        /// costs what the register spill cost the conv kernel, the prefill path is worth the day.</b></para>
        /// </summary>
        [Benchmark]
        public unsafe void Avx512_8x32_SpanAccumulators()
        {
            if (!Avx512F.IsSupported)
            {
                return;
            }

            Span<Vector512<float>> acc = stackalloc Vector512<float>[16];

            fixed (float* a = _a, b = _b, c = _c)
            {
                for (var s = 0; s < Sweeps; s++)
                {
                    acc.Clear();

                    for (var k = 0; k < K; k++)
                    {
                        var b0 = Vector512.Load(b + (k * 32));
                        var b1 = Vector512.Load(b + (k * 32) + 16);
                        var ak = a + k;

                        var r0 = Vector512.Create(ak[0 * K]);
                        acc[0] = Avx512F.FusedMultiplyAdd(r0, b0, acc[0]);
                        acc[1] = Avx512F.FusedMultiplyAdd(r0, b1, acc[1]);
                        var r1 = Vector512.Create(ak[1 * K]);
                        acc[2] = Avx512F.FusedMultiplyAdd(r1, b0, acc[2]);
                        acc[3] = Avx512F.FusedMultiplyAdd(r1, b1, acc[3]);
                        var r2 = Vector512.Create(ak[2 * K]);
                        acc[4] = Avx512F.FusedMultiplyAdd(r2, b0, acc[4]);
                        acc[5] = Avx512F.FusedMultiplyAdd(r2, b1, acc[5]);
                        var r3 = Vector512.Create(ak[3 * K]);
                        acc[6] = Avx512F.FusedMultiplyAdd(r3, b0, acc[6]);
                        acc[7] = Avx512F.FusedMultiplyAdd(r3, b1, acc[7]);
                        var r4 = Vector512.Create(ak[4 * K]);
                        acc[8] = Avx512F.FusedMultiplyAdd(r4, b0, acc[8]);
                        acc[9] = Avx512F.FusedMultiplyAdd(r4, b1, acc[9]);
                        var r5 = Vector512.Create(ak[5 * K]);
                        acc[10] = Avx512F.FusedMultiplyAdd(r5, b0, acc[10]);
                        acc[11] = Avx512F.FusedMultiplyAdd(r5, b1, acc[11]);
                        var r6 = Vector512.Create(ak[6 * K]);
                        acc[12] = Avx512F.FusedMultiplyAdd(r6, b0, acc[12]);
                        acc[13] = Avx512F.FusedMultiplyAdd(r6, b1, acc[13]);
                        var r7 = Vector512.Create(ak[7 * K]);
                        acc[14] = Avx512F.FusedMultiplyAdd(r7, b0, acc[14]);
                        acc[15] = Avx512F.FusedMultiplyAdd(r7, b1, acc[15]);
                    }

                    for (var i = 0; i < 16; i++)
                    {
                        acc[i].Store(c + (i * 16));
                    }
                }
            }

            Sink = _c[0];
        }

        /// <summary>AVX-512, 6 rows × three 512-bit vectors: 18 accumulators, 9 loads per 18 FMAs — the best ratio tried.</summary>
        [Benchmark]
        public unsafe void Avx512_6x48()
        {
            if (!Avx512F.IsSupported)
            {
                return;
            }

            fixed (float* a = _a, b = _b, c = _c)
            {
                for (var s = 0; s < Sweeps; s++)
                {
                    Vector512<float> a00 = default, a01 = default, a02 = default;
                    Vector512<float> a10 = default, a11 = default, a12 = default;
                    Vector512<float> a20 = default, a21 = default, a22 = default;
                    Vector512<float> a30 = default, a31 = default, a32 = default;
                    Vector512<float> a40 = default, a41 = default, a42 = default;
                    Vector512<float> a50 = default, a51 = default, a52 = default;

                    for (var k = 0; k < K; k++)
                    {
                        var b0 = Vector512.Load(b + (k * 48));
                        var b1 = Vector512.Load(b + (k * 48) + 16);
                        var b2 = Vector512.Load(b + (k * 48) + 32);
                        var ak = a + k;

                        var r = Vector512.Create(ak[0 * K]);
                        a00 = Avx512F.FusedMultiplyAdd(r, b0, a00);
                        a01 = Avx512F.FusedMultiplyAdd(r, b1, a01);
                        a02 = Avx512F.FusedMultiplyAdd(r, b2, a02);
                        r = Vector512.Create(ak[1 * K]);
                        a10 = Avx512F.FusedMultiplyAdd(r, b0, a10);
                        a11 = Avx512F.FusedMultiplyAdd(r, b1, a11);
                        a12 = Avx512F.FusedMultiplyAdd(r, b2, a12);
                        r = Vector512.Create(ak[2 * K]);
                        a20 = Avx512F.FusedMultiplyAdd(r, b0, a20);
                        a21 = Avx512F.FusedMultiplyAdd(r, b1, a21);
                        a22 = Avx512F.FusedMultiplyAdd(r, b2, a22);
                        r = Vector512.Create(ak[3 * K]);
                        a30 = Avx512F.FusedMultiplyAdd(r, b0, a30);
                        a31 = Avx512F.FusedMultiplyAdd(r, b1, a31);
                        a32 = Avx512F.FusedMultiplyAdd(r, b2, a32);
                        r = Vector512.Create(ak[4 * K]);
                        a40 = Avx512F.FusedMultiplyAdd(r, b0, a40);
                        a41 = Avx512F.FusedMultiplyAdd(r, b1, a41);
                        a42 = Avx512F.FusedMultiplyAdd(r, b2, a42);
                        r = Vector512.Create(ak[5 * K]);
                        a50 = Avx512F.FusedMultiplyAdd(r, b0, a50);
                        a51 = Avx512F.FusedMultiplyAdd(r, b1, a51);
                        a52 = Avx512F.FusedMultiplyAdd(r, b2, a52);
                    }

                    a00.Store(c);
                    a01.Store(c + 16);
                    a02.Store(c + 32);
                    a10.Store(c + 48);
                    a11.Store(c + 64);
                    a12.Store(c + 80);
                    a20.Store(c + 96);
                    a21.Store(c + 112);
                    a22.Store(c + 128);
                    a30.Store(c + 144);
                    a31.Store(c + 160);
                    a32.Store(c + 176);
                    a40.Store(c + 192);
                    a41.Store(c + 208);
                    a42.Store(c + 224);
                    a50.Store(c + 240);
                    a51.Store(c + 256);
                    a52.Store(c + 272);
                }
            }

            Sink = _c[0];
        }
    }
}
