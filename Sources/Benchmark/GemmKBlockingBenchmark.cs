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
    /// Measures whether making the micro-kernel's working set <b>L1-resident</b> recovers the 4× it loses
    /// between isolation and production — on one real VGG layer shape, single-threaded, before any of this is
    /// written into <c>Conv2DGemmKernels</c>.
    ///
    /// <para><b>The finding this tests.</b> The 8×8 micro-kernel measures 148 GFLOP/s per core in isolation
    /// (its AVX2 hardware peak) but ~38 GFLOP/s per core inside the conv GEMM — the same instructions, 4×
    /// slower. The suspected cause is the working set: the packed B panel is <c>K × 8</c> floats, which at
    /// K=4608 is <b>147 KB against a 32–48 KB L1</b>, and the A row-block is another 147 KB. Splitting the
    /// contraction into <c>Kc</c>-sized blocks shrinks both to <c>Kc × 8 × 4</c> bytes.</para>
    ///
    /// <para><b>Why this is a prototype, not a micro-benchmark.</b> It performs the entire layer GEMM at each
    /// blocking factor, so the arms differ only in <c>Kc</c>: total FLOPs, total A traffic and total B traffic
    /// are identical by construction (each element is still touched once per panel sweep). What changes is
    /// solely what fits in L1 — and the extra C read-modify-write that blocking forces, which is part of the
    /// cost being measured and is deliberately not excluded.</para>
    ///
    /// <para><b>Shape:</b> VGG-16 <c>conv5_1</c> — M=512 output channels, K=4608 (512·3·3), N=196 (14×14).
    /// Chosen because K is at its largest here, the extreme case for L1 overflow, while the small N keeps a
    /// single-threaded run short. <c>Kc = K</c> reproduces today's unblocked kernel exactly.</para>
    ///
    /// <para>Single-threaded on purpose: parallel scaling was measured separately at 21.6×, so mixing it in
    /// would only add noise to a cache question.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*GemmKBlocking*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class GemmKBlockingBenchmark
    {
        /// <summary>VGG-16 conv5_1: output channels.</summary>
        public const int M = 512;

        /// <summary>VGG-16 conv5_1: inChannels·3·3 — the contraction length.</summary>
        public const int K = 4608;

        /// <summary>VGG-16 conv5_1: outH·outW = 14·14.</summary>
        public const int N = 196;

        private const int Mr = 8;
        private const int Nr = 8;

        /// <summary>
        /// Contraction block. <c>4608</c> is today's behaviour (no blocking, 147 KB panel); the smaller values
        /// bring the packed panel to 16 KB / 8 KB / 4 KB, i.e. inside L1 with room for the A block beside it.
        /// </summary>
        [Params(4608, 1152, 512, 256, 128)]
        public int Kc
        {
            get; set;
        }

        private float[] _a = null!;
        private float[] _b = null!;
        private float[] _c = null!;
        private float[] _packB = null!;

        public float Sink;

        public static WorkAmount GetWorkAmount(BenchmarkCase benchmarkCase)
        {
            return WorkAmount.Matmul(M, K, N);
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260723);

            _a = new float[(long)M * K];
            _b = new float[(long)K * N];
            _c = new float[(long)M * N];
            _packB = new float[(long)K * Nr];

            for (var i = 0; i < _a.Length; i++)
            {
                _a[i] = (float)((rng.NextDouble() * 2.0) - 1.0);
            }

            for (var i = 0; i < _b.Length; i++)
            {
                _b[i] = (float)((rng.NextDouble() * 2.0) - 1.0);
            }
        }

        /// <summary>
        /// The layer GEMM with the contraction split into <see cref="Kc"/>-sized blocks. At <c>Kc == K</c> this
        /// is exactly the production loop: pack the whole panel, then sweep M once.
        /// </summary>
        [Benchmark]
        public unsafe void BlockedGemm()
        {
            var panels = (N + Nr - 1) / Nr;

            fixed (float* a = _a, b = _b, c = _c, packB = _packB)
            {
                new Span<float>(c, M * N).Clear();

                for (var k0 = 0; k0 < K; k0 += Kc)
                {
                    var kcEff = Math.Min(Kc, K - k0);

                    for (var np = 0; np < panels; np++)
                    {
                        var n0 = np * Nr;
                        var nrEff = Math.Min(Nr, N - n0);

                        for (var kk = 0; kk < kcEff; kk++)
                        {
                            var src = ((k0 + kk) * N) + n0;
                            var dst = kk * Nr;
                            for (var j = 0; j < Nr; j++)
                            {
                                packB[dst + j] = j < nrEff ? b[src + j] : 0f;
                            }
                        }

                        for (var m0 = 0; m0 + Mr <= M; m0 += Mr)
                        {
                            Accumulate8x8(a, m0, k0, kcEff, packB, c, n0, nrEff);
                        }
                    }
                }
            }

            Sink = _c[0];
        }

        /// <summary>
        /// One 8×8 tile, accumulated over a <paramref name="kcEff"/>-long slice of the contraction.
        ///
        /// <para>Unlike the production kernel this loads C in and stores it back, because a K-block only holds
        /// a partial sum. That extra read-modify-write per (K-block, panel, row-block) is the price of blocking
        /// and is measured here rather than assumed away — at <c>Kc = K</c> it happens once and costs nothing.</para>
        /// </summary>
        private static unsafe void Accumulate8x8(
            float* a, int m0, int k0, int kcEff, float* packB, float* c, int n0, int nrEff)
        {
            var a0 = a + ((long)(m0 + 0) * K) + k0;
            var a1 = a + ((long)(m0 + 1) * K) + k0;
            var a2 = a + ((long)(m0 + 2) * K) + k0;
            var a3 = a + ((long)(m0 + 3) * K) + k0;
            var a4 = a + ((long)(m0 + 4) * K) + k0;
            var a5 = a + ((long)(m0 + 5) * K) + k0;
            var a6 = a + ((long)(m0 + 6) * K) + k0;
            var a7 = a + ((long)(m0 + 7) * K) + k0;

            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;
            var acc4 = Vector256<float>.Zero;
            var acc5 = Vector256<float>.Zero;
            var acc6 = Vector256<float>.Zero;
            var acc7 = Vector256<float>.Zero;

            for (var kk = 0; kk < kcEff; kk++)
            {
                var bv = Vector256.Load(packB + (kk * Nr));

                acc0 = Fma.MultiplyAdd(Vector256.Create(a0[kk]), bv, acc0);
                acc1 = Fma.MultiplyAdd(Vector256.Create(a1[kk]), bv, acc1);
                acc2 = Fma.MultiplyAdd(Vector256.Create(a2[kk]), bv, acc2);
                acc3 = Fma.MultiplyAdd(Vector256.Create(a3[kk]), bv, acc3);
                acc4 = Fma.MultiplyAdd(Vector256.Create(a4[kk]), bv, acc4);
                acc5 = Fma.MultiplyAdd(Vector256.Create(a5[kk]), bv, acc5);
                acc6 = Fma.MultiplyAdd(Vector256.Create(a6[kk]), bv, acc6);
                acc7 = Fma.MultiplyAdd(Vector256.Create(a7[kk]), bv, acc7);
            }

            StoreRow(c, (m0 + 0) * N + n0, acc0, nrEff);
            StoreRow(c, (m0 + 1) * N + n0, acc1, nrEff);
            StoreRow(c, (m0 + 2) * N + n0, acc2, nrEff);
            StoreRow(c, (m0 + 3) * N + n0, acc3, nrEff);
            StoreRow(c, (m0 + 4) * N + n0, acc4, nrEff);
            StoreRow(c, (m0 + 5) * N + n0, acc5, nrEff);
            StoreRow(c, (m0 + 6) * N + n0, acc6, nrEff);
            StoreRow(c, (m0 + 7) * N + n0, acc7, nrEff);
        }

        private static unsafe void StoreRow(float* c, int offset, Vector256<float> acc, int nrEff)
        {
            var dst = c + offset;

            for (var j = 0; j < nrEff; j++)
            {
                dst[j] += acc.GetElement(j);
            }
        }
    }
}
