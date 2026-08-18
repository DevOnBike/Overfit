// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    /// <summary>
    /// Walks the conv GEMM from its isolated micro-kernel up to production, <b>adding one surrounding cost
    /// per arm</b>, so the gap between the two can be attributed instead of guessed at.
    ///
    /// <para><b>Why this and not another restructure (`XC-78`).</b> The isolated micro-kernel measured
    /// 132-148 GFLOP/s per core while production convolution ran far below that, and <b>four structural
    /// explanations have now been tried</b>: panel grouping (measured neutral), BLIS-style K-blocking with
    /// A-packing (measured a regression), K-blocking alone (measured inside the instrument's own error), and
    /// MR-major kernel packing (measured a win, and shipped). Guessing a fifth is the expensive way to find
    /// out. Every arm below runs the same micro-kernel over the same 3.70 GFLOP, so the ratio between two
    /// adjacent arms is the price of exactly one thing.</para>
    ///
    /// <para><b>The shape is VGG-16's conv10</b>: <c>M = 512</c> output channels, <c>N = 784</c> output
    /// positions, <c>K = 4608</c> contraction. That is 64 row blocks of <c>Mr = 8</c> against 25 panels of
    /// <c>Nr = 32</c>, so a full layer is 1,600 micro-kernel calls — which is exactly what each arm issues.</para>
    ///
    /// <list type="table">
    ///   <item><term>L0 ceiling</term><description>k = 128 per call, so both operand panels are L1-resident. The issue-rate ceiling.</description></item>
    ///   <item><term>L1 full K</term><description>k = 4608 in one pass. Adds: the B panel is 589 KB, so operands come from L2.</description></item>
    ///   <item><term>L2 real A</term><description>A cycles through all 64 row blocks of a real 9.4 MB packed matrix. Adds: A's working set.</description></item>
    ///   <item><term>L3 real B</term><description>B cycles through all 25 real panels, 14.7 MB. Adds: B's working set.</description></item>
    ///   <item><term>L4 real C</term><description>Results land in a real 1.6 MB output matrix at their true offsets. Adds: C traffic.</description></item>
    /// </list>
    ///
    /// <para><b>What this does NOT include</b>, deliberately: the patch gather, the parallel dispatch, and
    /// the per-panel setup. Those are the difference between L4 and a production single-threaded run, and
    /// leaving them out is what makes L4 a clean stopping point rather than a second production path.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*ConvGemmCostLadder*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public unsafe class ConvGemmCostLadderBenchmark : IDisposable
    {
        private const int M = 512;
        private const int N = 784;
        private const int K = 4608;

        private const int Mr = 8;
        private const int Nr = 32;

        private const int RowBlocks = M / Mr;      // 64
        private const int Panels = (N + Nr - 1) / Nr;  // 25

        /// <summary>k per call in the L1-resident arm: 128 * 32 * 4 = 16 KB of B, 4 KB of A.</summary>
        private const int SmallK = 128;

        private TensorStorage<float> _packedA = null!;   // [RowBlocks][K][Mr] — 9.4 MB
        private TensorStorage<float> _packedB = null!;   // [Panels][K][Nr]    — 14.7 MB
        private TensorStorage<float> _outputReal = null!; // [M][N]            — 1.6 MB
        private TensorStorage<float> _outputTile = null!; // one 8x32 tile     — 1 KB

        // VGG-16's conv10 in its real form: 512 in, 512 out, 28x28, 3x3 pad 1 — which is K = 4608 and
        // N = 784, the same arithmetic the rungs above issue by hand.
        private const int InChannels = 512;
        private const int InputHw = 28;
        private const int KernelSize = 3;

        private TensorStorage<float> _convInput = null!;
        private TensorStorage<float> _convKernels = null!;
        private TensorStorage<float> _convPackedKernels = null!;

        [GlobalSetup]
        public void Setup()
        {
            _packedA = new TensorStorage<float>(RowBlocks * K * Mr, clearMemory: false);
            _packedB = new TensorStorage<float>(Panels * K * Nr, clearMemory: false);
            _outputReal = new TensorStorage<float>(M * N, clearMemory: true);
            _outputTile = new TensorStorage<float>(Mr * Nr, clearMemory: true);

            Fill(_packedA.AsSpan(), seed: 3);
            Fill(_packedB.AsSpan(), seed: 11);

            _convInput = new TensorStorage<float>(InChannels * InputHw * InputHw, clearMemory: false);
            _convKernels = new TensorStorage<float>(M * K, clearMemory: false);
            _convPackedKernels = new TensorStorage<float>(
                Conv2DGemmKernels.PackedKernelLength(M, K), clearMemory: false);

            Fill(_convInput.AsSpan(), seed: 23);
            Fill(_convKernels.AsSpan(), seed: 29);

            Conv2DGemmKernels.PackKernels(
                _convKernels.AsReadOnlySpan(), _convPackedKernels.AsSpan(), M, K);
        }

        /// <summary>
        /// L0 — the same 1,600 calls with k = 256, so the B panel is 32 KB and the A slice 8 KB: both fit
        /// this core's 48 KB L1.
        ///
        /// <para><b>This arm settles a contradiction between two of this project's own measurements.</b>
        /// `GemmMicroKernelShapeBenchmark` reports the 8x32 tile at <b>340 GFLOP/s</b> with L1-resident
        /// panels — 95% of this core's 359 GFLOP/s peak — while every rung of this ladder sits near
        /// <b>95 GFLOP/s</b>. The rungs' smallest working set is 736 KB, which is L2, so the cliff would have
        /// to be between L1 and L2. But the K-blocking experiment put the B panel at 32 KB and moved nothing.
        /// <b>One of those three results is wrong and a cross-benchmark comparison cannot say which</b>, so
        /// this arm makes the comparison inside one structure: identical call count, identical C handling,
        /// only the panel size changes. FLOP counts differ between arms, so read GFLOP/s and not time.</para>
        /// </summary>
        [Benchmark]
        public void L0_L1Resident()
        {
            const int SmallK = 256;

            fixed (float* a = _packedA.AsSpan(), b = _packedB.AsSpan(), c = _outputTile.AsSpan())
            {
                for (var i = 0; i < RowBlocks * Panels; i++)
                {
                    Conv2DGemmKernels.MicroKernel8x32Avx512PackedA(a, Mr, SmallK, b, c, Nr, 0, 0, Nr);
                }
            }
        }

        /// <summary>
        /// L1b — A cycles through only FOUR row blocks, 590 KB, which still fits L2.
        ///
        /// <para><b>This arm exists because the first version of the ladder produced a backwards result</b>,
        /// and this project's rule is to suspect the benchmark before the silicon. L2 below, with a 9.4 MB
        /// A working set, measured <i>faster</i> than L1 with a 736 KB one. If that is real, the kernel is
        /// limited by operand LATENCY rather than by capacity: hammering one resident block gives the
        /// hardware prefetcher nothing to follow, while walking blocks in order lets it deliver A into L1
        /// ahead of use. Four blocks stream just as predictably as sixty-four but fit where one block fits,
        /// so <b>if L1b matches L2 the cause is streaming, and if it matches L1 the cause is size</b>.</para>
        /// </summary>
        [Benchmark]
        public void L1b_StreamingSmallA()
        {
            const int Blocks = 4;

            fixed (float* a = _packedA.AsSpan(), b = _packedB.AsSpan(), c = _outputTile.AsSpan())
            {
                for (var i = 0; i < RowBlocks * Panels; i++)
                {
                    Conv2DGemmKernels.MicroKernel8x32Avx512PackedA(
                        a + ((long)(i % Blocks) * K * Mr), Mr, K, b, c, Nr, 0, 0, Nr);
                }
            }
        }

        /// <summary>
        /// L1i — byte-for-byte the same work as <see cref="L1_FullK"/>, with the FMA sequence written
        /// inline instead of called.
        ///
        /// <para><b>The one lever.</b> Same buffers, same addresses, same 1,600 iterations, same k = 4608,
        /// same MR-major A layout, same contiguous B. The only difference is whether the accumulator loop
        /// lives in its own method or in this one.</para>
        ///
        /// <para><b>Why it is worth an arm.</b> `GemmMicroKernelShapeBenchmark` measures this shape at
        /// <b>340 GFLOP/s</b> with the sequence inlined, while every rung of this ladder that CALLS the
        /// production kernel sits near <b>97</b>. That was first read as an L1-versus-L2 residency cliff -
        /// wrongly, because the shape benchmark's own buffers total 552 KB and are therefore not
        /// L1-resident either. <b>Residency is not what separates the two benchmarks; the call is.</b></para>
        ///
        /// <para><b>The hypothesis this tests.</b> Sixteen <c>Vector512</c> accumulators need sixteen zmm
        /// registers. Inside a real method, with parameters, a <c>stackalloc</c> and eight store calls also
        /// live, the JIT may not keep them there - and spilling sixteen registers per k-step would cost
        /// about what is missing. <b>If this arm matches L1 the hypothesis is dead; if it reaches 340 the
        /// register allocator is the whole remaining gap.</b></para>
        /// </summary>
        [Benchmark]
        public void L1i_FullKInlined()
        {
            fixed (float* a = _packedA.AsSpan(), b = _packedB.AsSpan(), c = _outputTile.AsSpan())
            {
                for (var i = 0; i < RowBlocks * Panels; i++)
                {
                    Vector512<float> c00 = default, c01 = default;
                    Vector512<float> c10 = default, c11 = default;
                    Vector512<float> c20 = default, c21 = default;
                    Vector512<float> c30 = default, c31 = default;
                    Vector512<float> c40 = default, c41 = default;
                    Vector512<float> c50 = default, c51 = default;
                    Vector512<float> c60 = default, c61 = default;
                    Vector512<float> c70 = default, c71 = default;

                    for (var kk = 0; kk < K; kk++)
                    {
                        var b0 = Vector512.Load(b + (kk * Nr));
                        var b1 = Vector512.Load(b + (kk * Nr) + 16);
                        var aSlot = a + ((long)kk * Mr);

                        var r0 = Vector512.Create(aSlot[0]);
                        c00 = Avx512F.FusedMultiplyAdd(r0, b0, c00);
                        c01 = Avx512F.FusedMultiplyAdd(r0, b1, c01);
                        var r1 = Vector512.Create(aSlot[1]);
                        c10 = Avx512F.FusedMultiplyAdd(r1, b0, c10);
                        c11 = Avx512F.FusedMultiplyAdd(r1, b1, c11);
                        var r2 = Vector512.Create(aSlot[2]);
                        c20 = Avx512F.FusedMultiplyAdd(r2, b0, c20);
                        c21 = Avx512F.FusedMultiplyAdd(r2, b1, c21);
                        var r3 = Vector512.Create(aSlot[3]);
                        c30 = Avx512F.FusedMultiplyAdd(r3, b0, c30);
                        c31 = Avx512F.FusedMultiplyAdd(r3, b1, c31);
                        var r4 = Vector512.Create(aSlot[4]);
                        c40 = Avx512F.FusedMultiplyAdd(r4, b0, c40);
                        c41 = Avx512F.FusedMultiplyAdd(r4, b1, c41);
                        var r5 = Vector512.Create(aSlot[5]);
                        c50 = Avx512F.FusedMultiplyAdd(r5, b0, c50);
                        c51 = Avx512F.FusedMultiplyAdd(r5, b1, c51);
                        var r6 = Vector512.Create(aSlot[6]);
                        c60 = Avx512F.FusedMultiplyAdd(r6, b0, c60);
                        c61 = Avx512F.FusedMultiplyAdd(r6, b1, c61);
                        var r7 = Vector512.Create(aSlot[7]);
                        c70 = Avx512F.FusedMultiplyAdd(r7, b0, c70);
                        c71 = Avx512F.FusedMultiplyAdd(r7, b1, c71);
                    }

                    c00.Store(c + 0);
                    c01.Store(c + 16);
                    c10.Store(c + 32);
                    c11.Store(c + 48);
                    c20.Store(c + 64);
                    c21.Store(c + 80);
                    c30.Store(c + 96);
                    c31.Store(c + 112);
                    c40.Store(c + 128);
                    c41.Store(c + 144);
                    c50.Store(c + 160);
                    c51.Store(c + 176);
                    c60.Store(c + 192);
                    c61.Store(c + 208);
                    c70.Store(c + 224);
                    c71.Store(c + 240);
                }
            }
        }

        /// <summary>L1 — one pass over the whole of K. Adds: a 589 KB B panel, so operands come from L2.</summary>
        [Benchmark(Baseline = true)]
        public void L1_FullK()
        {
            fixed (float* a = _packedA.AsSpan(), b = _packedB.AsSpan(), c = _outputTile.AsSpan())
            {
                for (var i = 0; i < RowBlocks * Panels; i++)
                {
                    Conv2DGemmKernels.MicroKernel8x32Avx512PackedA(a, Mr, K, b, c, Nr, 0, 0, Nr);
                }
            }
        }

        /// <summary>L2 — A cycles through all 64 real row blocks. Adds: A's 9.4 MB working set.</summary>
        [Benchmark]
        public void L2_RealA()
        {
            fixed (float* a = _packedA.AsSpan(), b = _packedB.AsSpan(), c = _outputTile.AsSpan())
            {
                for (var panel = 0; panel < Panels; panel++)
                {
                    for (var rb = 0; rb < RowBlocks; rb++)
                    {
                        Conv2DGemmKernels.MicroKernel8x32Avx512PackedA(
                            a + ((long)rb * K * Mr), Mr, K, b, c, Nr, 0, 0, Nr);
                    }
                }
            }
        }

        /// <summary>L3 — B cycles through all 25 real panels too. Adds: B's 14.7 MB working set.</summary>
        [Benchmark]
        public void L3_RealB()
        {
            fixed (float* a = _packedA.AsSpan(), b = _packedB.AsSpan(), c = _outputTile.AsSpan())
            {
                for (var panel = 0; panel < Panels; panel++)
                {
                    var panelB = b + ((long)panel * K * Nr);

                    for (var rb = 0; rb < RowBlocks; rb++)
                    {
                        Conv2DGemmKernels.MicroKernel8x32Avx512PackedA(
                            a + ((long)rb * K * Mr), Mr, K, panelB, c, Nr, 0, 0, Nr);
                    }
                }
            }
        }

        /// <summary>L4 — results land in the real output matrix at their true offsets. Adds: C traffic.</summary>
        [Benchmark]
        public void L4_RealC()
        {
            fixed (float* a = _packedA.AsSpan(), b = _packedB.AsSpan(), c = _outputReal.AsSpan())
            {
                for (var panel = 0; panel < Panels; panel++)
                {
                    var panelB = b + ((long)panel * K * Nr);
                    var n0 = panel * Nr;
                    var nrEff = Math.Min(Nr, N - n0);

                    for (var rb = 0; rb < RowBlocks; rb++)
                    {
                        Conv2DGemmKernels.MicroKernel8x32Avx512PackedA(
                            a + ((long)rb * K * Mr), Mr, K, panelB, c, N, n0, rb * Mr, nrEff);
                    }
                }
            }
        }

        /// <summary>
        /// L5 — the production convolution at this shape: patch gather, panel setup, dispatch and all.
        /// The gap between L4 and here is everything the ladder above deliberately leaves out.
        /// </summary>
        [Benchmark]
        public void L5_Production()
        {
            Conv2DKernels.ForwardNchw(
                _convInput.AsReadOnlySpan(),
                _convKernels.AsReadOnlySpan(),
                _outputReal.AsSpan(),
                batchSize: 1,
                inChannels: InChannels,
                outChannels: M,
                inputH: InputHw,
                inputW: InputHw,
                kernelSize: KernelSize,
                padding: 1,
                stride: 1,
                packedKernels: _convPackedKernels.AsReadOnlySpan());
        }

        /// <summary>
        /// L5a — production with the patch gather ablated, so what is left is the micro-kernel sweep plus
        /// everything the wrapper does around it: per-panel origins, the packed-buffer rent, the item-to-
        /// (panel, block) mapping and the dispatch.
        ///
        /// <para><b>Produces wrong output by construction.</b> It measures cost, never correctness. Read
        /// L5 minus L5a as the gather, and L5a minus L4 as the rest of the wrapper - both as upper bounds,
        /// because removing one side also frees the other's cache pressure.</para>
        /// </summary>
        [Benchmark]
        public void L5a_ProductionNoGather()
        {
            Conv2DGemmKernels.AblatePackB = true;

            try
            {
                L5_Production();
            }
            finally
            {
                Conv2DGemmKernels.AblatePackB = false;
            }
        }

        /// <summary>
        /// L5b — production with the micro-kernel ablated, so what is left is the gather and the wrapper.
        /// Wrong output by construction, same as L5a.
        /// </summary>
        [Benchmark]
        public void L5b_ProductionGatherOnly()
        {
            Conv2DGemmKernels.AblateMicroKernel = true;

            try
            {
                L5_Production();
            }
            finally
            {
                Conv2DGemmKernels.AblateMicroKernel = false;
            }
        }

        [GlobalCleanup]
        public void Dispose()
        {
            _packedA?.Dispose();
            _packedB?.Dispose();
            _outputReal?.Dispose();
            _outputTile?.Dispose();
            _convInput?.Dispose();
            _convKernels?.Dispose();
            _convPackedKernels?.Dispose();
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
    }
}
