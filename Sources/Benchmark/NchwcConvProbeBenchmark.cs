// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// `XC-91` — does a managed direct convolution over NCHWc-blocked channels beat our im2col path on one
    /// VGG layer, single-threaded?
    ///
    /// <para><b>What this decides.</b> Measured 2026-08-19 by forcing ONNX Runtime down to
    /// <c>ORT_ENABLE_EXTENDED</c>, where its own NCHWc transform does not run: their convolution costs
    /// <b>7.65 ms blocked against 11.57 ms as im2col plus GEMM</b>, so <b>the layout is worth 1.51x on their
    /// assembly</b>. Ours is 17.74 ms in the same structure. The open question is not whether the layout
    /// helps — it does — but whether a <b>managed</b> kernel can capture it.</para>
    ///
    /// <para><b>Why that is genuinely in doubt, from this repository's own history.</b> A direct convolution
    /// holds an output tile live across the whole kernel window, so it wants more simultaneously-live vector
    /// registers than the im2col micro-kernel does. The largest single win on this branch — <b>3.46x</b> —
    /// came from stopping the existing micro-kernel spilling its sixteen accumulators. The NCHWc kernel is a
    /// harder register-allocation problem than the one the JIT has already failed once here.</para>
    ///
    /// <para><b>Scope is deliberately the kernel and nothing else.</b> Blocked input and filter are prepared
    /// in <see cref="Setup"/>, so no reorder cost is charged and the cache is warm — every advantage is given
    /// to the candidate. There is no pooling, no importer and no graph work. <b>Kill criterion: if it does
    /// not win here, it cannot win in the product, and `XC-90` closes for the price of one benchmark.</b></para>
    ///
    /// <para><b>Correctness is checked before either number is believed</b>, in <see cref="Setup"/>, against
    /// the production <see cref="ConvLayer"/> on the same weights. A fast kernel computing the wrong thing is
    /// easy to write by accident and worth nothing.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*NchwcConvProbe*"
    /// </summary>
    [Config(typeof(Config))]
    public unsafe class NchwcConvProbeBenchmark
    {
        private const int Kernel = 3;
        private const int Padding = 1;

        /// <summary>
        /// Which VGG-16 layer. Both are 3.70 GFLOP, and they sit at opposite ends of the shape range that
        /// matters here.
        ///
        /// <para><c>conv3_2</c> is compute-dense — <c>K = 2304</c>, <c>N = 3136</c> — and our im2col path
        /// already runs it at <b>83% of this machine's single-core FMA ceiling</b>, so no layout can win much
        /// there. <c>conv1_2</c> is gather-heavy — <c>K = 576</c>, <c>N = 50176</c>, an im2col matrix of
        /// <b>115 MB</b> — and we run it at <b>53%</b>. If the NCHWc layout pays anywhere, it pays there, and
        /// measuring only the first shape would have produced a verdict that did not cover the case.</para>
        /// </summary>
        [Params("conv3_2", "conv1_2")]
        public string Layer { get; set; } = "conv3_2";

        private int Channels;
        private int Height;
        private int Width;

        /// <summary>AVX-512 channel block, from MLAS <c>platform.cpp</c>: <c>NchwcBlockSize = 16</c>.</summary>
        private const int Block = 16;

        /// <summary>Output positions held in registers at once — eight accumulators plus one weight vector.</summary>
        private const int Lanes = 8;

        private ConvLayer _layer = null!;
        private float[] _input = null!;
        private float[] _output = null!;

        private float[] _blockedInput = null!;
        private float[] _blockedFilter = null!;
        private float[] _blockedOutput = null!;

        /// <summary>
        /// Sixteen threads — the physical core count on this machine.
        ///
        /// <para><b>It started at one, and one was the wrong scope.</b> Single-threaded, our im2col path runs
        /// at <b>83% of this machine's single-core FMA ceiling</b>, so no layout can be worth the 1.51x
        /// measured against ONNX Runtime — there is only 20% of headroom in total. That 1.51x was measured at
        /// <b>16 cores</b>, where im2col materialises a K x N matrix and its traffic meets the memory system.
        /// A single-core probe cannot see a bandwidth effect. Serial readings from the first run are kept in
        /// <c>docs/measured-baselines.md</c>.</para>
        ///
        /// <para>Warmup is 25 because the shared config's 5 has already read tier-0 code as a result on this
        /// machine — and did so again on this benchmark's first run, reporting 60.02 ms for an arm that
        /// measures 12.48 ms warm.</para>
        /// </summary>
        private sealed class Config : ManualConfig
        {
            public Config()
            {
                AddJob(Job.Default
                    .WithWarmupCount(25)
                    .WithIterationCount(15)
                    .WithInvocationCount(1)
                    .WithUnrollFactor(1)
                    .WithEnvironmentVariable("DOTNET_PROCESSOR_COUNT", "16"));
            }
        }

        [GlobalSetup]
        public void Setup()
        {
            if (Layer == "conv1_2")
            {
                Channels = 64;
                Height = 224;
                Width = 224;
            }

            if (Layer != "conv1_2")
            {
                Channels = 256;
                Height = 56;
                Width = 56;
            }

            var rng = new Random(20260819);

            _input = new float[Channels * Height * Width];
            _output = new float[Channels * Height * Width];

            for (var i = 0; i < _input.Length; i++)
            {
                _input[i] = (float)(rng.NextDouble() - 0.5);
            }

            var kernels = new float[Channels * Channels * Kernel * Kernel];

            for (var i = 0; i < kernels.Length; i++)
            {
                kernels[i] = (float)((rng.NextDouble() - 0.5) * 0.1);
            }

            _layer = new ConvLayer(Channels, Channels, Height, Width, Kernel, Padding, stride: 1);
            _layer.LoadParameters(kernels);
            _layer.Eval();

            _blockedInput = new float[_input.Length];
            _blockedFilter = new float[kernels.Length];
            _blockedOutput = new float[_output.Length];

            BlockActivation(_input, _blockedInput, Channels, Height, Width);
            BlockFilter(kernels, _blockedFilter);

            // Both paths on the same weights, then compared. The blocked output is unblocked first so the
            // comparison is against the production layout rather than against this file's own convention.
            _layer.ForwardInference(_input, _output);
            NchwcDirect();

            var unblocked = new float[_output.Length];
            UnblockActivation(_blockedOutput, unblocked, Channels, Height, Width);

            var worst = 0f;
            var scale = 0f;

            for (var i = 0; i < unblocked.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(unblocked[i] - _output[i]));
                scale = Math.Max(scale, Math.Abs(_output[i]));
            }

            Console.WriteLine($"[PROBE] NCHWc vs ConvLayer — max abs diff {worst:E3}, output scale {scale:E3}, "
                              + $"relative {worst / scale:E3}");

            // Timed here as well as by the harness, because the harness is the thing under suspicion: a
            // standalone run of this exact layer reads 12.594 ms at one thread and the harness reported
            // 60.02 ms for the same arm. Whichever of the two is wrong, the probe cannot be read until they
            // agree.
            Console.WriteLine($"[PROBE] ProcessorCount={Environment.ProcessorCount}");

            for (var i = 0; i < 30; i++)
            {
                _layer.ForwardInference(_input, _output);
            }

            var clock = System.Diagnostics.Stopwatch.StartNew();
            var calls = 0;

            while (clock.Elapsed.TotalSeconds < 4.0)
            {
                _layer.ForwardInference(_input, _output);
                calls++;
            }

            clock.Stop();
            Console.WriteLine($"[PROBE] im2col in-setup {clock.Elapsed.TotalMilliseconds / calls:F3} ms/call");

            // The variant is checked against the same reference before it is timed.
            NchwcDirect4x4();
            UnblockActivation(_blockedOutput, unblocked, Channels, Height, Width);

            var worst4 = 0f;

            for (var i = 0; i < unblocked.Length; i++)
            {
                worst4 = Math.Max(worst4, Math.Abs(unblocked[i] - _output[i]));
            }

            Console.WriteLine($"[PROBE] NCHWc 4x4 vs ConvLayer — relative {worst4 / scale:E3}");

            if (worst4 / scale > 1e-5f)
            {
                throw new InvalidOperationException(
                    $"the 4x4 NCHWc kernel does not compute the same convolution — relative {worst4 / scale:E3}");
            }

            for (var i = 0; i < 10; i++)
            {
                NchwcDirect();
            }

            clock.Restart();
            calls = 0;

            while (clock.Elapsed.TotalSeconds < 4.0)
            {
                NchwcDirect();
                calls++;
            }

            clock.Stop();
            Console.WriteLine($"[PROBE] nchwc  in-setup {clock.Elapsed.TotalMilliseconds / calls:F3} ms/call");

            for (var i = 0; i < 10; i++)
            {
                NchwcDirect4x4();
            }

            clock.Restart();
            calls = 0;

            while (clock.Elapsed.TotalSeconds < 4.0)
            {
                NchwcDirect4x4();
                calls++;
            }

            clock.Stop();
            Console.WriteLine($"[PROBE] nchwc4x4 in-setup {clock.Elapsed.TotalMilliseconds / calls:F3} ms/call");

            NchwcDirect4x4Parallel();
            UnblockActivation(_blockedOutput, unblocked, Channels, Height, Width);

            var worstParallel = 0f;

            for (var i = 0; i < unblocked.Length; i++)
            {
                worstParallel = Math.Max(worstParallel, Math.Abs(unblocked[i] - _output[i]));
            }

            Console.WriteLine($"[PROBE] NCHWc 4x4 parallel vs ConvLayer — relative {worstParallel / scale:E3}");

            if (worstParallel / scale > 1e-5f)
            {
                throw new InvalidOperationException(
                    $"the parallel NCHWc kernel does not compute the same convolution — "
                    + $"relative {worstParallel / scale:E3}");
            }

            for (var i = 0; i < 20; i++)
            {
                NchwcDirect4x4Parallel();
            }

            clock.Restart();
            calls = 0;

            while (clock.Elapsed.TotalSeconds < 4.0)
            {
                NchwcDirect4x4Parallel();
                calls++;
            }

            clock.Stop();
            Console.WriteLine(
                $"[PROBE] nchwc4x4par in-setup {clock.Elapsed.TotalMilliseconds / calls:F3} ms/call");

            // fp32 accumulation in a different order over 2304 terms: a relative difference this size is
            // reassociation, not a defect. Anything larger is a different computation.
            if (worst / scale > 1e-5f)
            {
                throw new InvalidOperationException(
                    $"the NCHWc kernel does not compute the same convolution — relative difference "
                    + $"{worst / scale:E3}. Neither timing below means anything.");
            }
        }

        /// <summary>The shipped path for this layer: fused im2col into a packed panel, then the GEMM micro-kernel.</summary>
        [Benchmark(Baseline = true)]
        public float Im2ColGemm()
        {
            _layer.ForwardInference(_input, _output);

            return _output[0];
        }

        /// <summary>
        /// Direct convolution over blocked channels. The sixteen output channels of a block are the vector
        /// lanes, so one weight load serves <see cref="Lanes"/> output positions and the accumulators never
        /// leave registers inside the window sweep.
        /// </summary>
        [Benchmark]
        public float NchwcDirect()
        {
            var blocks = Channels / Block;
            var window = Kernel * Kernel;

            fixed (float* input = _blockedInput, filter = _blockedFilter, output = _blockedOutput)
            {
                for (var ocb = 0; ocb < blocks; ocb++)
                {
                    var outBlock = output + ((long)ocb * Height * Width * Block);

                    for (var oh = 0; oh < Height; oh++)
                    {
                        var outRow = outBlock + ((long)oh * Width * Block);

                        for (var ow = 0; ow < Width; ow += Lanes)
                        {
                            var interior = ow >= Padding && ow + Lanes + Kernel - 1 - Padding < Width;

                            if (interior)
                            {
                                SweepInterior(input, filter, outRow + ((long)ow * Block), ocb, oh, ow, blocks);

                                continue;
                            }

                            SweepEdge(input, filter, outRow + ((long)ow * Block), ocb, oh, ow, blocks, window);
                        }
                    }
                }
            }

            return _blockedOutput[0];
        }

        /// <summary>
        /// The fast path: eight named accumulators, no bounds test in the inner loop, one weight vector load
        /// per (input channel, kernel position) serving all eight output positions.
        /// </summary>
        private void SweepInterior(
            float* input, float* filter, float* outAt, int ocb, int oh, int ow, int blocks)
        {
            var a0 = Vector512<float>.Zero;
            var a1 = Vector512<float>.Zero;
            var a2 = Vector512<float>.Zero;
            var a3 = Vector512<float>.Zero;
            var a4 = Vector512<float>.Zero;
            var a5 = Vector512<float>.Zero;
            var a6 = Vector512<float>.Zero;
            var a7 = Vector512<float>.Zero;

            for (var icb = 0; icb < blocks; icb++)
            {
                var inBlock = input + ((long)icb * Height * Width * Block);
                var filterBlock = filter
                    + ((((long)ocb * blocks) + icb) * Kernel * Kernel * Block * Block);

                for (var ky = 0; ky < Kernel; ky++)
                {
                    var ih = oh + ky - Padding;

                    if ((uint)ih >= Height)
                    {
                        continue;
                    }

                    var inRow = inBlock + ((long)ih * Width * Block);

                    for (var kx = 0; kx < Kernel; kx++)
                    {
                        var inAt = inRow + ((long)(ow + kx - Padding) * Block);
                        var weights = filterBlock + ((((long)ky * Kernel) + kx) * Block * Block);

                        for (var ic = 0; ic < Block; ic++)
                        {
                            var w = Vector512.Load(weights + ((long)ic * Block));
                            var x = inAt + ic;

                            a0 = Avx512F.FusedMultiplyAdd(Vector512.Create(x[0]), w, a0);
                            a1 = Avx512F.FusedMultiplyAdd(Vector512.Create(x[Block]), w, a1);
                            a2 = Avx512F.FusedMultiplyAdd(Vector512.Create(x[Block * 2]), w, a2);
                            a3 = Avx512F.FusedMultiplyAdd(Vector512.Create(x[Block * 3]), w, a3);
                            a4 = Avx512F.FusedMultiplyAdd(Vector512.Create(x[Block * 4]), w, a4);
                            a5 = Avx512F.FusedMultiplyAdd(Vector512.Create(x[Block * 5]), w, a5);
                            a6 = Avx512F.FusedMultiplyAdd(Vector512.Create(x[Block * 6]), w, a6);
                            a7 = Avx512F.FusedMultiplyAdd(Vector512.Create(x[Block * 7]), w, a7);
                        }
                    }
                }
            }

            Vector512.Store(a0, outAt);
            Vector512.Store(a1, outAt + Block);
            Vector512.Store(a2, outAt + (Block * 2));
            Vector512.Store(a3, outAt + (Block * 3));
            Vector512.Store(a4, outAt + (Block * 4));
            Vector512.Store(a5, outAt + (Block * 5));
            Vector512.Store(a6, outAt + (Block * 6));
            Vector512.Store(a7, outAt + (Block * 7));
        }

        /// <summary>
        /// The same convolution with the load-to-FMA ratio inverted: <b>four</b> output-channel blocks and
        /// four output positions, so one broadcast of an input scalar drives four FMAs instead of one.
        ///
        /// <para><see cref="NchwcDirect"/> loads one weight vector and issues eight broadcasts per
        /// <c>(input channel, kernel position)</c> — <b>1.125 loads per FMA</b>, which saturates the load
        /// ports. This shape issues four weight loads and four broadcasts for sixteen FMAs — <b>0.5 loads per
        /// FMA</b>. Sixteen accumulators is deliberately the budget the im2col micro-kernel already holds in
        /// registers on this JIT, so it is a shape known to fit rather than a guess.</para>
        /// </summary>
        [Benchmark]
        public float NchwcDirect4x4()
        {
            var blocks = Channels / Block;

            fixed (float* input = _blockedInput, filter = _blockedFilter, output = _blockedOutput)
            {
                for (var ocb = 0; ocb < blocks; ocb += 4)
                {
                    for (var oh = 0; oh < Height; oh++)
                    {
                        for (var ow = 0; ow < Width; ow += 4)
                        {
                            var interior = ow >= Padding && ow + 4 + Kernel - 1 - Padding < Width;

                            if (interior)
                            {
                                Sweep4x4(input, filter, output, ocb, oh, ow, blocks);

                                continue;
                            }

                            for (var o = 0; o < 4; o++)
                            {
                                SweepEdgeGeneric(input, filter, output, ocb + o, oh, ow, 4, blocks);
                            }
                        }
                    }
                }
            }

            return _blockedOutput[0];
        }

        /// <summary>Sixteen accumulators: four output-channel blocks by four output positions.</summary>
        private void Sweep4x4(
            float* input, float* filter, float* output, int ocb, int oh, int ow, int blocks)
        {
            Sweep4x4Static(input, filter, output, ocb, oh, ow, blocks, Height, Width);
        }

        /// <summary>The 4x4 kernel proper. Shape travels as arguments so the parallel worker, which is a
        /// function pointer and cannot be an instance method, runs this exact code rather than a copy.</summary>
        private static void Sweep4x4Static(
            float* input, float* filter, float* output,
            int ocb, int oh, int ow, int blocks, int Height, int Width)
        {
            var a00 = Vector512<float>.Zero; var a01 = Vector512<float>.Zero;
            var a02 = Vector512<float>.Zero; var a03 = Vector512<float>.Zero;
            var a10 = Vector512<float>.Zero; var a11 = Vector512<float>.Zero;
            var a12 = Vector512<float>.Zero; var a13 = Vector512<float>.Zero;
            var a20 = Vector512<float>.Zero; var a21 = Vector512<float>.Zero;
            var a22 = Vector512<float>.Zero; var a23 = Vector512<float>.Zero;
            var a30 = Vector512<float>.Zero; var a31 = Vector512<float>.Zero;
            var a32 = Vector512<float>.Zero; var a33 = Vector512<float>.Zero;

            var filterStride = (long)blocks * Kernel * Kernel * Block * Block;

            for (var icb = 0; icb < blocks; icb++)
            {
                var inBlock = input + ((long)icb * Height * Width * Block);
                var filterBase = filter + ((((long)ocb * blocks) + icb) * Kernel * Kernel * Block * Block);

                for (var ky = 0; ky < Kernel; ky++)
                {
                    var ih = oh + ky - Padding;

                    if ((uint)ih >= Height)
                    {
                        continue;
                    }

                    var inRow = inBlock + ((long)ih * Width * Block);

                    for (var kx = 0; kx < Kernel; kx++)
                    {
                        var inAt = inRow + ((long)(ow + kx - Padding) * Block);
                        var weights = filterBase + ((((long)ky * Kernel) + kx) * Block * Block);

                        for (var ic = 0; ic < Block; ic++)
                        {
                            var at = weights + ((long)ic * Block);
                            var w0 = Vector512.Load(at);
                            var w1 = Vector512.Load(at + filterStride);
                            var w2 = Vector512.Load(at + (filterStride * 2));
                            var w3 = Vector512.Load(at + (filterStride * 3));

                            var x = inAt + ic;
                            var x0 = Vector512.Create(x[0]);
                            var x1 = Vector512.Create(x[Block]);
                            var x2 = Vector512.Create(x[Block * 2]);
                            var x3 = Vector512.Create(x[Block * 3]);

                            a00 = Avx512F.FusedMultiplyAdd(x0, w0, a00);
                            a01 = Avx512F.FusedMultiplyAdd(x1, w0, a01);
                            a02 = Avx512F.FusedMultiplyAdd(x2, w0, a02);
                            a03 = Avx512F.FusedMultiplyAdd(x3, w0, a03);

                            a10 = Avx512F.FusedMultiplyAdd(x0, w1, a10);
                            a11 = Avx512F.FusedMultiplyAdd(x1, w1, a11);
                            a12 = Avx512F.FusedMultiplyAdd(x2, w1, a12);
                            a13 = Avx512F.FusedMultiplyAdd(x3, w1, a13);

                            a20 = Avx512F.FusedMultiplyAdd(x0, w2, a20);
                            a21 = Avx512F.FusedMultiplyAdd(x1, w2, a21);
                            a22 = Avx512F.FusedMultiplyAdd(x2, w2, a22);
                            a23 = Avx512F.FusedMultiplyAdd(x3, w2, a23);

                            a30 = Avx512F.FusedMultiplyAdd(x0, w3, a30);
                            a31 = Avx512F.FusedMultiplyAdd(x1, w3, a31);
                            a32 = Avx512F.FusedMultiplyAdd(x2, w3, a32);
                            a33 = Avx512F.FusedMultiplyAdd(x3, w3, a33);
                        }
                    }
                }
            }

            var plane = (long)Height * Width * Block;
            var outAt = output + ((long)ocb * plane) + ((long)oh * Width * Block) + ((long)ow * Block);

            Vector512.Store(a00, outAt);
            Vector512.Store(a01, outAt + Block);
            Vector512.Store(a02, outAt + (Block * 2));
            Vector512.Store(a03, outAt + (Block * 3));
            Vector512.Store(a10, outAt + plane);
            Vector512.Store(a11, outAt + plane + Block);
            Vector512.Store(a12, outAt + plane + (Block * 2));
            Vector512.Store(a13, outAt + plane + (Block * 3));
            Vector512.Store(a20, outAt + (plane * 2));
            Vector512.Store(a21, outAt + (plane * 2) + Block);
            Vector512.Store(a22, outAt + (plane * 2) + (Block * 2));
            Vector512.Store(a23, outAt + (plane * 2) + (Block * 3));
            Vector512.Store(a30, outAt + (plane * 3));
            Vector512.Store(a31, outAt + (plane * 3) + Block);
            Vector512.Store(a32, outAt + (plane * 3) + (Block * 2));
            Vector512.Store(a33, outAt + (plane * 3) + (Block * 3));
        }

        /// <summary>
        /// The 4x4 kernel across every worker, split over <c>(output-channel group, output row)</c> — 4 x 56
        /// = 224 items for 16 workers, so the tail is one item deep.
        ///
        /// <para>This is the arm that can see what the single-core arms cannot: whether the layout's value is
        /// the memory traffic im2col creates at scale rather than anything about the kernel itself.</para>
        /// </summary>
        [Benchmark]
        public float NchwcDirect4x4Parallel()
        {
            var blocks = Channels / Block;

            fixed (float* input = _blockedInput, filter = _blockedFilter, output = _blockedOutput)
            {
                var context = new ParallelContext(input, filter, output, blocks, Height, Width);

                OverfitParallel.For(0, (blocks / 4) * Height, 1, &RowWorker, &context);
            }

            return _blockedOutput[0];
        }

        private static void RowWorker(int itemStart, int itemEnd, void* contextPtr)
        {
            ref readonly var context = ref Unsafe.AsRef<ParallelContext>(contextPtr);

            var height = context.Height;
            var width = context.Width;

            for (var item = itemStart; item < itemEnd; item++)
            {
                var group = item / height;
                var oh = item - (group * height);
                var ocb = group * 4;

                for (var ow = 0; ow < width; ow += 4)
                {
                    var interior = ow >= Padding && ow + 4 + Kernel - 1 - Padding < width;

                    if (interior)
                    {
                        Sweep4x4Static(
                            context.Input, context.Filter, context.Output,
                            ocb, oh, ow, context.Blocks, height, width);

                        continue;
                    }

                    for (var o = 0; o < 4; o++)
                    {
                        SweepEdgeGenericStatic(
                            context.Input, context.Filter, context.Output,
                            ocb + o, oh, ow, 4, context.Blocks, height, width);
                    }
                }
            }
        }

        private readonly struct ParallelContext
        {
            public readonly float* Input;
            public readonly float* Filter;
            public readonly float* Output;
            public readonly int Blocks;
            public readonly int Height;
            public readonly int Width;

            public ParallelContext(
                float* input, float* filter, float* output, int blocks, int height, int width)
            {
                Input = input;
                Filter = filter;
                Output = output;
                Blocks = blocks;
                Height = height;
                Width = width;
            }
        }

        /// <summary>One output-channel block over <paramref name="count"/> positions, fully bounds-checked.</summary>
        private void SweepEdgeGeneric(
            float* input, float* filter, float* output, int ocb, int oh, int ow, int count, int blocks)
        {
            SweepEdgeGenericStatic(input, filter, output, ocb, oh, ow, count, blocks, Height, Width);
        }

        private static void SweepEdgeGenericStatic(
            float* input, float* filter, float* output,
            int ocb, int oh, int ow, int count, int blocks, int Height, int Width)
        {
            var plane = (long)Height * Width * Block;
            var outAt = output + ((long)ocb * plane) + ((long)oh * Width * Block) + ((long)ow * Block);

            for (var n = 0; n < count; n++)
            {
                Vector512.Store(Vector512<float>.Zero, outAt + ((long)n * Block));
            }

            for (var icb = 0; icb < blocks; icb++)
            {
                var inBlock = input + ((long)icb * Height * Width * Block);
                var filterBase = filter + ((((long)ocb * blocks) + icb) * Kernel * Kernel * Block * Block);

                for (var ky = 0; ky < Kernel; ky++)
                {
                    var ih = oh + ky - Padding;

                    if ((uint)ih >= Height)
                    {
                        continue;
                    }

                    var inRow = inBlock + ((long)ih * Width * Block);

                    for (var kx = 0; kx < Kernel; kx++)
                    {
                        var weights = filterBase + ((((long)ky * Kernel) + kx) * Block * Block);

                        for (var n = 0; n < count; n++)
                        {
                            var iw = ow + n + kx - Padding;

                            if ((uint)iw >= Width)
                            {
                                continue;
                            }

                            var inAt = inRow + ((long)iw * Block);
                            var slot = outAt + ((long)n * Block);
                            var acc = Vector512.Load(slot);

                            for (var ic = 0; ic < Block; ic++)
                            {
                                acc = Avx512F.FusedMultiplyAdd(
                                    Vector512.Create(inAt[ic]),
                                    Vector512.Load(weights + ((long)ic * Block)),
                                    acc);
                            }

                            Vector512.Store(acc, slot);
                        }
                    }
                }
            }
        }

        /// <summary>The edge groups, where some output positions read outside the input. Two of seven on this shape.</summary>
        private void SweepEdge(
            float* input, float* filter, float* outAt, int ocb, int oh, int ow, int blocks, int window)
        {
            for (var n = 0; n < Lanes; n++)
            {
                Vector512.Store(Vector512<float>.Zero, outAt + ((long)n * Block));
            }

            for (var icb = 0; icb < blocks; icb++)
            {
                var inBlock = input + ((long)icb * Height * Width * Block);
                var filterBlock = filter
                    + ((((long)ocb * blocks) + icb) * window * Block * Block);

                for (var ky = 0; ky < Kernel; ky++)
                {
                    var ih = oh + ky - Padding;

                    if ((uint)ih >= Height)
                    {
                        continue;
                    }

                    var inRow = inBlock + ((long)ih * Width * Block);

                    for (var kx = 0; kx < Kernel; kx++)
                    {
                        var weights = filterBlock + ((((long)ky * Kernel) + kx) * Block * Block);

                        for (var n = 0; n < Lanes; n++)
                        {
                            var iw = ow + n + kx - Padding;

                            if ((uint)iw >= Width)
                            {
                                continue;
                            }

                            var inAt = inRow + ((long)iw * Block);
                            var slot = outAt + ((long)n * Block);
                            var acc = Vector512.Load(slot);

                            for (var ic = 0; ic < Block; ic++)
                            {
                                acc = Avx512F.FusedMultiplyAdd(
                                    Vector512.Create(inAt[ic]),
                                    Vector512.Load(weights + ((long)ic * Block)),
                                    acc);
                            }

                            Vector512.Store(acc, slot);
                        }
                    }
                }
            }
        }

        /// <summary><c>[C][H][W]</c> to <c>[C/16][H][W][16]</c>.</summary>
        private void BlockActivation(float[] source, float[] target, int channels, int height, int width)
        {
            var plane = height * width;

            for (var c = 0; c < channels; c++)
            {
                var block = c / Block;
                var lane = c % Block;

                for (var p = 0; p < plane; p++)
                {
                    target[(((long)block * plane) + p) * Block + lane] = source[((long)c * plane) + p];
                }
            }
        }

        private void UnblockActivation(float[] source, float[] target, int channels, int height, int width)
        {
            var plane = height * width;

            for (var c = 0; c < channels; c++)
            {
                var block = c / Block;
                var lane = c % Block;

                for (var p = 0; p < plane; p++)
                {
                    target[((long)c * plane) + p] = source[(((long)block * plane) + p) * Block + lane];
                }
            }
        }

        /// <summary>
        /// <c>[OC][IC][kH][kW]</c> to <c>[OC/16][IC/16][kH][kW][ic 16][oc 16]</c> — MLAS's <c>OIHWBiBo</c>.
        /// The innermost sixteen floats are the output channels of one block, which is what makes a single
        /// vector load serve every output position in the sweep.
        /// </summary>
        private void BlockFilter(float[] source, float[] target)
        {
            var window = Kernel * Kernel;
            var blocks = Channels / Block;

            for (var oc = 0; oc < Channels; oc++)
            {
                for (var ic = 0; ic < Channels; ic++)
                {
                    for (var k = 0; k < window; k++)
                    {
                        var value = source[((((long)oc * Channels) + ic) * window) + k];
                        var index = ((((long)(oc / Block) * blocks) + (ic / Block)) * window + k) * Block * Block
                                    + ((long)(ic % Block) * Block)
                                    + (oc % Block);

                        target[index] = value;
                    }
                }
            }
        }
    }
}
