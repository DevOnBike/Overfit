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
    /// Sizes ONE Q4_K prefill projection — the unit of work behind the measured <b>3.76× prefill gap</b> to
    /// llama.cpp (541.7 vs 144 tok/s on <c>qwen.q4km.gguf</c>, 672-token prompt, 2026-07-22).
    ///
    /// <para><b>What this is for.</b> Before writing a better GEMM, establish which of the two suspected causes
    /// actually costs us. Both are visible here:</para>
    /// <list type="number">
    ///   <item><b>Kernel quality</b> — <see cref="Tiled_SingleThread"/> vs <see cref="WeightStationary"/>·(cores)
    ///     isolates the kernel from the dispatcher. <c>GemmTiled</c>'s own doc admits its per-column
    ///     accumulators live in <c>stackalloc</c> scratch and spill; llama.cpp's
    ///     <c>ggml_gemm_q4_K_8x8_q8_K</c> holds a 4×16 tile in registers. A tile whose accumulators spill wins
    ///     nothing — which is exactly the 0.999× tie measured on 2026-07-21.</item>
    ///   <item><b>Parallel efficiency</b> — <see cref="Tiled"/> vs <see cref="Tiled_SingleThread"/> shows how
    ///     much of the machine the dispatcher actually extracts.</item>
    /// </list>
    ///
    /// <para><b>Shapes are the real ones</b> (Qwen2.5-3B: hidden 2048, intermediate 11008). The decode profile
    /// put FFN at 69.3% of the step, so <c>ffn_gate_up</c> and <c>ffn_down</c> are the shapes that decide the
    /// outcome; <c>attn_qo</c> is included because attention is dispatched per head and may behave differently.</para>
    ///
    /// <para><b>FLOP reference for converting ns → throughput:</b> one projection is
    /// <c>2 · rows · inputSize · outputSize</c> MACs — at rows=672 that is 30.3 GFLOP for the FFN shapes and
    /// 5.6 GFLOP for <c>attn_qo</c>. llama.cpp's whole-model 541.7 tok/s works out to ≈3.7 TFLOP/s-equivalent
    /// against our ≈1.0, so a kernel here needs to land near 3.5–4 TFLOP/s to close the gap.</para>
    ///
    /// <para>The shared <see cref="BenchmarkConfig"/> (InvocationCount=1) is correct here and NOT the trap
    /// described in CLAUDE.md: a single projection at these shapes runs for tens of milliseconds, not
    /// microseconds, so there is no timer-noise problem to fix with a microbenchmark job.</para>
    ///
    /// <para><b>The synthetic weight is deliberately NOT prepacked</b>, so <c>IsPrepacked</c> cannot
    /// short-circuit <c>UseTiledPrefillQ4K</c> — the dead-flag trap that invalidated an entire measurement
    /// round on 2026-07-21, when a <c>*.gguf.repack</c> sidecar silently made both A/B arms identical.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*Q4KPrefillProjection*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class Q4KPrefillProjectionBenchmark
    {
        /// <summary>Prompt length used in the llama.cpp comparison, so the numbers are directly relatable.</summary>
        [Params(672, 512)]
        public int Rows
        {
            get; set;
        }

        [Params("ffn_gate_up", "ffn_down", "attn_qo", "llama_ref")]
        public string Shape
        {
            get; set;
        } = "ffn_gate_up";

        private DecodeWeight _weight;
        private Q4KWeight _q4k = null!;
        private float[] _input = null!;
        private float[] _output = null!;
        private sbyte[] _quants = null!;
        private float[] _scales = null!;
        private short[] _bsums = null!;
        private int _inputSize;
        private int _outputSize;
        private bool _originalTiled;
        private bool _originalStationary;

        private static (int InputSize, int OutputSize) ShapeOf(string shape)
        {
            return shape switch
            {
                "ffn_gate_up" => (2048, 11008),
                "ffn_down" => (11008, 2048),

                // The exact shape llama.cpp's own test-backend-ops reports (m=4096, k=14336): 60.13 GFLOP
                // at n=512, where its AVX2 build measured 1.56 TFLOPS for q4_K. Same shape, same thread
                // count (32) - the only like-for-like kernel comparison available without editing their tests.
                "llama_ref" => (14336, 4096),

                _ => (2048, 2048),
            };
        }

        /// <summary>
        /// Declares each benchmark's work amount so the TFLOP/s and GB/s columns are computed in-repo.
        ///
        /// <para>Note that <see cref="QuantizeActivationsOnly"/> declares <b>bytes, not FLOPs</b>: it performs no
        /// multiply-accumulate, so crediting it with the matmul's FLOP count — as an out-of-repo script briefly
        /// did on 2026-07-22, yielding a fictitious 29.6 TFLOP/s — describes memory traffic as arithmetic.</para>
        /// </summary>
        public static WorkAmount GetWorkAmount(BenchmarkCase benchmarkCase)
        {
            var rows = (int)benchmarkCase.Parameters["Rows"];
            var (inputSize, outputSize) = ShapeOf((string)benchmarkCase.Parameters["Shape"]);

            if (benchmarkCase.Descriptor.WorkloadMethod.Name == nameof(QuantizeActivationsOnly))
            {
                // Reads rows×inputSize floats, writes the same count of sbyte quants plus per-super-block
                // scales and bsums — the scales/bsums are ~1% of the traffic and are not modelled.
                return WorkAmount.Memory((long)rows * inputSize * (sizeof(float) + sizeof(sbyte)));
            }

            return WorkAmount.Matmul(rows, inputSize, outputSize);
        }

        [GlobalSetup]
        public void Setup()
        {
            (_inputSize, _outputSize) = ShapeOf(Shape);

            _originalTiled = BatchedQuantProjection.UseTiledPrefillQ4K;
            _originalStationary = BatchedQuantProjection.UseWeightStationaryQ4K;

            var rng = new Random(20260722);

            // Quantize a random F32 matrix into a real Q4_K weight — the same layout the loader produces.
            var f32 = new float[(long)_outputSize * _inputSize];
            for (var i = 0; i < f32.Length; i++)
            {
                f32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            _q4k = new Q4KWeight(GgmlQuant.QuantizeQ4_K(f32, _inputSize, _outputSize), _inputSize, _outputSize);
            _weight = _q4k;

            _input = new float[(long)Rows * _inputSize];
            for (var i = 0; i < _input.Length; i++)
            {
                _input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            _output = new float[(long)Rows * _outputSize];

            // Activation-quantization scratch for the single-thread kernel path (the dispatcher pools its own).
            var superBlocksPerRow = _q4k.SuperBlocksPerRow;
            _quants = new sbyte[(long)Rows * _inputSize];
            _scales = new float[(long)Rows * superBlocksPerRow];
            _bsums = new short[(long)Rows * superBlocksPerRow * Q4KDotKernel.GroupsPerSuperBlock];

            // Pay the one-off repack here so it is not attributed to the timed region.
            _q4k.EnsureRepacked();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = _originalTiled;
            BatchedQuantProjection.UseWeightStationaryQ4K = _originalStationary;
            Q4KGemvKernel.AblateF16Scales = false;
            Q4KGemvKernel.AblateScaleUnpack = false;
            Q4KGemvKernel.AblateNibbleUnpack = false;
            BatchedQuantProjection.TileColsOverride = 0;
            _weight.Dispose();
        }

        /// <summary>Today's production path for a bias-free projection when the tiled kernel is NOT enabled:
        /// decode each super-block once, reuse it across the row tile. The baseline everything else is judged against.</summary>
        [Benchmark(Baseline = true)]
        public void WeightStationary()
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = false;
            BatchedQuantProjection.UseWeightStationaryQ4K = true;
            BatchedQuantProjection.Dispatch(_input, Rows, in _weight, [], _output, _inputSize, _outputSize);
        }

        /// <summary>The register-tiled GEMM over <c>block_q4_Kx8</c>, parallelised across row tiles.</summary>
        [Benchmark]
        public void Tiled()
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = true;
            BatchedQuantProjection.UseWeightStationaryQ4K = false;
            BatchedQuantProjection.Dispatch(_input, Rows, in _weight, [], _output, _inputSize, _outputSize);
        }

        /// <summary>
        /// Q8_K activation quantization ALONE, for the same <c>rows × inputSize</c> the projections consume.
        ///
        /// <para>This sizes the next lever. Attention dispatches Q once <b>per head</b> over a
        /// loop-invariant <c>hidden</c>, so this cost is paid 16× per layer where once would do — a prefill
        /// profile put <c>attn_q</c> at 621.7 ms across 576 calls. Decode already fixed exactly this
        /// (<c>ProjectPreQuantized</c>, 2026-05); prefill never got the equivalent. What this benchmark
        /// answers is whether the redundant share is worth ~5% or ~13% of prefill — two estimates that
        /// differ by enough to change the decision.</para>
        /// </summary>
        [Benchmark]
        public void QuantizeActivationsOnly()
        {
            var superBlocksPerRow = _q4k.SuperBlocksPerRow;
            var bsumsPerRow = superBlocksPerRow * Q4KDotKernel.GroupsPerSuperBlock;

            for (var n = 0; n < Rows; n++)
            {
                Q4KDotKernel.QuantizeActivationQ8K(
                    _input.AsSpan(n * _inputSize, _inputSize),
                    _quants.AsSpan(n * _inputSize, _inputSize),
                    _scales.AsSpan(n * superBlocksPerRow, superBlocksPerRow),
                    _bsums.AsSpan(n * bsumsPerRow, bsumsPerRow));
            }
        }

        /// <summary>
        /// <see cref="Tiled"/> with the column tile forced to 8 — today's dispatcher choice, and the arm the
        /// wider tiles below are judged against.
        ///
        /// <para>A tile of NR columns walks the whole weight matrix, so the matrix is streamed <c>rows/NR</c>
        /// times. At NR=8 and 672 rows that is 84 passes over 12.68 MB = 1.07 GB per projection, roughly 70 GB/s
        /// against a measured 90 GB/s ceiling. If that traffic is what caps the kernel at 1.98 TFLOP/s against
        /// its instruction mix's 4.6, halving it should show here.</para>
        /// </summary>
        [Benchmark]
        public void Tiled_Cols8()
        {
            RunTiledWithCols(8);
        }

        /// <summary>Half the weight traffic of <see cref="Tiled_Cols8"/> — 42 passes instead of 84.</summary>
        [Benchmark]
        public void Tiled_Cols16()
        {
            RunTiledWithCols(16);
        }

        /// <summary>Twice the traffic of <see cref="Tiled_Cols8"/>, to confirm the trend runs both ways.</summary>
        [Benchmark]
        public void Tiled_Cols4()
        {
            RunTiledWithCols(4);
        }

        private void RunTiledWithCols(int cols)
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = true;
            BatchedQuantProjection.UseWeightStationaryQ4K = false;
            BatchedQuantProjection.TileColsOverride = cols;
            BatchedQuantProjection.Dispatch(_input, Rows, in _weight, [], _output, _inputSize, _outputSize);
            BatchedQuantProjection.TileColsOverride = 0;
        }

        /// <summary>
        /// <see cref="Tiled"/> with the per-block F16 scale/min decode replaced by constants — i.e. without
        /// <c>LoadF16x8Rearrange</c>, which stores a vector to <c>stackalloc</c> and reads it back through
        /// eight separate <c>BitConverter.UInt16BitsToHalf</c> calls, plus the second <c>LoadF16x8</c>.
        ///
        /// <para>This and the two ablations below split the unexplained gap between the kernel's measured
        /// 1.70 TFLOP/s and the 4.64 its own arithmetic instruction mix reaches — the residual is work the mix
        /// benchmark never modelled, and each of these is a candidate. Ratios against <see cref="Tiled"/> are
        /// <b>upper</b> bounds: removing a computation also lets the JIT fold what depended on it.</para>
        /// </summary>
        [Benchmark]
        public void Tiled_NoF16Decode()
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = true;
            BatchedQuantProjection.UseWeightStationaryQ4K = false;
            Q4KGemvKernel.AblateF16Scales = true;
            BatchedQuantProjection.Dispatch(_input, Rows, in _weight, [], _output, _inputSize, _outputSize);
            Q4KGemvKernel.AblateF16Scales = false;
        }

        /// <summary><see cref="Tiled"/> without the scalar <c>Unpack</c> of the 6-bit sub-block scales.</summary>
        [Benchmark]
        public void Tiled_NoScaleUnpack()
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = true;
            BatchedQuantProjection.UseWeightStationaryQ4K = false;
            Q4KGemvKernel.AblateScaleUnpack = true;
            BatchedQuantProjection.Dispatch(_input, Rows, in _weight, [], _output, _inputSize, _outputSize);
            Q4KGemvKernel.AblateScaleUnpack = false;
        }

        /// <summary><see cref="Tiled"/> without the 16 <c>And</c>/shift ops that split bytes into nibbles.</summary>
        [Benchmark]
        public void Tiled_NoNibbleUnpack()
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = true;
            BatchedQuantProjection.UseWeightStationaryQ4K = false;
            Q4KGemvKernel.AblateNibbleUnpack = true;
            BatchedQuantProjection.Dispatch(_input, Rows, in _weight, [], _output, _inputSize, _outputSize);
            Q4KGemvKernel.AblateNibbleUnpack = false;
        }

        /// <summary>The original re-decode-per-row kernel — kept as the reference the kernel docs' "~3×" claim
        /// is actually measured against.</summary>
        [Benchmark]
        public void ReDecodePerRow()
        {
            BatchedQuantProjection.UseTiledPrefillQ4K = false;
            BatchedQuantProjection.UseWeightStationaryQ4K = false;
            BatchedQuantProjection.Dispatch(_input, Rows, in _weight, [], _output, _inputSize, _outputSize);
        }

        /// <summary>
        /// <c>GemmTiled</c> over every row tile on ONE thread. Divided into <see cref="Tiled"/> this gives the
        /// dispatcher's parallel efficiency; on its own it is the raw kernel throughput to compare against
        /// llama.cpp's single-thread rate — the number that says whether the accumulator spill is the problem.
        /// </summary>
        [Benchmark]
        public void Tiled_SingleThread()
        {
            var superBlocksPerRow = _q4k.SuperBlocksPerRow;
            var bsumsPerRow = superBlocksPerRow * Q4KDotKernel.GroupsPerSuperBlock;

            for (var n = 0; n < Rows; n++)
            {
                Q4KDotKernel.QuantizeActivationQ8K(
                    _input.AsSpan(n * _inputSize, _inputSize),
                    _quants.AsSpan(n * _inputSize, _inputSize),
                    _scales.AsSpan(n * superBlocksPerRow, superBlocksPerRow),
                    _bsums.AsSpan(n * bsumsPerRow, bsumsPerRow));
            }

            var repacked = _q4k.EnsureRepacked();
            const int TileCols = 8;

            for (var start = 0; start < Rows; start += TileCols)
            {
                var cols = Math.Min(TileCols, Rows - start);
                Q4KGemvKernel.GemmTiled(
                    repacked,
                    _outputSize,
                    _inputSize,
                    cols,
                    _quants.AsSpan(start * _inputSize, cols * _inputSize),
                    _scales.AsSpan(start * superBlocksPerRow, cols * superBlocksPerRow),
                    _bsums.AsSpan(start * bsumsPerRow, cols * bsumsPerRow),
                    _output.AsSpan(start * _outputSize, cols * _outputSize));
            }
        }
    }
}
