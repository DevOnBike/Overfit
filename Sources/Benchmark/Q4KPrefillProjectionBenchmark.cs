// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
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
        [Params(672)]
        public int Rows
        {
            get; set;
        }

        [Params("ffn_gate_up", "ffn_down", "attn_qo")]
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

        [GlobalSetup]
        public void Setup()
        {
            (_inputSize, _outputSize) = Shape switch
            {
                "ffn_gate_up" => (2048, 11008),
                "ffn_down" => (11008, 2048),
                _ => (2048, 2048),
            };

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
