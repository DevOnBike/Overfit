// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;

namespace Benchmarks
{
    /// <summary>
    /// Prefill and decode of a real GGUF, end to end, in the two shapes <c>llama-bench</c> measures.
    ///
    /// <code>
    /// dotnet run -c Release --project Sources/Benchmark -- --filter "*GgufEndToEndThroughput*"
    /// </code>
    ///
    /// <para><b>Why this class exists.</b> <c>XC-76</c>, 2026-08-17: sixty-plus classes in this project and
    /// **not one loaded a real GGUF and measured end-to-end decode**. <c>DecodeGemvRooflineBenchmark</c> is a
    /// synthetic kernel roofline; <c>Q4KPrefillProjectionBenchmark</c> is one projection kernel and *quotes*
    /// a llama.cpp prefill figure in a comment without producing it; <c>ChatDetokenizeBenchmark</c> is the
    /// tokenizer. Every decode figure this project has published came from
    /// <c>Tests/LanguageModels/Loading/BielikSpeedTests.cs</c> — a <c>[LongFact]</c> that <c>dotnet test</c>
    /// never runs, single-arm, no canary, printing tok/s to test output. That is the shape <c>PB-12</c>
    /// condemned when it found three of four published decode-pool figures unsupported, and the structural
    /// cause it identified was exactly this: a number nothing in <c>Sources/Benchmark</c> can reproduce gets
    /// re-cited long after the code under it has moved.</para>
    ///
    /// <para><b>The units are tokens per second and this table does not print them.</b> Divide: <c>pp512</c>
    /// is <c>512 / mean</c> and <c>tg128</c> is <c>128 / mean</c>. No <c>WorkAmount</c> is declared, and that
    /// is deliberate rather than an omission — the TFLOP/s column would need a logical MAC count for the
    /// model, and <c>2 x n_params</c> is wrong here because the 311 M embedding parameters are a lookup and
    /// not a multiply. A plausible FLOP count over a kernel that does not perform those FLOPs is the exact
    /// error <see cref="WorkAmount"/> was written to make impossible.</para>
    ///
    /// <para><b>For the comparison against llama.cpp, use <c>Scripts/gguf_bench.py</c>, not this table.</b>
    /// A cross-engine ratio has to be measured by one instrument on both sides; that script wall-clocks this
    /// process and llama-bench's identically and uses neither one's internal timer. This class is the
    /// in-repository record of what the engine does, measured by BenchmarkDotNet's statistics.</para>
    ///
    /// <para><b>Fixture.</b> Qwen2.5-3B-Instruct Q4_K_M, resolved by
    /// <see cref="BenchmarkModelPaths.ResolveQwen3BQ4KM"/>, which throws with the probed paths rather than
    /// producing an <c>NA</c> row. A missing fixture fails this class and leaves the rest of a run intact.</para>
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class GgufEndToEndThroughputBenchmark
    {
        /// <summary>llama-bench's own default prompt length, so <c>pp512</c> means the same on both sides.</summary>
        private const int PromptTokens = 512;

        /// <summary>llama-bench's own default generation length, so <c>tg128</c> means the same on both sides.</summary>
        private const int GenerateTokens = 128;

        private GgufThroughputProbe _prefill = null!;
        private GgufThroughputProbe _decode = null!;

        /// <summary>
        /// Two probes rather than one, because the KV-cache size is part of the measurement: llama-bench
        /// sizes the context to the test it is running, so a prefill probe holding 640 positions and a
        /// decode probe holding 130 are two different memory footprints and must not share one session.
        /// </summary>
        [GlobalSetup]
        public void Setup()
        {
            var model = BenchmarkModelPaths.ResolveQwen3BQ4KM();

            _prefill = new GgufThroughputProbe(model, PromptTokens, generateTokens: 0);
            _decode = new GgufThroughputProbe(model, promptTokens: 1, generateTokens: GenerateTokens);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _prefill.Dispose();
            _decode.Dispose();
        }

        /// <summary>
        /// Clears both caches between invocations. <c>BenchmarkConfig</c> pins
        /// <c>InvocationCount = 1</c>, so this runs exactly once per timed body — the same place
        /// llama-bench clears its KV cache, which is outside its clock.
        /// </summary>
        [IterationSetup]
        public void ResetSessions()
        {
            _prefill.ResetForPrefill();
            _decode.ResetForDecode();
        }

        /// <summary>512 prompt tokens in one call; <c>512 / mean</c> is llama-bench's <c>pp512</c>.</summary>
        [Benchmark(Baseline = true, Description = "pp512 — 512-token prompt, one call")]
        public void Prefill512()
        {
            _prefill.RunPrefill();
        }

        /// <summary>128 generated tokens, one forward pass each; <c>128 / mean</c> is llama-bench's <c>tg128</c>.</summary>
        [Benchmark(Description = "tg128 — 128 generated tokens, one pass each")]
        public int Decode128()
        {
            return _decode.RunDecode();
        }
    }
}
