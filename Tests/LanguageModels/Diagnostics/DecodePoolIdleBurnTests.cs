// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Idle-burn gate for the decode spin-pool (spin-then-park, 2026-06-11): after a decode finishes,
    /// the pool must PARK — a serving container at rest must not spin cores (field report: 100% CPU at
    /// idle on a 16-core laptop with the pre-park pool). Decodes a few tokens to wake the pool, then
    /// sleeps 3 s and asserts the process burns &lt; 1 effective core during the idle window (parked
    /// pool ≈ 0; the old pure-spin pool burned ~10). [LongFact] — needs C:\qwen3-06b.
    /// </summary>
    public sealed class DecodePoolIdleBurnTests
    {
        private const string Path = @"C:\qwen3-06b\Qwen3-0.6B-Q4_K_M.gguf";
        private readonly ITestOutputHelper _out;
        public DecodePoolIdleBurnTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Pool_Parks_WhenIdle()
        {
            if (!File.Exists(Path)) { _out.WriteLine("missing gguf"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            using var session = engine.CreateSession(128);
            session.Reset(tok.Encode("Hello"));
            var sampling = SamplingOptions.Greedy;
            for (var i = 0; i < 8; i++) { session.GenerateNextToken(in sampling); }   // pool is hot now

            using var process = Process.GetCurrentProcess();
            process.Refresh();
            var cpuBefore = process.TotalProcessorTime;
            var sw = Stopwatch.StartNew();
            Thread.Sleep(3000);
            sw.Stop();
            process.Refresh();
            var cpuSeconds = (process.TotalProcessorTime - cpuBefore).TotalSeconds;
            var effectiveCores = cpuSeconds / sw.Elapsed.TotalSeconds;

            _out.WriteLine($"idle window {sw.Elapsed.TotalSeconds:F1}s: CPU {cpuSeconds:F2}s = {effectiveCores:F2} effective cores");
            Assert.True(effectiveCores < 1.0,
                $"decode pool is burning CPU at idle: {effectiveCores:F2} effective cores (expected ~0 after spin-then-park)");
        }
    }
}
