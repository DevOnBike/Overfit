// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Decode-throughput benchmark for Overfit on Qwen2.5-3B Q4_K_M — the SAME GGUF dotLLM is measured on,
    /// so the two engines' decode tok/s are directly comparable. Reports tok/s and whether the repacked
    /// 8×8 GEMV path is active (toggle via the OVERFIT_REPACK_GEMV env var across runs).
    /// [LongFact] — needs C:\qwen3b\qwen.q4km.gguf. Flip to [Fact] to run.
    /// </summary>
    public sealed class QwenDecodeSpeedTests
    {
        private const string Model = @"C:\qwen3b\qwen.q4km.gguf";
        private const string Prompt =
            "Write a detailed, step-by-step explanation of how a modern CPU executes one instruction, " +
            "covering fetch, decode, execute, memory access, and writeback, with examples.";

        private readonly ITestOutputHelper _out;
        public QwenDecodeSpeedTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Qwen3B_Q4KM_DecodeSpeed()
        {
            if (!File.Exists(Model)) { _out.WriteLine($"missing {Model}"); return; }

            using var client = OverfitClient.LoadGguf(Model, maxContextLength: 2048, mmap: true, maxNewTokens: 128);

            // Warm-up turn (JIT, page-ins, KV warm) — excluded from the measurement.
            client.Send("Hi.");
            client.Reset();

            var best = 0.0;
            var lastGen = 0;
            for (var run = 0; run < 3; run++) // best-of-3
            {
                client.Reset();
                client.Send(Prompt);
                var s = client.Chat.LastStats;
                if (s.TokensPerSecond > best) { best = s.TokensPerSecond; }
                lastGen = s.GeneratedTokens;
            }

            _out.WriteLine($"repacked 8x8 GEMV (OVERFIT_REPACK_GEMV): {Q4KGemvKernel.Enabled}");
            _out.WriteLine($"Overfit Qwen-3B Q4_K_M decode: {best:F2} tok/s (best-of-3, {lastGen} tokens/run)");
            Assert.True(best > 0);
        }
    }
}
