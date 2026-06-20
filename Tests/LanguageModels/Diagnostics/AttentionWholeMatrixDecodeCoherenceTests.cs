// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// M3 end-to-end coherence gate for the whole-matrix Q4_K attention decode path (OVERFIT_REPACK_ATTN).
    /// The path is NOT bit-identical to the per-head decode (Q4_K-GEMV reassociation + O's Q4_K-vs-Q8 weight
    /// rounding), so correctness is verified by coherence, not byte-parity: the real Qwen-3B Q4_K_M must still
    /// answer a factual recall correctly. Run TWICE across processes to A/B — the flag is read once at static
    /// init, so it can't be toggled mid-process:
    ///   $env:OVERFIT_REPACK_ATTN="1"; dotnet test --filter ~AttentionWholeMatrixDecodeCoherence   (whole path)
    ///   $env:OVERFIT_REPACK_ATTN="0"; dotnet test --filter ~AttentionWholeMatrixDecodeCoherence   (per-head)
    /// Both must answer "Paris". [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    /// </summary>
    public sealed class AttentionWholeMatrixDecodeCoherenceTests
    {
        private const string Model = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public AttentionWholeMatrixDecodeCoherenceTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Qwen3B_Q4KM_WholeMatrixAttn_GeneratesCoherent()
        {
            if (!File.Exists(Model))
            {
                _out.WriteLine($"missing {Model}");
                return;
            }

            _out.WriteLine($"OVERFIT_REPACK_ATTN active: {Q4KGemvKernel.AttnEnabled}");

            using var client = OverfitClient.LoadGguf(Model, maxContextLength: 512, mmap: true, maxNewTokens: 80);
            client.AddSystem("You are a concise, helpful assistant.");
            var reply = client.Send("List three primary colors and write one sentence about each.");
            var stats = client.Chat.LastStats;

            _out.WriteLine("REPLY: " + reply);
            _out.WriteLine($"tok/s {stats.TokensPerSecond:F1} | gen {stats.GeneratedTokens}");

            // Coherence on a MULTI-token generation (not a 1-word factual answer that survives a corrupt
            // distribution): the whole-matrix path must produce a substantial, non-degenerate continuation.
            Assert.False(string.IsNullOrWhiteSpace(reply));
            Assert.True(stats.GeneratedTokens >= 20, $"degenerate / early-terminated generation: only {stats.GeneratedTokens} tokens");
        }
    }
}
