// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Loading;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Loads the Phi-3.5-mini-instruct Q4_K_M GGUF and asserts coherent English output. Phi-3 is the first arch
    /// with FUSED tensors — one <c>attn_qkv.weight</c> (Q/K/V packed) and one <c>ffn_up.weight</c> (gate+up packed,
    /// 2×dFF) — plus NEOX RoPE with "longrope" per-dimension frequency factors. Without the fused split + longrope,
    /// decode is garbage, so a coherent answer here is the end-to-end correctness gate for the Phi-3 loader.
    /// [LongFact] — needs C:\phi. Flip to [Fact] to run.
    /// </summary>
    public sealed class Phi3SmokeTests
    {
        private const string Path = @"C:\phi\Phi-3.5-mini-instruct-Q4_K_M.gguf";
        private readonly ITestOutputHelper _out;

        public Phi3SmokeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Phi3_Loads_And_Generates_Coherent_English()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing Phi-3.5-mini gguf");
                return;
            }

            using (var reader = new GgufReader(Path))
            {
                _out.WriteLine("architecture : " + reader.GetMeta("general.architecture", "?"));
                _out.WriteLine("embed_len    : " + reader.GetMeta("phi3.embedding_length", -1));
                _out.WriteLine("rope_dim     : " + reader.GetMeta("phi3.rope.dimension_count", -1));
                _out.WriteLine("orig_ctx     : " + reader.GetMeta("phi3.rope.scaling.original_context_length", -1));
                _out.WriteLine("has fused qkv: " + reader.Tensors.ContainsKey("blk.0.attn_qkv.weight"));
            }

            using var client = OverfitClient.LoadGguf(Path);
            client.AddSystem("You are a concise, helpful assistant. Answer in one short sentence.");
            var reply = client.Send("What is the capital of France? Answer with just the city name.");
            var stats = client.Chat.LastStats;

            _out.WriteLine("REPLY: " + reply);
            _out.WriteLine($"tok/s {stats.TokensPerSecond:F1} | prompt {stats.PromptTokens} | gen {stats.GeneratedTokens} | alloc {stats.AllocatedBytes} B");

            Assert.False(string.IsNullOrWhiteSpace(reply));
            Assert.True(stats.GeneratedTokens > 0);
            Assert.Contains("Paris", reply, StringComparison.OrdinalIgnoreCase);
        }
    }
}
