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
    /// Loads the Qwen3-0.6B Q8_0 GGUF and asserts coherent English output. Qwen3 is the first arch with
    /// an explicit <c>head_dim</c> (≠ d_model/n_heads — 0.6B is hidden 1024, 16 heads, head_dim 128) and
    /// per-head QK-RMSNorm on Q/K before RoPE. Without both, decode produces garbage — so a coherent answer
    /// here is the end-to-end correctness gate for the Qwen3 loader + attention path.
    /// [LongFact] — needs C:\qwen3-06b. Flip to [Fact] to run.
    /// </summary>
    public sealed class Qwen3SmokeTests
    {
        private const string Path = @"C:\qwen3-06b\Qwen3-0.6B-Q8_0.gguf";
        private readonly ITestOutputHelper _out;
        public Qwen3SmokeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Qwen3_Loads_And_Generates_Coherent_English()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing Qwen3-0.6B gguf");
                return;
            }

            using (var reader = new GgufReader(Path))
            {
                _out.WriteLine("architecture : " + reader.GetMeta("general.architecture", "?"));
                _out.WriteLine("tokenizer    : " + reader.GetMeta("tokenizer.ggml.model", "?"));
                _out.WriteLine("head_dim     : " + reader.GetMeta("qwen3.attention.key_length", -1));
                _out.WriteLine("n_heads      : " + reader.GetMeta("qwen3.attention.head_count", -1));
                _out.WriteLine("embed_len    : " + reader.GetMeta("qwen3.embedding_length", -1));
            }

            using var client = OverfitClient.LoadGguf(Path);
            client.AddSystem("You are a concise, helpful assistant. Answer in one short sentence.");
            // Qwen3 ships with a "thinking" mode; keep this a plain factual recall the base must get right.
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
