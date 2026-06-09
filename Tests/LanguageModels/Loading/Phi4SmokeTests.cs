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
    /// Loads the Phi-4 (14B) Q4_K_M GGUF and asserts coherent English output. Phi-4 reuses the <c>phi3</c>
    /// architecture (fused <c>attn_qkv.weight</c> + fused <c>ffn_up.weight</c>, NEOX split-half RoPE), so the
    /// existing Phi-3 loader path covers it with NO new arch code. The two differences vs Phi-3.5 are non-breaking:
    /// (1) no <c>rope_factors_short/long</c> tensors — native 16k context, plain RoPE (the longrope read is guarded
    /// on tensor presence); (2) a tiktoken/cl100k byte-level BPE tokenizer (vocab ~100k, <c>tokenizer.ggml.model
    /// == "gpt2"</c>) instead of Phi-3's Llama SPM — already handled by the default cl100k split pattern. A coherent
    /// answer here is the end-to-end gate validating that thesis on a real 14B reasoning model.
    /// [LongFact] — needs C:\phi\phi-4-Q4_K_M.gguf. Flip to [Fact] to run.
    /// </summary>
    public sealed class Phi4SmokeTests
    {
        private const string Path = @"C:\phi\phi-4-Q4_K_M.gguf";
        private readonly ITestOutputHelper _out;

        public Phi4SmokeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Phi4_Loads_And_Generates_Coherent_English()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing Phi-4 gguf (expected C:\\phi\\phi-4-Q4_K_M.gguf)");
                return;
            }

            using (var reader = new GgufReader(Path))
            {
                _out.WriteLine("architecture : " + reader.GetMeta("general.architecture", "?"));
                _out.WriteLine("embed_len    : " + reader.GetMeta("phi3.embedding_length", -1));
                _out.WriteLine("block_count  : " + reader.GetMeta("phi3.block_count", -1));
                _out.WriteLine("context_len  : " + reader.GetMeta("phi3.context_length", -1));
                _out.WriteLine("tokenizer    : " + reader.GetMeta("tokenizer.ggml.model", "?"));
                _out.WriteLine("tokenizer.pre: " + reader.GetMeta("tokenizer.ggml.pre", "default"));
                _out.WriteLine("has fused qkv: " + reader.Tensors.ContainsKey("blk.0.attn_qkv.weight"));
                _out.WriteLine("has longrope : " + reader.Tensors.ContainsKey("rope_factors_short.weight"));
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
