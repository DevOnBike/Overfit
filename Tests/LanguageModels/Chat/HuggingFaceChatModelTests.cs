// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    /// <summary>
    /// Family-generic turnkey pipeline (<see cref="HuggingFaceChatModel"/>) on the real
    /// Qwen2.5-0.5B directory — proves the generic <c>HuggingFaceBpeTokenizer</c> + detected
    /// chat template drive an end-to-end multi-turn chat with no per-model hard-coding and no
    /// Python. (Same proof as <c>QwenChatModelTests</c> but through the family-generic path;
    /// dropping a Llama-3 / Mistral dir validates those families.) [LongFact] — loads ~1 GB.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class HuggingFaceChatModelTests
    {
        private readonly ITestOutputHelper _out;
        public HuggingFaceChatModelTests(ITestOutputHelper output) => _out = output;

        private static string SafetensorsPath => Path.Combine(TestModelPaths.Qwen3B.Dir, "model.safetensors");

        [LongFact]
        public void Chat_RealQwen05B_GenericPipeline_Responds()
        {
            if (!File.Exists(SafetensorsPath)) { _out.WriteLine("model.safetensors not present — skipping."); return; }

            using var model = HuggingFaceChatModel.LoadFromDirectory(TestModelPaths.Qwen3B.Dir, maxContextLength: 512, quantize: false);
            Assert.Equal(ChatTemplateFormat.ChatML, model.Format);   // Qwen detected as ChatML

            model.Chat.AddSystem("You are a concise assistant. Answer in one short sentence.");
            var options = new GenerationOptions(
                maxNewTokens: 32, maxContextLength: 512, sampling: SamplingOptions.Greedy, stopOnEndOfTextToken: true);

            var reply = model.Chat.Send("What is the capital of France?", in options);
            _out.WriteLine($"reply: '{reply}'");

            Assert.False(string.IsNullOrWhiteSpace(reply), "Chat produced no text.");
            Assert.DoesNotContain("<|im_end|>", reply);
            Assert.Contains("Paris", reply, StringComparison.OrdinalIgnoreCase);
            Assert.Equal("assistant", model.Chat.History[^1].Role);
        }
    }
}
