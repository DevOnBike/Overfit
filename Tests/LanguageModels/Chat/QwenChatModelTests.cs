// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    /// <summary>
    /// Full zero-Python turnkey pipeline on the real Qwen2.5-0.5B HF directory:
    /// <see cref="QwenChatModel.LoadFromDirectory"/> wires native safetensors load +
    /// Qwen tokenizer + chat template from <c>tokenizer_config.json</c> into a
    /// <see cref="ChatSession"/>, and a multi-turn exchange runs end-to-end with no
    /// intermediate conversion. [LongFact] — loads ~1 GB.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class QwenChatModelTests
    {
        private readonly ITestOutputHelper _out;
        public QwenChatModelTests(ITestOutputHelper output) => _out = output;

        private static string SafetensorsPath => Path.Combine(TestModelPaths.Qwen3B.Dir, "model.safetensors");

        [LongFact]
        public void Chat_RealQwen05B_FromDirectory_ZeroPython_Responds()
        {
            if (!File.Exists(SafetensorsPath))
            {
                _out.WriteLine("model.safetensors not present — skipping.");
                return;
            }

            using var model = QwenChatModel.LoadFromDirectory(TestModelPaths.Qwen3B.Dir, maxContextLength: 512, quantize: false);
            model.Chat.AddSystem("You are a concise assistant. Answer in one short sentence.");

            var options = new GenerationOptions(
                maxNewTokens: 32,
                maxContextLength: 512,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: true,
                endOfTextTokenId: -1);

            var streamed = new StringBuilder();
            var reply = model.Chat.Send("What is the capital of France?", in options, s => streamed.Append(s));

            _out.WriteLine($"reply: '{reply}'");

            Assert.False(string.IsNullOrWhiteSpace(reply), "Chat produced no text.");
            Assert.Equal(reply, streamed.ToString());          // stream == return
            Assert.DoesNotContain("<|im_end|>", reply);        // ChatML terminator suppressed
            Assert.Equal(3, model.Chat.History.Count);         // system + user + assistant
            Assert.Equal("assistant", model.Chat.History[^1].Role);
            Assert.Contains("Paris", reply, StringComparison.OrdinalIgnoreCase);
        }
    }
}
