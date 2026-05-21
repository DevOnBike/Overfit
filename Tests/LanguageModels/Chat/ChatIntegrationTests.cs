// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    /// <summary>
    /// End-to-end chat path on the real Qwen GGUF: detect the chat template from the
    /// GGUF metadata, render a multi-turn conversation with <see cref="ChatTemplate"/>,
    /// tokenize, generate, and assemble the stream through <see cref="StopSequenceDetector"/>
    /// (string stop on <c>&lt;|im_end|&gt;</c>). [LongFact] — loads the model.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class ChatIntegrationTests
    {
        private readonly ITestOutputHelper _out;
        public ChatIntegrationTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Chat_DetectTemplate_RenderMultiTurn_StreamWithStopDetector()
        {
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            TestModelPaths.Qwen3B.RequireTokenizerJsonPath();
            var modelPath = TestModelPaths.Qwen3B.Q4KmGgufPath;

            // 1. Detect the chat template straight from the GGUF metadata.
            ChatTemplate template;
            using (var reader = new GgufReader(modelPath))
            {
                var jinja = reader.GetMeta("tokenizer.chat_template", string.Empty);
                template = ChatTemplate.Detect(jinja);
                _out.WriteLine($"Detected template: {template.Format} (chat_template {jinja.Length} chars)");
            }
            Assert.Equal(ChatTemplateFormat.ChatML, template.Format); // Qwen is ChatML

            // 2. Render a multi-turn chat → prompt string → tokens.
            var prompt = template.Render(new[]
            {
                ChatMessage.System("You are a concise assistant."),
                ChatMessage.User("What is 2+2? Answer with just the number."),
            });
            _out.WriteLine($"--- prompt ---\n{prompt}");

            using var engine = GgufLlamaLoader.Load(modelPath);
            var tok = QwenTokenizer.Load(TestModelPaths.Qwen3B.Dir);
            using var session = engine.CreateSession(512);
            session.Reset(tok.Encode(prompt));

            // 3. Stream, assembling text through the string-stop detector.
            var stops = new StopSequenceDetector("<|im_end|>", "\n<|im_start|>");
            var sampling = SamplingOptions.GreedyWithPenalty(1.15f);
            var sb = new StringBuilder();

            for (var i = 0; i < 64 && !session.IsFull; i++)
            {
                var token = session.GenerateNextToken(in sampling);
                if (token == QwenTokenizer.EndOfText || token == QwenTokenizer.ImEnd)
                {
                    break;
                }
                sb.Append(stops.Append(tok.DecodeToken(token)));
                if (stops.Stopped)
                {
                    break;
                }
            }
            sb.Append(stops.Flush());

            var response = sb.ToString();
            _out.WriteLine($"--- response ---\n{response}");

            Assert.False(string.IsNullOrWhiteSpace(response), "Chat produced no text.");
            Assert.DoesNotContain("<|im_end|>", response);   // the string stop was suppressed
            Assert.DoesNotContain("<|im_start|>", response);
        }
    }
}
