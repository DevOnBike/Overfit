// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints
{
    /// <summary>
    /// End-to-end JSON-mode on the real Qwen GGUF: a <see cref="JsonGrammarConstraint"/> passed to
    /// <see cref="ChatSession.Send"/> must make the reply parse as JSON regardless of how the small
    /// model would otherwise have answered (no prose, no truncation). [LongFact] — loads the model.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class JsonConstrainedChatTests
    {
        private readonly ITestOutputHelper _out;
        public JsonConstrainedChatTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Send_WithJsonConstraint_ProducesParseableJson()
        {
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            TestModelPaths.Qwen3B.RequireTokenizerJsonPath();
            var modelPath = TestModelPaths.Qwen3B.Q4KmGgufPath;

            ChatTemplate template;
            using (var reader = new GgufReader(modelPath))
            {
                template = ChatTemplate.Detect(reader.GetMeta("tokenizer.chat_template", string.Empty));
            }

            using var engine = GgufLlamaLoader.Load(modelPath);
            var tok = new QwenChatTokenizer(QwenTokenizer.Load(TestModelPaths.Qwen3B.Dir));
            using var session = engine.CreateSession(512);

            var chat = new ChatSession(session, tok, template, ["<|im_end|>", "\n<|im_start|>"]);
            chat.AddSystem("You output only JSON.");

            var options = new GenerationOptions(
                maxNewTokens: 128,
                maxContextLength: 512,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: true,
                endOfTextTokenId: QwenTokenizer.ImEnd);

            var constraint = new JsonGrammarConstraint(tok);
            var reply = chat.Send(
                "Give a person object with fields name and age for Alice, who is 30.",
                in options,
                onText: null,
                constraint: constraint);

            _out.WriteLine($"--- constrained reply ---\n{reply}");

            Assert.False(string.IsNullOrWhiteSpace(reply), "Constrained generation produced no text.");

            // The whole point: it parses as JSON, no try/repair. Throws → test fails.
            using var doc = JsonDocument.Parse(reply);
            Assert.True(
                doc.RootElement.ValueKind is JsonValueKind.Object or JsonValueKind.Array
                    or JsonValueKind.String or JsonValueKind.Number
                    or JsonValueKind.True or JsonValueKind.False or JsonValueKind.Null,
                $"Unexpected root kind {doc.RootElement.ValueKind}.");
        }
    }
}
