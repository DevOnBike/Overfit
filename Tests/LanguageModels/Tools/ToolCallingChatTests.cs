// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.LanguageModels.Tools;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tools
{
    /// <summary>
    /// End-to-end function calling on the real Qwen GGUF: with a <see cref="ToolCallConstraint"/> the
    /// reply is a guaranteed-valid tool-call envelope, which parses, names a registered tool, and
    /// dispatches to a C# delegate. The differentiating in-process-agentic-.NET capability.
    /// [LongFact] — loads the model.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class ToolCallingChatTests
    {
        private readonly ITestOutputHelper _out;
        public ToolCallingChatTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Send_WithToolConstraint_ProducesDispatchableToolCall()
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

            var tools = new[]
            {
                new ToolDefinition("get_weather", "Get the current weather for a city."),
                new ToolDefinition("get_time", "Get the current time in a timezone."),
            };

            // The C# functions the model can call.
            var dispatch = new Dictionary<string, Func<JsonElement, string>>
            {
                ["get_weather"] = args => $"weather({args})",
                ["get_time"] = args => $"time({args})",
            };

            var chat = new ChatSession(session, tok, template, ["<|im_end|>", "\n<|im_start|>"]);
            chat.AddSystem(
                "You can call tools. Tools: get_weather(city), get_time(timezone). " +
                "Respond with a single tool call.");

            var options = new GenerationOptions(
                maxNewTokens: 128,
                maxContextLength: 512,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: true,
                endOfTextTokenId: QwenTokenizer.ImEnd);

            var constraint = new ToolCallConstraint(tools, tok);
            var reply = chat.Send("What is the weather in Paris?", in options, onText: null, constraint: constraint);

            _out.WriteLine($"--- tool call ---\n{reply}");

            Assert.True(ToolCall.TryParse(reply, out var call), $"reply was not a valid tool call: {reply}");
            Assert.Contains(call.Name, new[] { "get_weather", "get_time" });

            // arguments are guaranteed well-formed JSON
            using (var argsDoc = JsonDocument.Parse(call.Arguments))
            {
                Assert.Equal(JsonValueKind.Object, argsDoc.RootElement.ValueKind);

                // Dispatch to the registered C# function — the agentic payoff.
                var result = dispatch[call.Name](argsDoc.RootElement);
                _out.WriteLine($"dispatched → {result}");
                Assert.StartsWith(call.Name == "get_weather" ? "weather(" : "time(", result);
            }
        }
    }
}
