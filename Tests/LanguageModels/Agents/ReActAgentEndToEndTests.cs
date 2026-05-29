// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Agents;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.LanguageModels.Tools;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Agents
{
    /// <summary>
    /// End-to-end smoke test of <see cref="ReActAgent"/> on a real model.
    ///
    /// IMPORTANT (empirical finding 2026-05-29): Qwen2.5-3B Q4_K_M with greedy sampling under the
    /// `ToolCallConstraint` envelope is too weak to reliably close JSON strings — small constrained
    /// models fall into repetition / Unicode-drift basins inside string values. A single-shot
    /// constrained tool call works (see <c>ToolCallingChatTests</c>), but the multi-turn ReAct loop's
    /// stricter end-to-end parse requirement is below this model's reliability floor. The agent
    /// itself is correct — verified by 6 unit tests through <c>ReActAgent.RunLoop</c>. Run this
    /// [LongFact] against a bigger model (Qwen1.5-MoE A2.7B, Mixtral, Llama-3 8B+, or any 7B+
    /// instruction-tuned variant) for a green pass.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class ReActAgentEndToEndTests
    {
        private readonly ITestOutputHelper _out;
        public ReActAgentEndToEndTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Run_OnQwen_DispatchesToolThenFinishes()
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
            using var session = engine.CreateSession(1024);
            var chat = new ChatSession(session, tok, template, ["<|im_end|>", "\n<|im_start|>"]);

            var tools = new[]
            {
                new ToolDefinition("get_weather", "Get the weather for a city. Example: {\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}"),
            };

            var weatherCalled = 0;
            var handlers = new Dictionary<string, Func<JsonElement, string>>(StringComparer.Ordinal)
            {
                ["get_weather"] = args =>
                {
                    weatherCalled++;
                    var city = args.TryGetProperty("city", out var c) && c.ValueKind == JsonValueKind.String
                        ? c.GetString()
                        : "<unknown>";
                    return $"18°C and sunny in {city}";
                },
            };

            var agent = new ReActAgent(chat, tok, tools, handlers, maxSteps: 4);

            var options = new GenerationOptions(
                maxNewTokens: 256,
                maxContextLength: 1024,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: true,
                endOfTextTokenId: QwenTokenizer.ImEnd);

            var result = agent.Run(
                "Call the get_weather tool to find out the weather in Paris, then return that as the final answer.",
                in options);

            foreach (var s in result.Steps)
            {
                _out.WriteLine($"[step] {s.ToolName}  args={s.ArgumentsJson}  obs={s.Observation}  finished={s.Finished}");
            }
            _out.WriteLine($"[completion] {result.Completion}");
            _out.WriteLine($"[answer] {result.Answer}");

            Assert.True(weatherCalled >= 1, "the agent must have called get_weather at least once");
            Assert.Equal(ReActCompletion.Finish, result.Completion);
            Assert.NotEmpty(result.Answer);
            Assert.Contains(result.Steps, s => s.ToolName == "get_weather");
            Assert.Contains(result.Steps, s => s.Finished);
        }
    }
}
