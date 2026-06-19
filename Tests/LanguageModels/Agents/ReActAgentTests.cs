// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Agents;
using DevOnBike.Overfit.LanguageModels.Tools;

namespace DevOnBike.Overfit.Tests.LanguageModels.Agents
{
    /// <summary>
    /// Unit tests for <see cref="ReActAgent"/>'s pure loop logic — exercised via the internal
    /// <c>RunLoop</c> hook that takes a delegate in place of a real model. Verifies dispatch,
    /// step-cap, error handling, and the synthetic <c>finish</c> exit. Real-model behavior is covered
    /// by a [LongFact] end-to-end test on Qwen elsewhere.
    /// </summary>
    public sealed class ReActAgentTests
    {
        private static IReadOnlyDictionary<string, Func<JsonElement, string>> Dispatch(
            params (string Name, Func<JsonElement, string> Fn)[] entries)
        {
            var d = new Dictionary<string, Func<JsonElement, string>>(StringComparer.Ordinal);
            foreach (var (n, f) in entries)
            {
                d[n] = f;
            }
            return d;
        }

        [Fact]
        public void SingleToolThenFinish_ReturnsAnswer()
        {
            var replies = new Queue<ToolCall>(new[]
            {
                new ToolCall("get_weather", "{\"city\":\"Paris\"}"),
                new ToolCall(ReActAgent.FinishToolName, "{\"answer\":\"18°C and rainy in Paris\"}"),
            });

            var handlers = Dispatch(("get_weather", args =>
                $"18°C and rainy ({args.GetProperty("city").GetString()})"));

            var result = ReActAgent.RunLoop("What's the weather in Paris?", handlers, maxSteps: 8, _ => replies.Dequeue());

            Assert.Equal(ReActCompletion.Finish, result.Completion);
            Assert.Equal("18°C and rainy in Paris", result.Answer);
            Assert.Equal(2, result.Steps.Count);
            Assert.Equal("get_weather", result.Steps[0].ToolName);
            Assert.False(result.Steps[0].Finished);
            Assert.Equal("18°C and rainy (Paris)", result.Steps[0].Observation);
            Assert.True(result.Steps[1].Finished);
        }

        [Fact]
        public void MultiStep_ChainsObservationsBackAsNextUserMessage()
        {
            var seenMessages = new List<string>();
            var replies = new Queue<ToolCall>(new[]
            {
                new ToolCall("step_a", "{}"),
                new ToolCall("step_b", "{}"),
                new ToolCall("step_c", "{}"),
                new ToolCall(ReActAgent.FinishToolName, "{\"answer\":\"done\"}"),
            });

            var handlers = Dispatch(
                ("step_a", _ => "result_a"),
                ("step_b", _ => "result_b"),
                ("step_c", _ => "result_c"));

            var result = ReActAgent.RunLoop("start", handlers, 8, msg =>
            {
                seenMessages.Add(msg);
                return replies.Dequeue();
            });

            Assert.Equal(ReActCompletion.Finish, result.Completion);
            Assert.Equal("done", result.Answer);
            Assert.Equal(4, result.Steps.Count);
            // First call sees the user query; subsequent calls see the previous observation.
            Assert.Equal(new[] { "start", "Observation: result_a", "Observation: result_b", "Observation: result_c" }, seenMessages);
        }

        [Fact]
        public void StepCap_ExitsWithExhaustionResult()
        {
            // Model never calls finish — loop must terminate at maxSteps.
            var handlers = Dispatch(("ping", _ => "pong"));
            var result = ReActAgent.RunLoop(
                "go",
                handlers,
                maxSteps: 3,
                _ => new ToolCall("ping", "{}"));

            Assert.Equal(ReActCompletion.StepCap, result.Completion);
            Assert.Equal(3, result.Steps.Count);
            Assert.Contains("Step cap", result.Answer);
        }

        [Fact]
        public void UnknownTool_SurfacesAsObservationAndContinues()
        {
            var replies = new Queue<ToolCall>(new[]
            {
                new ToolCall("not_in_handlers", "{}"),
                new ToolCall(ReActAgent.FinishToolName, "{\"answer\":\"recovered\"}"),
            });

            var handlers = Dispatch(("known_tool", _ => "ok"));
            var result = ReActAgent.RunLoop("q", handlers, 8, _ => replies.Dequeue());

            Assert.Equal(ReActCompletion.Finish, result.Completion);
            Assert.Equal("recovered", result.Answer);
            Assert.Contains("ERROR: unknown tool", result.Steps[0].Observation);
        }

        [Fact]
        public void HandlerThrows_BecomesErrorObservation()
        {
            var replies = new Queue<ToolCall>(new[]
            {
                new ToolCall("flaky", "{}"),
                new ToolCall(ReActAgent.FinishToolName, "{\"answer\":\"fine\"}"),
            });

            var handlers = Dispatch(("flaky", _ => throw new InvalidOperationException("boom")));
            var result = ReActAgent.RunLoop("q", handlers, 8, _ => replies.Dequeue());

            Assert.Equal(ReActCompletion.Finish, result.Completion);
            Assert.Contains("ERROR: boom", result.Steps[0].Observation);
        }

        [Fact]
        public void Finish_WithNonStringAnswer_FallsBackToRawJson()
        {
            var replies = new Queue<ToolCall>(new[]
            {
                new ToolCall(ReActAgent.FinishToolName, "{\"answer\":{\"value\":42}}"),
            });

            var handlers = Dispatch();
            var result = ReActAgent.RunLoop("q", handlers, 8, _ => replies.Dequeue());

            // No string `answer` field — fall back to the raw JSON object so nothing is silently lost.
            Assert.Equal(ReActCompletion.Finish, result.Completion);
            Assert.Contains("\"value\":42", result.Answer);
        }

        // ----- ctor validation (the full ReActAgent type; uses a real ChatSession proxy is not
        // needed — these throw before any chat interaction).

        // Note: the ctor needs a ChatSession and ITokenizer; we skip its direct ctor tests here
        // because constructing those without a model is heavy. Validation logic mirrors the
        // documented invariants and is exercised whenever the type is constructed in the
        // [LongFact] end-to-end test.
    }
}
