// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Tools;

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>
    /// A driver over <see cref="ChatSession"/> + <see cref="ToolCallConstraint"/> that loops
    /// <em>model → tool-call → observe → repeat</em> until the model calls the synthetic
    /// <c>finish</c> tool or a step cap is hit. Turns the shipped tool-calling primitive into actual
    /// agents — no LangGraph-style state machine, no framework, just a loop.
    ///
    /// Each assistant turn is constrained to a single JSON tool-call envelope (the <see cref="ToolCallConstraint"/>
    /// masks logits at decode time so output is always parseable). The agent auto-registers a
    /// <see cref="FinishToolName"/> tool taking <c>{"answer": "..."}</c> — calling it is how the model
    /// signals "done". User tools dispatch through the supplied handler map; their return value becomes
    /// the next user message (<c>"Observation: ..."</c>).
    ///
    /// Out of scope by design: graph runtime, multi-agent supervisor, HITL gate, parallel fan-out
    /// (build those on this primitive, not into it).
    /// </summary>
    public sealed class ReActAgent
    {
        /// <summary>Reserved name of the auto-registered exit tool. Calling it ends the loop.</summary>
        public const string FinishToolName = "finish";

        private readonly ChatSession _chat;
        private readonly ITokenizer _tokenizer;
        private readonly ToolDefinition[] _allTools;
        private readonly Dictionary<string, Func<JsonElement, string>> _handlers;
        private readonly int _maxSteps;

        public ReActAgent(
            ChatSession chat,
            ITokenizer tokenizer,
            IEnumerable<ToolDefinition> userTools,
            IReadOnlyDictionary<string, Func<JsonElement, string>> handlers,
            int maxSteps = 8)
        {
            ArgumentNullException.ThrowIfNull(chat);
            ArgumentNullException.ThrowIfNull(tokenizer);
            ArgumentNullException.ThrowIfNull(userTools);
            ArgumentNullException.ThrowIfNull(handlers);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSteps);

            _chat = chat;
            _tokenizer = tokenizer;
            _maxSteps = maxSteps;

            var combined = new List<ToolDefinition>();
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var t in userTools)
            {
                if (string.Equals(t.Name, FinishToolName, StringComparison.Ordinal))
                {
                    throw new ArgumentException(
                        $"Tool name '{FinishToolName}' is reserved by ReActAgent for the exit tool.",
                        nameof(userTools));
                }

                if (!seen.Add(t.Name))
                {
                    throw new ArgumentException($"Duplicate tool name '{t.Name}'.", nameof(userTools));
                }

                if (!handlers.ContainsKey(t.Name))
                {
                    throw new ArgumentException(
                        $"No handler supplied for tool '{t.Name}'.", nameof(handlers));
                }

                combined.Add(t);
            }

            combined.Add(new ToolDefinition(
                FinishToolName,
                "Return the final answer. Call this once you have enough information. " +
                "Example: {\"name\":\"finish\",\"arguments\":{\"answer\":\"<your final answer to the user>\"}}"));

            _allTools = combined.ToArray();
            _handlers = new Dictionary<string, Func<JsonElement, string>>(handlers, StringComparer.Ordinal);
        }

        public int MaxSteps => _maxSteps;

        /// <summary>The full tool menu the agent presents to the model (user tools + <c>finish</c>).</summary>
        public IReadOnlyList<ToolDefinition> Tools => _allTools;

        /// <summary>
        /// Runs the loop on <paramref name="userQuery"/> with the given sampling/generation options.
        /// The agent prepends a system message describing the tool menu (so the model can pick names).
        /// </summary>
        public ReActResult Run(string userQuery, in GenerationOptions options)
        {
            ArgumentNullException.ThrowIfNull(userQuery);

            _chat.AddSystem(BuildSystemPrompt(_allTools));

            var capturedOptions = options;
            var chatStopTokenId = options.StopOnEndOfTextToken ? options.EndOfTextTokenId : -1;

            ToolCall AskForToolCall(string msg)
            {
                // ToolCallConstraint masks its tokenizer's EndOfTextTokenId until the envelope closes,
                // but the chat session may stop on a DIFFERENT id (e.g. Qwen's <|im_end|> ≠ <|endoftext|>).
                // Wrap the constraint to mask the chat's stop token too while incomplete — otherwise
                // the model can halt generation mid-envelope and the reply won't parse.
                ITokenConstraint constraint = new ToolCallConstraint(_allTools, _tokenizer);
                if (chatStopTokenId >= 0 && chatStopTokenId != _tokenizer.EndOfTextTokenId)
                {
                    constraint = new ExtraMaskedTokensConstraint(constraint, chatStopTokenId);
                }

                var reply = _chat.Send(msg, in capturedOptions, constraint: constraint);
                if (!ToolCall.TryParse(reply, out var call))
                {
                    throw new OverfitRuntimeException(
                        $"ReActAgent: constrained reply did not parse as a tool call. Reply: '{reply}'.");
                }

                return call;
            }

            return RunLoop(userQuery, _handlers, _maxSteps, AskForToolCall);
        }

        /// <summary>
        /// Pure loop logic — testable without a model. Calls <paramref name="askForToolCall"/> for
        /// each step; user-tools dispatch through <paramref name="handlers"/>; exits on
        /// <see cref="FinishToolName"/> or after <paramref name="maxSteps"/> iterations.
        /// </summary>
        internal static ReActResult RunLoop(
            string userQuery,
            IReadOnlyDictionary<string, Func<JsonElement, string>> handlers,
            int maxSteps,
            Func<string, ToolCall> askForToolCall)
        {
            var steps = new List<ReActStep>();
            var nextMessage = userQuery;

            for (var i = 0; i < maxSteps; i++)
            {
                var call = askForToolCall(nextMessage);

                using var argsDoc = JsonDocument.Parse(call.Arguments);

                if (string.Equals(call.Name, FinishToolName, StringComparison.Ordinal))
                {
                    var answer = argsDoc.RootElement.TryGetProperty("answer", out var a) && a.ValueKind == JsonValueKind.String
                        ? a.GetString()!
                        : argsDoc.RootElement.GetRawText();
                    steps.Add(new ReActStep(call.Name, call.Arguments, answer, finished: true));
                    return new ReActResult(answer, steps, ReActCompletion.Finish);
                }

                if (!handlers.TryGetValue(call.Name, out var handler))
                {
                    // The constraint should make this unreachable, but if it ever happens (e.g. a
                    // user bypassing the agent), surface it back to the model as an observation so
                    // the loop can recover rather than crashing.
                    var err = $"ERROR: unknown tool '{call.Name}'.";
                    steps.Add(new ReActStep(call.Name, call.Arguments, err, finished: false));
                    nextMessage = "Observation: " + err;
                    continue;
                }

                string observation;
                try
                {
                    observation = handler(argsDoc.RootElement);
                }
                catch (Exception ex)
                {
                    observation = $"ERROR: {ex.Message}";
                }

                steps.Add(new ReActStep(call.Name, call.Arguments, observation, finished: false));
                nextMessage = "Observation: " + observation;
            }

            return new ReActResult(
                "Step cap reached before the model called the 'finish' tool.",
                steps,
                ReActCompletion.StepCap);
        }

        private static string BuildSystemPrompt(IReadOnlyList<ToolDefinition> tools)
        {
            // Small constrained models do best with the lightest possible system prompt — too much
            // structure or too many examples derails them. Match the proven minimal pattern from
            // `ToolCallingChatTests` (a working baseline on Qwen 2.5-3B Q4_K_M).
            var sb = new StringBuilder();
            sb.Append("You can call tools. Tools: ");
            for (var i = 0; i < tools.Count; i++)
            {
                if (i > 0) { sb.Append(", "); }
                sb.Append(tools[i].Name);
            }
            sb.AppendLine(".");
            sb.AppendLine("Respond with a single tool call.");
            sb.Append("When done, call ").Append(FinishToolName).AppendLine(" with {\"answer\":\"...\"}.");
            return sb.ToString();
        }
    }
}
