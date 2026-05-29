// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Demo.LocalAgent.Tools;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Tools;

namespace DevOnBike.Overfit.Demo.LocalAgent.Agent
{
    /// <summary>
    /// Turns the chat model into an actor. Two constrained-decoding flows:
    ///
    /// <list type="bullet">
    /// <item><b>Tool calling</b> (<see cref="RunToolCall"/>): a <see cref="ToolCallConstraint"/> forces
    /// the model to emit exactly one valid <c>{"name": ..., "arguments": {...}}</c> envelope choosing a
    /// registered C# tool. The reply always parses; the chosen tool is dispatched to its C# handler and
    /// the result is returned. No prompt-and-pray, no regex, no retry loop.</item>
    /// <item><b>Guaranteed JSON</b> (<see cref="RunJson"/>): a <see cref="JsonGrammarConstraint"/> forces
    /// the output to be well-formed JSON by construction — an invalid token cannot be sampled.</item>
    /// </list>
    ///
    /// Both run in-process against the local model; nothing leaves the process.
    /// </summary>
    public sealed class AgentService
    {
        private readonly ToolRegistry _tools;
        private readonly ILogger<AgentService> _logger;

        public AgentService(ToolRegistry tools, ILogger<AgentService> logger)
        {
            _tools = tools;
            _logger = logger;
        }

        /// <summary>
        /// Forces the model to choose and call exactly one registered tool for <paramref name="message"/>,
        /// then dispatches that call to its C# handler. Always returns a parsed call + the handler result.
        /// </summary>
        public AgentResult RunToolCall(OverfitClient client, string message)
        {
            // ToolCallConstraint masks the tokenizer's EndOfTextTokenId until the envelope closes, which
            // is exactly OverfitClient's chat stop token (LoadGguf wires the chat stop to the tokenizer's
            // EOS). Any chat stop *sequence* token (e.g. Qwen's <|im_end|>) is also masked mid-envelope
            // because its text can't advance the JSON. So the constraint alone keeps the model going until
            // the call is complete — no extra masking needed for this client. (The full multi-step loop
            // with a heterogeneous stop token is handled by the library's public ReActAgent.)
            var constraint = new ToolCallConstraint(_tools.Definitions, client.Tokenizer);

            // Stateless: tool routing is a one-shot decision, so it neither inherits prior chat turns
            // nor leaves its turn behind (keeps it fast and unbiased regardless of conversation state).
            var reply = client.Complete(BuildToolPrompt(message), constraint: constraint);

            if (!ToolCall.TryParse(reply, out var call))
            {
                // The constraint makes this practically unreachable; surface it loudly if it ever happens.
                throw new InvalidOperationException(
                    $"Constrained reply did not parse as a tool call. Reply: '{reply}'.");
            }

            _logger.LogInformation("Tool call: {Tool} {Args}", call.Name, call.Arguments);
            var result = _tools.Dispatch(call);

            return new AgentResult(call.Name, call.Arguments, result);
        }

        /// <summary>
        /// Generates a reply to <paramref name="message"/> constrained to well-formed JSON. The caller
        /// describes the desired shape in the message; the constraint guarantees the output parses as
        /// JSON (field-level schema typing is the JSON-Schema follow-on).
        /// </summary>
        public JsonModeResult RunJson(OverfitClient client, string message)
        {
            var constraint = new JsonGrammarConstraint(client.Tokenizer);
            // Stateless one-shot — guaranteed-JSON generation shouldn't accumulate conversation state.
            var reply = client.Complete(message, constraint: constraint);
            return new JsonModeResult(reply);
        }

        private string BuildToolPrompt(string message)
        {
            var sb = new StringBuilder();
            sb.AppendLine("Pick the ONE tool whose purpose matches the request, then supply its arguments:");
            foreach (var tool in _tools.Definitions)
            {
                sb.Append("- ").Append(tool.Name).Append(": ").AppendLine(tool.Description);
            }
            sb.AppendLine();
            sb.Append("Request: ").AppendLine(message);
            return sb.ToString();
        }
    }

}
