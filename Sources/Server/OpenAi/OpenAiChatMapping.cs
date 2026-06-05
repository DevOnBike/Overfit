// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// Pure, host-agnostic mapping from the OpenAI chat-completions wire format onto the Overfit runtime —
    /// sampling options, the <c>response_format</c> → decode-time constraint, and conversation replay. Shared by
    /// the <c>overfit serve</c> self-host and the ASP.NET demo so the two never drift in how they honour the API.
    /// </summary>
    public static class OpenAiChatMapping
    {
        /// <summary>Maps OpenAI temperature / top_p / max_tokens onto Overfit sampling. temperature ≈ 0 → greedy.</summary>
        public static (SamplingOptions Sampling, int MaxTokens) BuildSampling(ChatCompletionRequest req)
        {
            var maxTokens = req.MaxTokens ?? req.MaxCompletionTokens ?? 512;
            if (maxTokens <= 0)
            {
                maxTokens = 512;
            }

            var temperature = req.Temperature ?? 1.0f;
            var sampling = temperature <= 0.0001f
                ? SamplingOptions.Greedy
                : new SamplingOptions(SamplingStrategy.TopP, temperature, topK: 0, topP: req.TopP ?? 1.0f, seed: 0);

            return (sampling, maxTokens);
        }

        /// <summary>
        /// Maps OpenAI <c>response_format</c> to a decode-time constraint. <c>json_object</c> → guaranteed
        /// well-formed JSON; <c>json_schema</c> → output constrained to conform to
        /// <c>response_format.json_schema.schema</c>; <c>text</c> / absent / unknown → unconstrained (null).
        /// Throws <see cref="JsonException"/> when <c>json_schema</c> is requested without a schema object.
        /// </summary>
        public static ITokenConstraint? BuildResponseFormatConstraint(JsonElement? responseFormat, ITokenizer tokenizer)
        {
            if (responseFormat is not { ValueKind: JsonValueKind.Object } format)
            {
                return null;
            }
            if (!format.TryGetProperty("type", out var typeEl) || typeEl.ValueKind != JsonValueKind.String)
            {
                return null;
            }

            var type = typeEl.GetString();
            if (string.Equals(type, "json_object", StringComparison.Ordinal))
            {
                return new JsonGrammarConstraint(tokenizer, requireObject: true);
            }
            if (string.Equals(type, "json_schema", StringComparison.Ordinal))
            {
                if (format.TryGetProperty("json_schema", out var js) && js.ValueKind == JsonValueKind.Object
                    && js.TryGetProperty("schema", out var schema) && schema.ValueKind == JsonValueKind.Object)
                {
                    return new JsonSchemaConstraint(tokenizer, schema.GetRawText());
                }
                throw new JsonException("response_format.json_schema.schema (a JSON-Schema object) is required.");
            }
            return null;   // "text" or unknown → unconstrained
        }

        /// <summary>
        /// Replays the request's <c>messages[]</c> (all but the final user turn) onto a fresh conversation so
        /// the chat is STATELESS per request — the full history travels in every call, OpenAI-style. The final
        /// (user) message is returned for the caller to generate against.
        /// </summary>
        public static void ReplayHistory(ChatSession chat, List<OpenAiMessage> messages)
        {
            chat.ResetConversation();

            for (var i = 0; i < messages.Count - 1; i++)
            {
                var content = messages[i].Content ?? string.Empty;
                switch ((messages[i].Role ?? "user").ToLowerInvariant())
                {
                    case "system":
                        chat.AddSystem(content);
                        break;
                    case "assistant":
                        chat.AddAssistant(content);
                        break;
                    default:
                        chat.AddUser(content);   // user / tool / unknown → user turn
                        break;
                }
            }
        }

        /// <summary>Parses an OpenAI embeddings <c>input</c> (a string or an array of strings) into a list.</summary>
        public static List<string> ParseInputs(JsonElement input)
        {
            var list = new List<string>();
            if (input.ValueKind == JsonValueKind.String)
            {
                list.Add(input.GetString() ?? string.Empty);
            }
            else if (input.ValueKind == JsonValueKind.Array)
            {
                foreach (var e in input.EnumerateArray())
                {
                    if (e.ValueKind == JsonValueKind.String)
                    {
                        list.Add(e.GetString() ?? string.Empty);
                    }
                }
            }
            return list;
        }
    }
}
