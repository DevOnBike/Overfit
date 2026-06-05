// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.LanguageModels.Chat
{
    /// <summary>
    /// Renders a multi-turn chat into a model's prompt string. GGUF files ship the real
    /// template as a Jinja string under <c>tokenizer.chat_template</c>; running Jinja is
    /// Native-AOT-hostile (it needs an interpreter), so instead <see cref="Detect"/>
    /// fingerprints that string and picks one of a few hand-written formatters covering
    /// the formats in wide use (ChatML, Llama-3, Mistral). The rendered text is then fed
    /// to the tokenizer like any prompt.
    /// </summary>
    public sealed class ChatTemplate
    {
        public ChatTemplate(ChatTemplateFormat format)
        {
            Format = format;
        }

        public ChatTemplateFormat Format { get; }

        /// <summary>
        /// Picks a formatter by fingerprinting a GGUF <c>tokenizer.chat_template</c> Jinja
        /// string. Falls back to <see cref="ChatTemplateFormat.ChatML"/> (the most common)
        /// when the string is missing or unrecognised.
        /// </summary>
        public static ChatTemplate Detect(string? jinjaTemplate)
        {
            if (!string.IsNullOrEmpty(jinjaTemplate))
            {
                if (jinjaTemplate.Contains("<|im_start|>", StringComparison.Ordinal))
                {
                    return new ChatTemplate(ChatTemplateFormat.ChatML);
                }
                if (jinjaTemplate.Contains("<|start_header_id|>", StringComparison.Ordinal))
                {
                    return new ChatTemplate(ChatTemplateFormat.Llama3);
                }
                if (jinjaTemplate.Contains("[INST]", StringComparison.Ordinal))
                {
                    return new ChatTemplate(ChatTemplateFormat.Mistral);
                }
            }
            return new ChatTemplate(ChatTemplateFormat.ChatML);
        }

        /// <summary>
        /// Renders <paramref name="messages"/> to a prompt string. When
        /// <paramref name="addGenerationPrompt"/> is true (default) the assistant turn is
        /// opened so the model continues as the assistant.
        /// </summary>
        public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
        {
            if (messages is null) { throw new ArgumentNullException(nameof(messages)); }

            var sb = new StringBuilder();
            switch (Format)
            {
                case ChatTemplateFormat.ChatML:
                    RenderChatML(sb, messages, addGenerationPrompt);
                    break;
                case ChatTemplateFormat.Llama3:
                    RenderLlama3(sb, messages, addGenerationPrompt);
                    break;
                case ChatTemplateFormat.Mistral:
                    RenderMistral(sb, messages, addGenerationPrompt);
                    break;
                default:
                    throw new OverfitRuntimeException($"Unsupported chat template format {Format}.");
            }
            return sb.ToString();
        }

        private static void RenderChatML(StringBuilder sb, IReadOnlyList<ChatMessage> messages, bool gen)
        {
            for (var i = 0; i < messages.Count; i++)
            {
                var m = messages[i];
                sb.Append("<|im_start|>").Append(m.Role).Append('\n')
                  .Append(m.Content).Append("<|im_end|>\n");
            }
            if (gen)
            {
                sb.Append("<|im_start|>assistant\n");
            }
        }

        private static void RenderLlama3(StringBuilder sb, IReadOnlyList<ChatMessage> messages, bool gen)
        {
            sb.Append("<|begin_of_text|>");
            for (var i = 0; i < messages.Count; i++)
            {
                var m = messages[i];
                sb.Append("<|start_header_id|>").Append(m.Role).Append("<|end_header_id|>\n\n")
                  .Append(m.Content).Append("<|eot_id|>");
            }
            if (gen)
            {
                sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }

        // Mistral / Llama-2 [INST] style. Mistral has no native system role; the system
        // message is folded into the first user turn. BOS/EOS are left to the tokenizer.
        private static void RenderMistral(StringBuilder sb, IReadOnlyList<ChatMessage> messages, bool gen)
        {
            string? pendingSystem = null;
            for (var i = 0; i < messages.Count; i++)
            {
                var m = messages[i];
                if (m.Role == "system")
                {
                    pendingSystem = pendingSystem is null ? m.Content : pendingSystem + "\n\n" + m.Content;
                }
                else if (m.Role == "user")
                {
                    var content = pendingSystem is null ? m.Content : pendingSystem + "\n\n" + m.Content;
                    pendingSystem = null;
                    sb.Append("[INST] ").Append(content).Append(" [/INST]");
                }
                else // assistant
                {
                    sb.Append(' ').Append(m.Content).Append("</s>");
                }
            }
            // For Mistral the open "[INST] … [/INST]" already prompts the assistant reply,
            // so addGenerationPrompt needs no extra marker.
            _ = gen;
        }
    }
}
