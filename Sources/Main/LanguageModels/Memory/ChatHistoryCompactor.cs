// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.LanguageModels.Chat;

namespace DevOnBike.Overfit.LanguageModels.Memory
{
    /// <summary>
    /// Pure planning logic for memory compaction: decides which messages need to be summarized vs.
    /// kept verbatim, based on a character-count threshold and a "recent turns to preserve" count.
    /// System messages are always preserved verbatim (they encode behaviour, not state). Extracted
    /// from <c>SummarizingChatSession</c> so the decision logic is testable without a model.
    /// </summary>
    public static class ChatHistoryCompactor
    {
        /// <summary>
        /// Builds a compaction plan. Trigger: total characters across all non-system messages
        /// exceeds <paramref name="summarizeAtChars"/>. When triggered: the last
        /// <paramref name="recentTurnsToKeep"/> non-system messages stay verbatim, everything older
        /// becomes <see cref="CompactionPlan.ToSummarize"/>. When not triggered the returned plan
        /// has <c>ToSummarize</c> empty (<see cref="CompactionPlan.HasWork"/> = false).
        /// </summary>
        public static CompactionPlan Plan(
            IReadOnlyList<ChatMessage> history,
            int summarizeAtChars,
            int recentTurnsToKeep)
        {
            ArgumentNullException.ThrowIfNull(history);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(summarizeAtChars);
            ArgumentOutOfRangeException.ThrowIfNegative(recentTurnsToKeep);

            var systemMessages = new List<ChatMessage>();
            var nonSystem = new List<ChatMessage>();
            var nonSystemChars = 0;

            foreach (var m in history)
            {
                if (string.Equals(m.Role, "system", StringComparison.Ordinal))
                {
                    systemMessages.Add(m);
                }
                else
                {
                    nonSystem.Add(m);
                    nonSystemChars += m.Content.Length;
                }
            }

            // Below threshold or not enough non-system messages to compact — nothing to do.
            if (nonSystemChars <= summarizeAtChars || nonSystem.Count <= recentTurnsToKeep)
            {
                return new CompactionPlan(systemMessages, Array.Empty<ChatMessage>(), nonSystem);
            }

            var splitIndex = nonSystem.Count - recentTurnsToKeep;
            var toSummarize = new List<ChatMessage>(splitIndex);
            for (var i = 0; i < splitIndex; i++) { toSummarize.Add(nonSystem[i]); }

            var recent = new List<ChatMessage>(recentTurnsToKeep);
            for (var i = splitIndex; i < nonSystem.Count; i++) { recent.Add(nonSystem[i]); }

            return new CompactionPlan(systemMessages, toSummarize, recent);
        }

        /// <summary>
        /// Renders a list of messages as a plain transcript suitable for feeding to the model as a
        /// summarization prompt. One line per message: <c>"role: content"</c>.
        /// </summary>
        public static string RenderTranscript(IReadOnlyList<ChatMessage> messages)
        {
            ArgumentNullException.ThrowIfNull(messages);
            var sb = new System.Text.StringBuilder();
            foreach (var m in messages)
            {
                sb.Append(m.Role).Append(": ").AppendLine(m.Content);
            }

            return sb.ToString();
        }
    }
}
