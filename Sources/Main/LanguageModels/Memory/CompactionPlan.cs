// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.LanguageModels.Chat;

namespace DevOnBike.Overfit.LanguageModels.Memory
{
    /// <summary>
    /// What <see cref="ChatHistoryCompactor.Plan"/> decided to do with a conversation: which system
    /// messages stay verbatim, which non-system turns to fold into a summary, and which recent
    /// non-system turns to keep verbatim alongside the summary.
    /// </summary>
    public sealed class CompactionPlan
    {
        public CompactionPlan(
            IReadOnlyList<ChatMessage> systemMessages,
            IReadOnlyList<ChatMessage> toSummarize,
            IReadOnlyList<ChatMessage> recentToKeep)
        {
            SystemMessages = systemMessages;
            ToSummarize = toSummarize;
            RecentToKeep = recentToKeep;
        }

        public IReadOnlyList<ChatMessage> SystemMessages
        {
            get;
        }

        public IReadOnlyList<ChatMessage> ToSummarize
        {
            get;
        }

        public IReadOnlyList<ChatMessage> RecentToKeep
        {
            get;
        }

        /// <summary>True when the plan actually reduces history (there is something to summarize).</summary>
        public bool HasWork => ToSummarize.Count > 0;
    }
}
