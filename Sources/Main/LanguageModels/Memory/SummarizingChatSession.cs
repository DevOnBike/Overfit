// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Memory
{
    /// <summary>
    /// A <see cref="ChatSession"/> wrapper that auto-compresses long conversations to keep them
    /// within model context budget. Complements (does not replace) <c>ChatSession</c>'s sliding-window
    /// KV eviction — sliding-window drops tokens silently; this preserves their *meaning* via a
    /// model-generated summary, so the assistant doesn't "forget" key facts.
    ///
    /// Mechanism: before each <see cref="Send"/>, if the total non-system content exceeds the
    /// configured character threshold, the older turns are summarised by the same underlying model
    /// and the conversation is rebuilt as <c>[system…] + [running summary] + [recent N turns
    /// verbatim]</c>. Summary calls are isolated by save/restore-around-summarize so the user-facing
    /// conversation history is never polluted by the summarisation prompt itself.
    /// </summary>
    public sealed class SummarizingChatSession
    {
        private const string SummaryRoleLabel = "Summary so far";

        private readonly ChatSession _chat;
        private readonly int _summarizeAtChars;
        private readonly int _recentTurnsToKeep;
        private readonly string _summarizationInstruction;
        private readonly GenerationOptions _summarizationOptions;

        public SummarizingChatSession(
            ChatSession chat,
            int summarizeAtChars,
            int recentTurnsToKeep,
            GenerationOptions summarizationOptions,
            string summarizationInstruction =
                "You are a concise summariser. Compress the following conversation transcript into 2-3 sentences " +
                "capturing the key facts, decisions, and any open questions. Output only the summary.")
        {
            ArgumentNullException.ThrowIfNull(chat);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(summarizeAtChars);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(recentTurnsToKeep);
            ArgumentNullException.ThrowIfNull(summarizationInstruction);

            _chat = chat;
            _summarizeAtChars = summarizeAtChars;
            _recentTurnsToKeep = recentTurnsToKeep;
            _summarizationInstruction = summarizationInstruction;
            _summarizationOptions = summarizationOptions;
        }

        /// <summary>The underlying chat session. Use this for read-only inspection — driving it directly bypasses compaction.</summary>
        public ChatSession Inner => _chat;

        /// <summary>True after at least one compaction pass has run on this session.</summary>
        public bool Compacted { get; private set; }

        /// <summary>The number of summary regenerations performed (one per compaction pass).</summary>
        public int CompactionCount { get; private set; }

        /// <summary>
        /// Like <see cref="ChatSession.Send"/>, but first compresses old turns if the conversation
        /// has grown beyond the configured threshold.
        /// </summary>
        public string Send(
            string userMessage,
            in GenerationOptions options,
            Action<string>? onText = null,
            ITokenConstraint? constraint = null)
        {
            CompactIfNeeded();
            return _chat.Send(userMessage, in options, onText, constraint);
        }

        /// <summary>
        /// Forces a compaction pass right now (regardless of threshold). Useful from a host that wants
        /// to compact on a schedule or after a known-large turn. No-op when nothing to summarise.
        /// </summary>
        public void Compact() => RunCompaction();

        private void CompactIfNeeded()
        {
            var plan = ChatHistoryCompactor.Plan(_chat.History, _summarizeAtChars, _recentTurnsToKeep);
            if (plan.HasWork) { ApplyPlan(plan); }
        }

        private void RunCompaction()
        {
            // Force compaction with the same recent-turns budget but threshold 1 so any non-empty
            // older content triggers — useful when the host wants to compact unconditionally.
            var plan = ChatHistoryCompactor.Plan(_chat.History, summarizeAtChars: 1, _recentTurnsToKeep);
            if (plan.HasWork) { ApplyPlan(plan); }
        }

        private void ApplyPlan(CompactionPlan plan)
        {
            var summary = SummariseOldTurns(plan.ToSummarize);
            RehydrateHistory(plan.SystemMessages, summary, plan.RecentToKeep);
            Compacted = true;
            CompactionCount++;
        }

        /// <summary>
        /// Calls the underlying model to summarise <paramref name="oldTurns"/>. Saves and restores the
        /// chat history around the call so the summarisation prompt doesn't leak into the user-visible
        /// conversation. The opts use the caller's <c>summarizationOptions</c>, not the per-Send opts.
        /// </summary>
        private string SummariseOldTurns(IReadOnlyList<ChatMessage> oldTurns)
        {
            var saved = new List<ChatMessage>(_chat.History);
            _chat.ResetConversation();
            _chat.AddSystem(_summarizationInstruction);
            var transcript = ChatHistoryCompactor.RenderTranscript(oldTurns);
            var opts = _summarizationOptions;
            var summary = _chat.Send(transcript, in opts).Trim();
            RestoreHistory(saved);
            return summary;
        }

        private void RehydrateHistory(IReadOnlyList<ChatMessage> systemMessages, string summary, IReadOnlyList<ChatMessage> recentToKeep)
        {
            _chat.ResetConversation();
            foreach (var s in systemMessages) { _chat.AddSystem(s.Content); }
            _chat.AddSystem(SummaryRoleLabel + ": " + summary);
            foreach (var m in recentToKeep)
            {
                AppendRaw(m);
            }
        }

        private void RestoreHistory(IReadOnlyList<ChatMessage> saved)
        {
            _chat.ResetConversation();
            foreach (var m in saved) { AppendRaw(m); }
        }

        private void AppendRaw(ChatMessage m)
        {
            switch (m.Role)
            {
                case "system": _chat.AddSystem(m.Content); break;
                case "user": _chat.AddUser(m.Content); break;
                case "assistant": _chat.AddAssistant(m.Content); break;
                default: throw new InvalidOperationException($"Unknown chat role '{m.Role}'.");
            }
        }
    }
}
