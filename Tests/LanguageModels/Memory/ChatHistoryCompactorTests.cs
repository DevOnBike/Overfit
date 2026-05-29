// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Memory;

namespace DevOnBike.Overfit.Tests.LanguageModels.Memory
{
    /// <summary>
    /// <see cref="ChatHistoryCompactor"/>: planning logic for summarising old turns. Verifies the
    /// threshold trigger, the recent-turn preservation, the system-message-always-kept rule, and the
    /// transcript renderer. No model needed.
    /// </summary>
    public sealed class ChatHistoryCompactorTests
    {
        private static ChatMessage S(string c) => ChatMessage.System(c);
        private static ChatMessage U(string c) => ChatMessage.User(c);
        private static ChatMessage A(string c) => ChatMessage.Assistant(c);

        [Fact]
        public void BelowThreshold_NoCompactionNeeded()
        {
            var history = new[] { S("sys"), U("hi"), A("hello") };
            var plan = ChatHistoryCompactor.Plan(history, summarizeAtChars: 1000, recentTurnsToKeep: 4);

            Assert.False(plan.HasWork);
            Assert.Empty(plan.ToSummarize);
            Assert.Equal(2, plan.RecentToKeep.Count);
            Assert.Single(plan.SystemMessages);
        }

        [Fact]
        public void AboveThreshold_OlderNonSystemTurnsAreScheduledForSummary()
        {
            var history = new[]
            {
                S("sys"),
                U(new string('a', 200)), A(new string('b', 200)),  // old: 400 chars total
                U(new string('c', 200)), A(new string('d', 200)),  // old: 400 chars total
                U("recent question"), A("recent answer"),
            };
            var plan = ChatHistoryCompactor.Plan(history, summarizeAtChars: 500, recentTurnsToKeep: 2);

            Assert.True(plan.HasWork);
            Assert.Equal(4, plan.ToSummarize.Count);
            Assert.Equal(new[] { "user", "assistant" }, new[] { plan.RecentToKeep[0].Role, plan.RecentToKeep[1].Role });
            Assert.Equal("recent question", plan.RecentToKeep[0].Content);
            Assert.Single(plan.SystemMessages);
        }

        [Fact]
        public void SystemMessages_StayInSystemBucketRegardlessOfPosition()
        {
            var history = new[]
            {
                S("first"),
                U(new string('x', 500)),
                S("interleaved"), // unusual but legal — interleaved system
                A(new string('y', 500)),
                U("recent"),
            };
            var plan = ChatHistoryCompactor.Plan(history, summarizeAtChars: 100, recentTurnsToKeep: 1);

            Assert.Equal(2, plan.SystemMessages.Count);
            Assert.Equal(2, plan.ToSummarize.Count); // the two old non-system messages
            Assert.Single(plan.RecentToKeep);
            Assert.Equal("recent", plan.RecentToKeep[0].Content);
        }

        [Fact]
        public void NotEnoughNonSystemMessages_NoCompactionEvenAboveThreshold()
        {
            // Only one non-system, even if it's huge — compaction needs > recentTurnsToKeep messages.
            var history = new[] { S("sys"), U(new string('a', 10_000)) };
            var plan = ChatHistoryCompactor.Plan(history, summarizeAtChars: 100, recentTurnsToKeep: 2);

            Assert.False(plan.HasWork);
        }

        [Fact]
        public void RenderTranscript_OneLinePerMessage_WithRoleLabel()
        {
            var messages = new[] { U("hi"), A("hello") };
            var transcript = ChatHistoryCompactor.RenderTranscript(messages);

            Assert.Contains("user: hi", transcript);
            Assert.Contains("assistant: hello", transcript);
        }
    }
}
