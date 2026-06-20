// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Chat
{
    /// <summary>One chat turn. Role is "system", "user" or "assistant".</summary>
    public readonly struct ChatMessage
    {
        public ChatMessage(string role, string content)
        {
            Role = role ?? throw new ArgumentNullException(nameof(role));
            Content = content ?? throw new ArgumentNullException(nameof(content));
        }

        public string Role
        {
            get;
        }
        public string Content
        {
            get;
        }

        public static ChatMessage System(string content) => new("system", content);
        public static ChatMessage User(string content) => new("user", content);
        public static ChatMessage Assistant(string content) => new("assistant", content);
    }
}
