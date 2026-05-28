// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>
    /// A critic's evaluation of one candidate: approved (loop exits) or rejected with feedback that
    /// the generator sees on the next revision pass.
    /// </summary>
    public readonly struct CriticVerdict
    {
        public CriticVerdict(bool approved, string feedback)
        {
            Approved = approved;
            Feedback = feedback ?? string.Empty;
        }

        public bool Approved { get; }

        /// <summary>Free-form critique fed back to the generator on revision. Empty when approved.</summary>
        public string Feedback { get; }

        public static CriticVerdict Approve() => new(true, string.Empty);

        public static CriticVerdict Reject(string feedback) => new(false, feedback);
    }
}
