// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>One pass of the critic loop: the candidate produced, and the critic's verdict on it.</summary>
    public sealed class CriticIteration
    {
        public CriticIteration(string candidate, CriticVerdict verdict)
        {
            Candidate = candidate;
            Verdict = verdict;
        }

        public string Candidate { get; }

        public CriticVerdict Verdict { get; }
    }
}
