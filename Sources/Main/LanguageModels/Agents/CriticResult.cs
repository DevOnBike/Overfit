// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>Result of <c>CriticLoop.Run</c>: final candidate + per-iteration trace + exit reason.</summary>
    public sealed class CriticResult
    {
        public CriticResult(string finalCandidate, IReadOnlyList<CriticIteration> trace, CircuitBreakerOutcome outcome)
        {
            FinalCandidate = finalCandidate;
            Trace = trace;
            Outcome = outcome;
        }

        /// <summary>The last candidate produced — approved on Accepted, best-effort on MaxIterations / Timeout.</summary>
        public string FinalCandidate
        {
            get;
        }

        /// <summary>True when the critic approved the final candidate.</summary>
        public bool Approved => Outcome == CircuitBreakerOutcome.Accepted;

        public IReadOnlyList<CriticIteration> Trace
        {
            get;
        }

        public CircuitBreakerOutcome Outcome
        {
            get;
        }
    }
}
