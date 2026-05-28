// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>Why a <c>CircuitBreaker.Run</c> exited.</summary>
    public enum CircuitBreakerOutcome
    {
        /// <summary>The acceptance predicate returned true — successful exit.</summary>
        Accepted = 0,

        /// <summary>The configured max iteration count was hit without acceptance.</summary>
        MaxIterations = 1,

        /// <summary>The wall-clock budget was hit without acceptance.</summary>
        Timeout = 2
    }
}
