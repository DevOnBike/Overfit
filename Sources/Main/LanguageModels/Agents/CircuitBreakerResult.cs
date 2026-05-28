// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>Result of <c>CircuitBreaker.Run</c>: last value produced, iteration count, wall time, exit reason.</summary>
    public sealed class CircuitBreakerResult<T>
    {
        public CircuitBreakerResult(T lastValue, int iterations, TimeSpan elapsed, CircuitBreakerOutcome outcome)
        {
            LastValue = lastValue;
            Iterations = iterations;
            Elapsed = elapsed;
            Outcome = outcome;
        }

        /// <summary>The most recent value produced by the iterate delegate (default on zero-iteration paths).</summary>
        public T LastValue { get; }

        public int Iterations { get; }

        public TimeSpan Elapsed { get; }

        public CircuitBreakerOutcome Outcome { get; }
    }
}
