// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>
    /// A tiny reliability primitive — wraps a "loop until accepted, with caps" pattern that recurs
    /// in agent-style code (ReAct loops, self-reflection / critic loops, retry-with-backoff,
    /// agentic-task drivers). Caps iteration count and (optionally) wall-clock time. Reports the
    /// exit reason so the caller can distinguish success, exhaustion, and timeout. Pure utility — no
    /// dependencies on chat / model / tokenizer types.
    /// </summary>
    public static class CircuitBreaker
    {
        /// <summary>
        /// Runs <paramref name="iterate"/> up to <paramref name="maxIterations"/> times (and within
        /// <paramref name="maxElapsed"/> wall time if supplied), returning the moment
        /// <paramref name="isAccepted"/> approves a value. The acceptance check runs AFTER each
        /// iteration so the iterate delegate observes every iteration index from 0 onward.
        /// </summary>
        public static CircuitBreakerResult<T> Run<T>(
            int maxIterations,
            TimeSpan? maxElapsed,
            Func<int, T> iterate,
            Func<T, bool> isAccepted)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxIterations);
            ArgumentNullException.ThrowIfNull(iterate);
            ArgumentNullException.ThrowIfNull(isAccepted);

            var stopwatch = Stopwatch.StartNew();
            var last = default(T)!;
            var done = 0;
            for (var i = 0; i < maxIterations; i++)
            {
                last = iterate(i);
                done = i + 1;

                if (isAccepted(last))
                {
                    return new CircuitBreakerResult<T>(last, done, stopwatch.Elapsed, CircuitBreakerOutcome.Accepted);
                }

                if (maxElapsed is { } cap && stopwatch.Elapsed > cap)
                {
                    return new CircuitBreakerResult<T>(last, done, stopwatch.Elapsed, CircuitBreakerOutcome.Timeout);
                }
            }

            return new CircuitBreakerResult<T>(last, done, stopwatch.Elapsed, CircuitBreakerOutcome.MaxIterations);
        }
    }
}
