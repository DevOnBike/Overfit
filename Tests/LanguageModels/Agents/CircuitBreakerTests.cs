// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Agents;

namespace DevOnBike.Overfit.Tests.LanguageModels.Agents
{
    /// <summary>
    /// <see cref="CircuitBreaker"/>: a tiny capped-loop utility that the agentic primitives stack on
    /// top of. Verifies the three exit reasons (Accepted / MaxIterations / Timeout) trigger
    /// correctly, the last value is preserved, and the iteration count reflects what actually ran.
    /// </summary>
    public sealed class CircuitBreakerTests
    {
        [Fact]
        public void Accepted_OnFirstSatisfyingIteration_ExitsImmediately()
        {
            var seen = 0;
            var result = CircuitBreaker.Run(
                maxIterations: 10,
                maxElapsed: null,
                iterate: i => { seen = i; return i; },
                isAccepted: v => v >= 3);

            Assert.Equal(CircuitBreakerOutcome.Accepted, result.Outcome);
            Assert.Equal(3, result.LastValue);
            Assert.Equal(4, result.Iterations); // 0,1,2,3 → 4 iterations
            Assert.Equal(3, seen);
        }

        [Fact]
        public void MaxIterations_WhenNeverSatisfied_ReportsLastValue()
        {
            var result = CircuitBreaker.Run(
                maxIterations: 5,
                maxElapsed: null,
                iterate: i => i * 2,
                isAccepted: _ => false);

            Assert.Equal(CircuitBreakerOutcome.MaxIterations, result.Outcome);
            Assert.Equal(8, result.LastValue);   // last iteration was i=4, 4*2 = 8
            Assert.Equal(5, result.Iterations);
        }

        [Fact]
        public void Timeout_FiresWhenWallTimeExceeded()
        {
            var result = CircuitBreaker.Run(
                maxIterations: 1000,
                maxElapsed: TimeSpan.FromMilliseconds(100),
                iterate: i => { Thread.Sleep(40); return i; },
                isAccepted: _ => false);

            Assert.Equal(CircuitBreakerOutcome.Timeout, result.Outcome);
            Assert.True(result.Iterations >= 2 && result.Iterations < 1000,
                $"expected to bail mid-loop, ran {result.Iterations} iterations");
        }

        [Fact]
        public void Ctor_RejectsNonPositiveMaxIterations()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => CircuitBreaker.Run(
                maxIterations: 0,
                maxElapsed: null,
                iterate: _ => 0,
                isAccepted: _ => true));
        }
    }
}
