// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Agents;

namespace DevOnBike.Overfit.Tests.LanguageModels.Agents
{
    /// <summary>
    /// <see cref="CriticLoop"/>: synthetic generator + critic delegates verify the generate→critique→
    /// revise→repeat loop. Feedback must reach the generator on revision; approval must exit; caps
    /// must surface correctly. Real-model behaviour is the host's concern.
    /// </summary>
    public sealed class CriticLoopTests
    {
        [Fact]
        public void Approved_FirstCandidate_ExitsImmediatelyWithOneTraceEntry()
        {
            var loop = new CriticLoop(
                generate: _ => "good",
                critique: _ => CriticVerdict.Approve(),
                maxIterations: 3);

            var result = loop.Run("write something");
            Assert.True(result.Approved);
            Assert.Equal(CircuitBreakerOutcome.Accepted, result.Outcome);
            Assert.Equal("good", result.FinalCandidate);
            Assert.Single(result.Trace);
        }

        [Fact]
        public void Rejected_FeedbackReachesGeneratorOnNextRound()
        {
            var seenInputs = new List<string>();
            var loop = new CriticLoop(
                generate: input => { seenInputs.Add(input); return seenInputs.Count == 1 ? "v1" : "v2"; },
                critique: c => c == "v2" ? CriticVerdict.Approve() : CriticVerdict.Reject("be more specific"),
                maxIterations: 3);

            var result = loop.Run("draft something");
            Assert.True(result.Approved);
            Assert.Equal("v2", result.FinalCandidate);
            Assert.Equal(2, result.Trace.Count);
            // Round-2 input must reference the previous attempt and the critic's feedback.
            Assert.Contains("Previous attempt:", seenInputs[1]);
            Assert.Contains("v1", seenInputs[1]);
            Assert.Contains("be more specific", seenInputs[1]);
        }

        [Fact]
        public void MaxIterations_ReachedWhenCriticNeverApproves()
        {
            var loop = new CriticLoop(
                generate: _ => "candidate",
                critique: _ => CriticVerdict.Reject("nope"),
                maxIterations: 4);

            var result = loop.Run("input");
            Assert.False(result.Approved);
            Assert.Equal(CircuitBreakerOutcome.MaxIterations, result.Outcome);
            Assert.Equal(4, result.Trace.Count);
            Assert.Equal("candidate", result.FinalCandidate);
        }
    }
}
