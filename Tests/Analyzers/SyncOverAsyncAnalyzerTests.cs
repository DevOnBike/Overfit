// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT039 — blocking on a task.
    ///
    /// <para><b>The negative cases are the ones that decide whether this rule survives.</b> A reliability
    /// rule that fires on correct code is turned off wholesale, so the tests that matter most here are the
    /// ones asserting SILENCE: <c>SemaphoreSlim.Wait()</c>, <c>CountdownEvent.Wait()</c> and an unrelated
    /// <c>Result</c> property. <c>Runtime/OverfitParallel.cs</c> has three correct blocking waits in its
    /// decode loop and every one of them must stay quiet.</para>
    /// </summary>
    public sealed class SyncOverAsyncAnalyzerTests
    {
        [Fact]
        public void GetAwaiterGetResultOnATaskIsReported()
        {
            Assert.Equal(["OVERFIT039"], Run("Work().GetAwaiter().GetResult();"));
        }

        [Fact]
        public void GetAwaiterGetResultOnAGenericTaskIsReported()
        {
            Assert.Equal(["OVERFIT039"], Run("var n = Number().GetAwaiter().GetResult();"));
        }

        [Fact]
        public void ResultOnATaskIsReported()
        {
            Assert.Equal(["OVERFIT039"], Run("var n = Number().Result;"));
        }

        [Fact]
        public void WaitOnATaskIsReported()
        {
            Assert.Equal(["OVERFIT039"], Run("Work().Wait();"));
        }

        [Fact]
        public void ValueTaskIsCoveredToo()
        {
            Assert.Equal(["OVERFIT039"], Run("Valued().GetAwaiter().GetResult();"));
        }

        /// <summary>
        /// <b>The false positive that would kill the rule.</b> A semaphore's <c>Wait</c> is not a task —
        /// blocking is what it is for, and there is no continuation to starve. Three such calls in
        /// <c>OverfitParallel</c>'s decode loop are correct and must stay silent. (<c>CountdownEvent</c> and
        /// <c>ManualResetEventSlim</c> are the same case; they are not exercised here because the analyzer
        /// harness's reference set does not carry them, and the gate they would test is the same one — the
        /// receiver's type, not the method's name.)
        /// </summary>
        [Fact]
        public void SemaphoreWaitIsNotReported()
        {
            Assert.Empty(Run("gate.Wait();"));
        }

        /// <summary>An unrelated <c>Result</c> property is common and must not be flagged.</summary>
        [Fact]
        public void AnUnrelatedResultPropertyIsNotReported()
        {
            Assert.Empty(Run("var v = outcome.Result;"));
        }

        /// <summary>Awaiting is the correct shape and stays silent, or the rule would flag its own fix.</summary>
        [Fact]
        public void AwaitingIsNotReported()
        {
            Assert.Empty(Run("var n = await Number();", isAsync: true));
        }

        /// <summary>
        /// <c>GetResult()</c> on something that is not an awaiter — the method name alone must not be
        /// enough, or every fluent API with a GetResult would be flagged.
        /// </summary>
        [Fact]
        public void AnUnrelatedGetResultIsNotReported()
        {
            Assert.Empty(Run("var v = outcome.GetAwaiter().GetResult();"));
        }

        private static IReadOnlyList<string> Run(string body, bool isAsync = false)
        {
            var signature = isAsync ? "public static async System.Threading.Tasks.Task M(" : "public static void M(";

            var source = $$"""
                using System.Threading;
                using System.Threading.Tasks;

                namespace N
                {
                    public sealed class Outcome
                    {
                        public int Result => 1;

                        public Outcome GetAwaiter() => this;

                        public int GetResult() => 1;
                    }

                    public static class C
                    {
                        {{signature}}SemaphoreSlim gate, Outcome outcome)
                        {
                            {{body}}
                        }

                        private static Task Work() => Task.CompletedTask;

                        private static Task<int> Number() => Task.FromResult(1);

                        private static ValueTask Valued() => default;
                    }
                }
                """;

            return AnalyzerHarness.Run(new SyncOverAsyncAnalyzer(), source);
        }
    }
}
