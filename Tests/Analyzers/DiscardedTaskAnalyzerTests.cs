// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT046 — a task discarded with <c>_ =</c>.
    ///
    /// <para><b>The silence tests are the ones that decide whether this rule survives, and they outnumber
    /// the detection tests here on purpose.</b> The explicit discard is a common and correct idiom in this
    /// tree for a completely unrelated reason — <c>_ = someParameter;</c> silences an unused parameter, and
    /// <c>Sources</c> holds seventeen of them. A rule that flagged those would be switched off within a day,
    /// taking the two real findings with it. <see cref="DiscardingAParameterIsNotReported"/> is the test that
    /// pins it, and removing the type gate from the analyzer must turn that test red immediately.</para>
    ///
    /// <para>The other negative worth naming is <see cref="AssigningToALocalNamedUnderscoreIsNotReported"/>.
    /// A local actually called <c>_</c> is legal C#, and assigning a task to it keeps the task reachable — so
    /// it is not a discard at all. The analyzer asks the compiler (<c>IDiscardSymbol</c>) instead of matching
    /// the identifier, and this is where that choice is checked.</para>
    /// </summary>
    public sealed class DiscardedTaskAnalyzerTests
    {
        [Fact]
        public void DiscardingATaskIsReported()
        {
            Assert.Equal(["OVERFIT046"], Run("_ = Work();"));
        }

        [Fact]
        public void DiscardingAGenericTaskIsReported()
        {
            Assert.Equal(["OVERFIT046"], Run("_ = Number();"));
        }

        /// <summary>ValueTask counts, exactly as it does for OVERFIT039 — the gate is shared.</summary>
        [Fact]
        public void DiscardingAValueTaskIsReported()
        {
            Assert.Equal(["OVERFIT046"], Run("_ = Valued();"));
        }

        [Fact]
        public void DiscardingAGenericValueTaskIsReported()
        {
            Assert.Equal(["OVERFIT046"], Run("_ = ValuedNumber();"));
        }

        /// <summary>
        /// The shape the lab drivers use — <c>Demo/LabWorkload/Program.cs</c> discards three of these at its
        /// fault-injection endpoints. Reported, and each of those sites takes a pragma naming why.
        /// </summary>
        [Fact]
        public void DiscardingATaskRunIsReported()
        {
            Assert.Equal(["OVERFIT046"], Run("_ = System.Threading.Tasks.Task.Run(() => { });"));
        }

        /// <summary>
        /// <b>The false positive that would kill the rule.</b> Seventeen sites in <c>Sources</c> discard a
        /// parameter to silence an unused-parameter warning, and every one must stay quiet. This is the
        /// mutation target: delete the awaitable check in the analyzer and this test goes red.
        /// </summary>
        [Fact]
        public void DiscardingAParameterIsNotReported()
        {
            Assert.Empty(Run("_ = count;"));
        }

        /// <summary>A discarded call that does not return a task is not this rule's business.</summary>
        [Fact]
        public void DiscardingANonTaskCallIsNotReported()
        {
            Assert.Empty(Run("_ = Text();"));
        }

        /// <summary>
        /// Awaiting first and discarding the RESULT is correct — the task was observed, and what is thrown
        /// away is an <c>int</c>. The rule would otherwise flag its own fix.
        /// </summary>
        [Fact]
        public void DiscardingTheResultOfAnAwaitIsNotReported()
        {
            Assert.Empty(Run("_ = await Number();", isAsync: true));
        }

        /// <summary>
        /// A local genuinely named <c>_</c> is not a discard: the task stays reachable and can still be
        /// awaited. Matching the identifier instead of asking for <c>IDiscardSymbol</c> would report this.
        /// </summary>
        [Fact]
        public void AssigningToALocalNamedUnderscoreIsNotReported()
        {
            Assert.Empty(Run("System.Threading.Tasks.Task _ = System.Threading.Tasks.Task.CompletedTask; _ = Work();"));
        }

        /// <summary>
        /// A bare un-awaited call is CS4014's job — an error repo-wide already. This rule covers only the
        /// discard that CS4014 does not see, and reporting both would put two diagnostics on one defect.
        /// </summary>
        [Fact]
        public void ABareUnawaitedCallIsNotReported()
        {
            Assert.Empty(Run("Work();"));
        }

        /// <summary>
        /// The shape at <c>Server.AspNet/OverfitAspNetServer.cs:75</c> —
        /// <c>cancellationToken.Register(() =&gt; _ = app.StopAsync())</c>. The discard is the body of a
        /// lambda rather than a statement in a method, and it must still be seen; that site is the one the
        /// rule expects to keep, behind a pragma carrying the sentence it already has.
        /// </summary>
        [Fact]
        public void ADiscardInsideALambdaIsReported()
        {
            Assert.Equal(["OVERFIT046"], Run("System.Action a = () => _ = Work(); a();"));
        }

        private static IReadOnlyList<string> Run(string body, bool isAsync = false)
        {
            var signature = isAsync
                ? "public static async System.Threading.Tasks.Task M("
                : "public static void M(";

            var source = $$"""
                using System.Threading.Tasks;

                namespace N
                {
                    public static class C
                    {
                        {{signature}}int count)
                        {
                            {{body}}
                        }

                        private static Task Work() => Task.CompletedTask;

                        private static Task<int> Number() => Task.FromResult(1);

                        private static ValueTask Valued() => default;

                        private static ValueTask<int> ValuedNumber() => default;

                        private static string Text() => "x";
                    }
                }
                """;

            return AnalyzerHarness.Run(new DiscardedTaskAnalyzer(), source);
        }
    }
}
