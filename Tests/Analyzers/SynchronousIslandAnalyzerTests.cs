// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT040 — a synchronous method calling APIs that have asynchronous siblings.
    ///
    /// <para><b>The negative cases carry the rule.</b> Its first version asked only whether a method named
    /// <c>{Name}Async</c> existed and produced 155 hits across the solution, most of them nonsense:
    /// <c>JsonSerializer.Serialize</c> has a <c>SerializeAsync</c>, but one returns a string and the other
    /// writes to a stream. Requiring the sibling to take the same arguments and return a task of the same
    /// result cut it to 92. Every "not reported" test below pins one of those discriminations.</para>
    /// </summary>
    public sealed class SynchronousIslandAnalyzerTests
    {
        [Fact]
        public void ASynchronousMethodCallingASyncApiWithAnAsyncSiblingIsReported()
        {
            Assert.Equal(["OVERFIT040"], Run("public static void M(Io io) { io.Read(); }"));
        }

        /// <summary>The sibling may take an extra cancellation token — that is how the BCL spells its pairs.</summary>
        [Fact]
        public void AnExtraCancellationTokenOnTheSiblingStillCounts()
        {
            Assert.Equal(["OVERFIT040"], Run("public static void M(Io io) { io.Save(\"x\"); }"));
        }

        /// <summary>
        /// <b>The false positive that made the first version useless.</b> <c>Pack</c> and <c>PackAsync</c>
        /// share a name and nothing else — different arguments, so a different operation.
        /// </summary>
        [Fact]
        public void ASiblingWithDifferentParametersIsNotReported()
        {
            Assert.Empty(Run("public static void M(Io io) { io.Pack(1); }"));
        }

        /// <summary>A sibling returning a task of something else is likewise a different operation.</summary>
        [Fact]
        public void ASiblingWithADifferentResultIsNotReported()
        {
            Assert.Empty(Run("public static void M(Io io) { var s = io.Fetch(); }"));
        }

        /// <summary>Already asynchronous: that is CA1849's ground, and doubling up gets rules ignored.</summary>
        [Fact]
        public void AnAsyncMethodIsNotReported()
        {
            Assert.Empty(Run(
                "public static async System.Threading.Tasks.Task M(Io io) { io.Read(); await System.Threading.Tasks.Task.Yield(); }"));
        }

        /// <summary>A method that already returns a task is CA1849's too, even without the keyword.</summary>
        [Fact]
        public void ATaskReturningMethodIsNotReported()
        {
            Assert.Empty(Run(
                "public static System.Threading.Tasks.Task M(Io io) { io.Read(); return System.Threading.Tasks.Task.CompletedTask; }"));
        }

        /// <summary>
        /// Disposal is a lifetime contract, not a slow operation; <c>IAsyncDisposable</c> is a separate
        /// decision with its own rule.
        /// </summary>
        [Fact]
        public void DisposeIsNotReported()
        {
            Assert.Empty(Run("public static void M(Io io) { io.Dispose(); }"));
        }

        /// <summary>
        /// A call inside a lambda belongs to the lambda, not to the method that declares it — blaming the
        /// enclosing declaration would point the reader at code that is not the problem.
        /// </summary>
        [Fact]
        public void ACallInsideALambdaDoesNotBlameTheEnclosingMethod()
        {
            Assert.Empty(Run("public static void M(Io io) { System.Action a = () => io.Read(); a(); }"));
        }

        /// <summary>One report per method, however many offending calls it contains.</summary>
        [Fact]
        public void AMethodIsReportedOnceNotPerCall()
        {
            Assert.Equal(["OVERFIT040"], Run("public static void M(Io io) { io.Read(); io.Read(); io.Read(); }"));
        }

        /// <summary>
        /// <b>The false positive this exclusion exists for, and the one it kept missing.</b>
        /// <c>Console.Out</c> is a <c>TextWriter.Synchronized</c> wrapper whose <c>WriteLineAsync</c> does the
        /// same synchronous write and hands back a completed task. The rule reported it anyway for a day,
        /// because the invoked symbol is <c>TextWriter.WriteLine</c> and the exclusion tested the containing
        /// type — 18 sites in the service projects, 13 of them answered with a file-scoped pragma.
        /// </summary>
        [Fact]
        public void WritingToConsoleOutIsNotReported()
        {
            Assert.Empty(Run("public static void M() { Console.Out.WriteLine(\"x\"); }"));
        }

        [Fact]
        public void WritingToConsoleErrorIsNotReported()
        {
            Assert.Empty(Run("public static void M() { Console.Error.WriteLine(\"x\"); }"));
        }

        /// <summary>The reader half: <c>Console.In</c> is the synchronised <c>TextReader</c>.</summary>
        [Fact]
        public void ReadingFromConsoleInIsNotReported()
        {
            Assert.Empty(Run("public static void M() { var s = Console.In.ReadLine(); }"));
        }

        /// <summary>
        /// The receiver is resolved through the semantic model, not matched on the text <c>Console</c>, so
        /// the fully qualified spelling is the same call and gets the same answer.
        /// </summary>
        [Fact]
        public void TheFullyQualifiedConsoleOutIsAlsoNotReported()
        {
            Assert.Empty(Run("public static void M() { System.Console.Out.WriteLine(\"x\"); }"));
        }

        /// <summary>
        /// Resolving the receiver semantically also covers the spelling that has no <c>Console</c> token in
        /// it at all. Tested rather than asserted in a comment, because it is exactly the kind of claim that
        /// reads as obvious and is cheap to be wrong about.
        /// </summary>
        [Fact]
        public void ABareOutReachedThroughUsingStaticIsNotReported()
        {
            Assert.Empty(Run("public static void M() { Out.WriteLine(\"x\"); }"));
        }

        /// <summary>
        /// <b>THE REFUTING ARM — the test that makes the four above worth having.</b> An exclusion that
        /// swallowed every <c>TextWriter</c> receiver would pass all of them and would silence the real
        /// case: a writer to disk, held by a synchronous method, which is a thread occupied for as long as
        /// the write takes. `Sources/Anomalies/Monitoring/HistoricalCsvLoader.cs:117` is exactly this shape
        /// and is still reported.
        /// </summary>
        [Fact]
        public void WritingToATextWriterThatIsNotTheConsoleIsStillReported()
        {
            Assert.Equal(
                ["OVERFIT040"],
                Run("public static void M(System.IO.TextWriter writer) { writer.WriteLine(\"x\"); }"));
        }

        /// <summary>Same again through a field, which is how a file logger is usually held.</summary>
        [Fact]
        public void WritingToATextWriterFieldIsStillReported()
        {
            Assert.Equal(
                ["OVERFIT040"],
                Run("""
                    private static readonly System.IO.TextWriter Log = new System.IO.StreamWriter("app.log");

                    public static void M() { Log.WriteLine("x"); }
                    """));
        }

        /// <summary>
        /// <b>The documented limit, pinned as what it IS rather than as what would be nice.</b> A local alias
        /// of <c>Console.Out</c> is still reported: following it would need dataflow through assignments, and
        /// the case has not been measured to occur here. Written down so the next reader finds a decision
        /// instead of rediscovering a bug — if this test ever changes, the doc comment on
        /// <c>IsConsoleStreamReceiver</c> changes with it.
        /// </summary>
        [Fact]
        public void ALocalAliasOfConsoleOutIsStillReported()
        {
            Assert.Equal(
                ["OVERFIT040"],
                Run("public static void M() { var w = Console.Out; w.WriteLine(\"x\"); }"));
        }

        private static IReadOnlyList<string> Run(string method)
        {
            var source = $$"""
                using System;
                using System.Threading;
                using System.Threading.Tasks;
                using static System.Console;

                namespace N
                {
                    public sealed class Io : System.IDisposable
                    {
                        public int Read() => 0;

                        public Task<int> ReadAsync() => Task.FromResult(0);

                        public void Save(string path) { }

                        public Task SaveAsync(string path, CancellationToken ct) { return Task.CompletedTask; }

                        // Same name, different arguments — not an async version of Pack.
                        public void Pack(int n) { }

                        public Task PackAsync(System.IO.Stream target) => Task.CompletedTask;

                        // Same name and arguments, different result — also not a pair.
                        public int Fetch() => 0;

                        public Task<string> FetchAsync() => Task.FromResult("");

                        public void Dispose() { }

                        public ValueTask DisposeAsync() => default;
                    }

                    public static class C
                    {
                        {{method}}
                    }
                }
                """;

            return AnalyzerHarness.Run(new SynchronousIslandAnalyzer(), source);
        }
    }
}
