// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// This file is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// The three syntax-preference rules the maintainer asked for: OVERFIT043 (ranges), OVERFIT044 (null
    /// patterns) and OVERFIT045 (primary constructors).
    ///
    /// <para>These encode a preference, not a measurement, so what the tests can check is that each rule
    /// catches what was asked for and — the part that matters — nothing adjacent. A style rule that fires
    /// one line wider than intended is turned off wholesale, and then the rules that carry real defects go
    /// with it.</para>
    /// </summary>
    public sealed class SyntaxPreferenceAnalyzerTests
    {
        // ── OVERFIT043: ranges ──────────────────────────────────────────────────

        [Theory]
        [InlineData("var t = data.AsSpan()[1..];")]
        [InlineData("var t = data.AsSpan()[..2];")]
        [InlineData("var t = data.AsSpan()[1..3];")]
        public void ARangeIsReported(string body)
        {
            Assert.Equal(["OVERFIT043"], Run(new RangeExpressionAnalyzer(), body));
        }

        [Fact]
        public void AnExplicitSliceIsNotReported()
        {
            Assert.Empty(Run(new RangeExpressionAnalyzer(), "var t = data.AsSpan().Slice(1, 2);"));
        }

        /// <summary>An ordinary index is not a range, or the rule would swallow every array read.</summary>
        [Fact]
        public void AnOrdinaryIndexIsNotReported()
        {
            Assert.Empty(Run(new RangeExpressionAnalyzer(), "var t = data[1];"));
        }

        // ── OVERFIT044: null patterns ───────────────────────────────────────────

        [Fact]
        public void IsNullIsReported()
        {
            Assert.Equal(["OVERFIT044"], Run(new NullPatternAnalyzer(), "var b = text is null;"));
        }

        [Fact]
        public void IsNotNullIsReported()
        {
            Assert.Equal(["OVERFIT044"], Run(new NullPatternAnalyzer(), "var b = text is not null;"));
        }

        [Fact]
        public void TheComparisonTheRulePushesTowardsIsNotReported()
        {
            Assert.Empty(Run(new NullPatternAnalyzer(), "var b = text == null; var c = text != null;"));
        }

        /// <summary>
        /// <b>Other patterns are untouched.</b> The objection was to one spelling of a null check, not to
        /// pattern matching — and this codebase leans on type and property patterns heavily.
        /// </summary>
        [Theory]
        [InlineData("var b = text is string s;")]
        [InlineData("var b = data is { Length: 0 };")]
        [InlineData("var b = data.Length is > 0;")]
        [InlineData("var b = text is not string;")]
        public void OtherPatternsAreNotReported(string body)
        {
            Assert.Empty(Run(new NullPatternAnalyzer(), body));
        }

        // ── OVERFIT045: primary constructors ────────────────────────────────────

        [Fact]
        public void APrimaryConstructorOnAClassIsReported()
        {
            Assert.Equal(["OVERFIT045"], RunOnType(new PrimaryConstructorAnalyzer(),
                "public sealed class Holder(int value) { public int Value => value; }"));
        }

        [Fact]
        public void APrimaryConstructorOnAStructIsReported()
        {
            Assert.Equal(["OVERFIT045"], RunOnType(new PrimaryConstructorAnalyzer(),
                "public struct Point(int x) { public int X => x; }"));
        }

        /// <summary>
        /// <b>Positional records are exempt and this pins it.</b> There the parameter list declares the
        /// members and the equality contract; 106 files here rely on it.
        /// </summary>
        [Fact]
        public void APositionalRecordIsNotReported()
        {
            Assert.Empty(RunOnType(new PrimaryConstructorAnalyzer(), "public record Point(int X, int Y);"));
        }

        [Fact]
        public void APositionalRecordStructIsNotReported()
        {
            Assert.Empty(RunOnType(new PrimaryConstructorAnalyzer(),
                "public readonly record struct Point(int X, int Y);"));
        }

        [Fact]
        public void AnOrdinaryConstructorIsNotReported()
        {
            Assert.Empty(RunOnType(new PrimaryConstructorAnalyzer(),
                "public sealed class Holder { private readonly int _value; public Holder(int value) => _value = value; }"));
        }

        private static IReadOnlyList<string> Run(DiagnosticAnalyzer analyzer, string body)
        {
            var source = $$"""
                using System;

                namespace N
                {
                    public static class C
                    {
                        public static void M(int[] data, string text)
                        {
                            {{body}}
                        }
                    }
                }
                """;

            return AnalyzerHarness.Run(analyzer, source);
        }

        private static IReadOnlyList<string> RunOnType(DiagnosticAnalyzer analyzer, string type)
        {
            var source = $$"""
                using System;

                namespace N
                {
                    {{type}}
                }
                """;

            return AnalyzerHarness.Run(analyzer, source);
        }
    }
}
