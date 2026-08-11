// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT042 — the index-from-end operator.
    ///
    /// <para>Two discriminations carry this rule and both are tested: <c>^</c> as an index must be caught,
    /// and <c>^</c> as XOR must not be. They share a character, and the benchmark checksums use the second
    /// one — a rule that confused them would fire on correct arithmetic.</para>
    ///
    /// <para>Ranges are the third: 251 sites here use <c>x[1..]</c> and they are deliberately untouched.</para>
    /// </summary>
    public sealed class IndexFromEndAnalyzerTests
    {
        [Fact]
        public void AnIndexFromTheEndIsReported()
        {
            Assert.Equal(["OVERFIT042"], Run("var last = data[^1];"));
        }

        [Fact]
        public void AnIndexFromTheEndInAnAssignmentTargetIsReported()
        {
            Assert.Equal(["OVERFIT042"], Run("data[^1] = 0;"));
        }

        /// <summary>A from-end bound inside a range is still the operator, and still reported.</summary>
        [Fact]
        public void AFromEndBoundInsideARangeIsReported()
        {
            Assert.Equal(["OVERFIT042"], Run("var tail = data.AsSpan()[^3..];"));
        }

        /// <summary><b>The discrimination that matters.</b> XOR shares the character and is a different node.</summary>
        [Fact]
        public void ExclusiveOrIsNotReported()
        {
            Assert.Empty(Run("var x = data[0] ^ data[1];"));
        }

        [Fact]
        public void ExclusiveOrAssignmentIsNotReported()
        {
            Assert.Empty(Run("var x = 0; x ^= data[data.Length - 1];"));
        }

        /// <summary>Ranges are the idiom this codebase is written in and are deliberately untouched.</summary>
        [Fact]
        public void AnOrdinaryRangeIsNotReported()
        {
            Assert.Empty(Run("var head = data.AsSpan()[..2];"));
        }

        [Fact]
        public void AnOpenEndedRangeIsNotReported()
        {
            Assert.Empty(Run("var tail = data.AsSpan()[1..];"));
        }

        /// <summary>The explicit arithmetic the rule pushes towards is silent, or it would flag its own fix.</summary>
        [Fact]
        public void TheExplicitArithmeticIsNotReported()
        {
            Assert.Empty(Run("var last = data[data.Length - 1];"));
        }

        private static IReadOnlyList<string> Run(string body)
        {
            var source = $$"""
                using System;

                namespace N
                {
                    public static class C
                    {
                        public static void M(int[] data)
                        {
                            {{body}}
                        }
                    }
                }
                """;

            return AnalyzerHarness.Run(new IndexFromEndAnalyzer(), source);
        }
    }
}
