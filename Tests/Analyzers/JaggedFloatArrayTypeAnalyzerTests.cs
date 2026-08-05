// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT033 — the ban on the <c>float[][]</c> type, which replaced an MSBuild task on 2026-08-05.
    ///
    /// <para><b>Two of these tests exist because the regular expression it replaced got them wrong</b>, and
    /// they are the reason the move was worth making rather than tidying. The old task stripped only
    /// <c>//</c> comments, so <c>float[][]</c> inside a <c>/* … */</c> comment was reported as a violation
    /// — an error about code that does not exist. And it matched text, so it saw only the two-bracket
    /// spelling and missed the type arriving through a generic argument.</para>
    ///
    /// <para>The rest pin the boundary: what is banned is <c>float</c> rows specifically, not jagged arrays
    /// in general — <c>int[][]</c> and <c>Parameter[][]</c> stay legal and are OVERFIT002's business as a
    /// per-call allocation concern.</para>
    /// </summary>
    public sealed class JaggedFloatArrayTypeAnalyzerTests
    {
        private static IReadOnlyList<string> Run(string source)
        {
            return AnalyzerHarness.Run(new JaggedFloatArrayTypeAnalyzer(), source);
        }

        [Fact]
        public void AFieldOfJaggedFloatIsReported()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        private float[][] _rows = null!;
                    }
                }
                """);

            Assert.Equal(["OVERFIT033"], ids);
        }

        [Fact]
        public void AParameterAndAReturnTypeAreReported()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        public float[][] Make(float[][] input) => input;
                    }
                }
                """);

            Assert.Equal(["OVERFIT033", "OVERFIT033"], ids);
        }

        /// <summary>
        /// The regression the old regex could not catch: the type arrives as a generic argument, spelled the
        /// same but nested inside another type. A text match anchored on the declaration shape misses it.
        /// </summary>
        [Fact]
        public void AGenericArgumentOfJaggedFloatIsReported()
        {
            var ids = Run("""
                using System.Collections.Generic;

                namespace N
                {
                    public class C
                    {
                        private List<float[][]> _batches = new();
                    }
                }
                """);

            Assert.Equal(["OVERFIT033"], ids);
        }

        /// <summary>
        /// <b>The false positive the old task produced.</b> It skipped lines beginning with <c>//</c> and cut
        /// everything after a <c>//</c> on a code line — but a block comment is neither, so this line was
        /// reported as a banned jagged array. An error about a comment is worse than a missed violation:
        /// it teaches people that the rule is noise.
        /// </summary>
        [Fact]
        public void JaggedFloatInsideABlockCommentIsNotReported()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        /* the old shape was float[][] and is banned */
                        private float[] _flat = new float[4]; /* was float[][] */
                    }
                }
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void AFlatFloatArrayIsNotReported()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        private float[] _flat = new float[16];
                    }
                }
                """);

            Assert.Empty(ids);
        }

        /// <summary>Only float rows are banned; other element types remain legal by design.</summary>
        [Fact]
        public void JaggedArraysOfOtherElementTypesAreNotReported()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        private int[][] _ints = null!;
                        private string[][] _text = null!;
                    }
                }
                """);

            Assert.Empty(ids);
        }

        /// <summary>
        /// A rectangular <c>float[,]</c> is a single allocation with no pointer chase, so it is not what the
        /// rule is about. Worth pinning because the two spellings look alike at a glance.
        /// </summary>
        [Fact]
        public void ARectangularFloatArrayIsNotReported()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        private float[,] _grid = new float[4, 4];
                    }
                }
                """);

            Assert.Empty(ids);
        }

        /// <summary>
        /// <c>float[][]</c> parses as an array whose element is an array, so a naive walk reports both nodes
        /// and an author sees two errors for one mistake.
        /// </summary>
        [Fact]
        public void ANestedTypeProducesExactlyOneDiagnostic()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        private float[][][] _deep = null!;
                    }
                }
                """);

            Assert.Single(ids);
        }
    }
}
