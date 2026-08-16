// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT037 — comments must be in English.
    ///
    /// <para><b>The tests that matter here are the negative ones.</b> A rule like this fails by firing on
    /// correct code, not by missing a violation: a hand-written first pass over this repository used a word
    /// list containing <c>pod</c>, <c>to</c> and <c>test</c> and reported 154 violations where there were 4,
    /// because "pod" appears in most English comments in a Kubernetes codebase. So the cases below pin the
    /// English comments that must stay silent at least as firmly as the Polish ones that must report.</para>
    /// </summary>
    public sealed class NonEnglishCommentAnalyzerTests
    {
        private static IReadOnlyList<string> Run(string source)
        {
            return AnalyzerHarness.Run(new NonEnglishCommentAnalyzer(), source);
        }

        [Fact]
        public void AnOrdinaryEnglishCommentIsSilent()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        // Decode is memory-bandwidth-bound, so a faster dot kernel saves nothing.
                        public int Value;
                    }
                }
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void EnglishAboutKubernetesPodsIsSilent()
        {
            // The exact shape that produced 150 false positives when the word list was too wide.
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        /// <summary>One pod per metric per cycle, aligned to a shared grid, so a test can
                        /// compare each pod to its peers without a cluster.</summary>
                        public int Value;
                    }
                }
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void APolishDiacriticReports()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        // Grad storage jest osobny, bo backward zapisuje gradient w kształcie view.
                        public int Value;
                    }
                }
                """);

            Assert.Equal(["OVERFIT037"], ids);
        }

        [Fact]
        public void PolishWithoutDiacriticsStillReports()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        // to zawsze na koncu - bo zalezy od HasFma (pola sa inicjowane od gory do dolu)
                        public int Value;
                    }
                }
                """);

            Assert.Equal(["OVERFIT037"], ids);
        }

        [Fact]
        public void ADocumentationCommentIsCheckedToo()
        {
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        /// <summary>Greedy bez system message — najlepsza szansa dla malego modelu.</summary>
                        public int Value;
                    }
                }
                """);

            Assert.Equal(["OVERFIT037"], ids);
        }

        [Fact]
        public void OneAmbiguousWordIsNotEnough()
        {
            // "wynik" alone must not report — a single hit is how a heuristic becomes noise.
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        // The wynik variable name is legacy and will be renamed.
                        public int Value;
                    }
                }
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void ACommentAboutNonAsciiHandlingIsExempt()
        {
            // Real case: the BM25 tokenizer documents the Polish letters it must not destroy.
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        /// <summary>Every non-ASCII alphabet must survive tokenisation: Polish
                        /// (ą, ć, ę, ł, ń, ó, ś, ź, ż) and the rest.</summary>
                        public int Value;
                    }
                }
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void CodeIdentifiersAreNotComments()
        {
            // The rule reads comment trivia only — a Polish identifier is a different problem.
            var ids = Run("""
                namespace N
                {
                    public class C
                    {
                        public int WynikTegoTesty;
                    }
                }
                """);

            Assert.Empty(ids);
        }
    }
}
