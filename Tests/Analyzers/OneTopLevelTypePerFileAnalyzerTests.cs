// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT034 — one namespace-level type per file, which replaced an MSBuild task on 2026-08-05.
    ///
    /// <para><b>The test that justifies the move is
    /// <see cref="AFileScopedNamespaceIsStillChecked"/>.</b> The regular expression it replaced anchored on
    /// <c>^    </c> — exactly four spaces — because every file in the tree used block-scoped namespaces at
    /// the time. Under a file-scoped namespace the types are not indented at all, so the rule matched
    /// nothing and reported nothing, for every such file, silently. A guard that stops applying without
    /// saying so is worse than no guard, because the directory still looks covered.</para>
    /// </summary>
    public sealed class OneTopLevelTypePerFileAnalyzerTests
    {
        private static IReadOnlyList<string> Run(string source)
        {
            return AnalyzerHarness.Run(new OneTopLevelTypePerFileAnalyzer(), source);
        }

        [Fact]
        public void OneTypeIsFine()
        {
            var ids = Run("""
                namespace N
                {
                    public class Only
                    {
                    }
                }
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void TwoTopLevelTypesAreReported()
        {
            var ids = Run("""
                namespace N
                {
                    public class First
                    {
                    }

                    public enum Second
                    {
                        Value,
                    }
                }
                """);

            Assert.Equal(["OVERFIT034"], ids);
        }

        /// <summary>
        /// The hole the regex had, and the reason this is an analyzer. Zero indentation meant zero matches,
        /// so the rule was off for any file written this way and nothing said so.
        /// </summary>
        [Fact]
        public void AFileScopedNamespaceIsStillChecked()
        {
            var ids = Run("""
                namespace N;

                public class First
                {
                }

                public record Second(int Value);
                """);

            Assert.Equal(["OVERFIT034"], ids);
        }

        /// <summary>Nesting is the recommended alternative, so it must not be what gets flagged.</summary>
        [Fact]
        public void NestedTypesAreFine()
        {
            var ids = Run("""
                namespace N
                {
                    public class Outer
                    {
                        public sealed class Inner
                        {
                        }

                        public enum Kind
                        {
                            One,
                        }
                    }
                }
                """);

            Assert.Empty(ids);
        }

        /// <summary>
        /// Partial declarations of the SAME type are one type. This is why the rule counts distinct names
        /// rather than declarations — `ComputationGraph` alone is split across a dozen files here.
        /// </summary>
        [Fact]
        public void PartialsOfTheSameTypeAreOneType()
        {
            var ids = Run("""
                namespace N
                {
                    public partial class Split
                    {
                        public int A => 1;
                    }

                    public partial class Split
                    {
                        public int B => 2;
                    }
                }
                """);

            Assert.Empty(ids);
        }

        /// <summary>
        /// One diagnostic for the file, not one per extra type. Five types in a file is one decision to
        /// revisit, and five squiggles telling the same story is how a rule gets suppressed wholesale.
        /// </summary>
        [Fact]
        public void ManyTypesProduceExactlyOneDiagnostic()
        {
            var ids = Run("""
                namespace N
                {
                    public class A
                    {
                    }

                    public class B
                    {
                    }

                    public class C
                    {
                    }

                    public interface ID
                    {
                    }
                }
                """);

            Assert.Single(ids);
        }

        /// <summary>A delegate is a type declaration too, and it is the one people forget.</summary>
        [Fact]
        public void ATopLevelDelegateCountsAsAType()
        {
            var ids = Run("""
                namespace N
                {
                    public class Holder
                    {
                    }

                    public delegate void Callback(int value);
                }
                """);

            Assert.Equal(["OVERFIT034"], ids);
        }
    }
}
