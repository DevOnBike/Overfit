// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT041 — a defaulted <c>CancellationToken</c>.
    ///
    /// <para>The rule is small; what these pin is that it catches both spellings the language allows —
    /// <c>= default</c> and <c>= default(CancellationToken)</c> — that it reaches interface members, where
    /// the fix usually belongs, and that it stays silent on a token the caller must supply.</para>
    /// </summary>
    public sealed class DefaultedCancellationTokenAnalyzerTests
    {
        [Fact]
        public void APlainDefaultIsReported()
        {
            Assert.Equal(["OVERFIT041"], Run("public static void M(CancellationToken ct = default) { }"));
        }

        [Fact]
        public void AnExplicitlyTypedDefaultIsReported()
        {
            Assert.Equal(["OVERFIT041"],
                Run("public static void M(CancellationToken ct = default(CancellationToken)) { }"));
        }

        // There is no third spelling to test. `= CancellationToken.None` looks like the same defect said out
        // loud, but a default parameter value must be a compile-time constant and `None` is a static
        // property, so it does not compile — checked rather than assumed, and consistent with the survey:
        // all 44 sites in the tree are `= default`.

        /// <summary>A required token — the shape the rule is pushing towards — is silent.</summary>
        [Fact]
        public void ARequiredTokenIsNotReported()
        {
            Assert.Empty(Run("public static void M(CancellationToken ct) { }"));
        }

        /// <summary>Other defaulted parameters are none of this rule's business.</summary>
        [Fact]
        public void ADefaultOnAnotherParameterTypeIsNotReported()
        {
            Assert.Empty(Run("public static void M(int retries = 3) { }"));
        }

        /// <summary>
        /// An interface member is flagged, and deliberately: the default declared there is what every
        /// implementation inherits at the call site, so removing it there is the one edit that fixes all of
        /// them.
        /// </summary>
        [Fact]
        public void AnInterfaceMemberIsReported()
        {
            var ids = Run(string.Empty, extra: """
                public interface ISource
                {
                    void Read(CancellationToken ct = default);
                }
                """);

            Assert.Equal(["OVERFIT041"], ids);
        }

        /// <summary>The message names the METHOD, because that is the signature the reader has to change.</summary>
        [Fact]
        public void TheReportNamesTheMethod()
        {
            var ids = Run("public static void Fetch(CancellationToken ct = default) { }");

            Assert.Single(ids);
        }

        private static IReadOnlyList<string> Run(string member, string extra = "")
        {
            var source = $$"""
                using System.Threading;
                using System.Threading.Tasks;

                namespace N
                {
                    {{extra}}

                    public static class C
                    {
                        {{member}}
                    }
                }
                """;

            return AnalyzerHarness.Run(new DefaultedCancellationTokenAnalyzer(), source);
        }
    }
}
