// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// Drives <see cref="AssemblyComparer"/> with inputs whose answer is known by construction.
    ///
    /// <para><b>No fixture, on purpose.</b> Every input is compiled here, at test time. A test that reads a
    /// package out of the NuGet cache or a DLL off <c>C:\</c> passes on the dev box and skips forever on
    /// Linux CI, and a test that never runs is worse than no test because it looks like coverage.</para>
    /// </summary>
    public sealed class AssemblyComparerTests
    {
        private const string TwoMethods = """
            namespace Sample
            {
                public static class Ops
                {
                    public static int A() { return B() + 1; }

                    public static int B() { return 41; }
                }
            }
            """;

        [Fact]
        public void SameSourceCompiledTwice_DiffersOnlyInBuildStamps()
        {
            // The Tensors case, and the most important test in the file: two builds of identical source ARE
            // different files, and the comparator has to say so and still answer "no logic changed".
            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(TwoMethods),
                TinyAssemblyCompiler.Compile(TwoMethods));

            Assert.True(comparison.IlIdentical, comparison.Report());
            Assert.True(comparison.PublicApiIdentical, comparison.Report());
            Assert.Equal(ChangeLevel.Inert, comparison.HighestLevel);
            Assert.Empty(comparison.Warnings);

            // Capability check: without a real inert difference this would pass on two identical files and
            // prove nothing about seeing through build stamps.
            Assert.Contains(comparison.InertDifferences, difference => difference.Kind == "Mvid");
        }

        [Fact]
        public void SameImageComparedToItself_IsFullyIdentical()
        {
            // The anchor for the whole suite: a comparator that reports differences everywhere fails here, and
            // one that reports nothing anywhere passes here and fails every test below.
            var image = TinyAssemblyCompiler.Compile(TwoMethods);
            var comparison = AssemblyComparer.CompareImages(image, image);

            Assert.Equal(ChangeLevel.None, comparison.HighestLevel);
            Assert.Empty(comparison.MethodDifferences);
            Assert.Empty(comparison.ApiDifferences);
            Assert.Empty(comparison.InertDifferences);
            Assert.Empty(comparison.Warnings);
        }

        [Fact]
        public void OneStatementChanged_ReportsExactlyThatMethod()
        {
            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(TwoMethods),
                TinyAssemblyCompiler.Compile(TwoMethods.Replace("return 41;", "return 42;")));

            var changed = Assert.Single(comparison.ChangedMethods);

            Assert.Contains("Sample.Ops.B(", changed, StringComparison.Ordinal);
            Assert.DoesNotContain("Sample.Ops.A(", changed, StringComparison.Ordinal);
            Assert.Equal(ChangeLevel.InternalOnly, comparison.HighestLevel);
            Assert.True(comparison.PublicApiIdentical, comparison.Report());
        }

        [Fact]
        public void StringLiteralChanged_IsDetected()
        {
            // Guards a specific way the normaliser could be wrong: a #US token is an offset into a heap, and
            // replacing it with a placeholder instead of the literal would make every string change invisible.
            const string source = """
                namespace Sample
                {
                    public static class Ops
                    {
                        public static string S() { return "alpha"; }
                    }
                }
                """;

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source),
                TinyAssemblyCompiler.Compile(source.Replace("alpha", "beta")));

            var changed = Assert.Single(comparison.ChangedMethods);

            Assert.Contains("Sample.Ops.S(", changed, StringComparison.Ordinal);
        }

        [Fact]
        public void MethodAdded_LeavesUntouchedMethodBytesEqual_EvenThoughRawIlMoved()
        {
            // The token-shift case, and the one a naive byte comparison fails. Z is inserted BEFORE B, which
            // moves B's MethodDef row, which changes the four operand bytes of the `call` inside A — a method
            // whose source was not touched.
            var left = TinyAssemblyCompiler.Compile(TwoMethods);
            var right = TinyAssemblyCompiler.Compile(
                TwoMethods.Replace(
                    "public static int B()",
                    "public static int Z() { return 7; }\n\n        public static int B()"));

            // Capability check for this test's premise: if the raw bytes of A had NOT moved, the test would
            // pass without exercising normalisation at all.
            Assert.NotEqual(RawIl(left, "A"), RawIl(right, "A"));

            var comparison = AssemblyComparer.CompareImages(left, right);

            Assert.True(comparison.SharedMethodBodiesIdentical, comparison.Report());
            Assert.Empty(comparison.ChangedMethods);

            var method = Assert.Single(comparison.MethodDifferences);

            Assert.Equal(DifferenceKind.Added, method.Kind);
            Assert.Contains("Sample.Ops.Z(", method.Method, StringComparison.Ordinal);

            var api = Assert.Single(comparison.ApiDifferences);

            Assert.Equal(DifferenceKind.Added, api.Kind);
            Assert.Equal("Z", api.Right.Name);
            Assert.Equal(ChangeLevel.Additive, comparison.HighestLevel);
        }

        [Fact]
        public void OnlyInformationalVersionChanged_IsReportedAsInert()
        {
            const string source = """
                [assembly: System.Reflection.AssemblyInformationalVersion("10.0.10+f7d90799")]

                namespace Sample
                {
                    public static class Ops
                    {
                        public static int A() { return 1; }
                    }
                }
                """;

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source),
                TinyAssemblyCompiler.Compile(source.Replace("10.0.10+f7d90799", "10.0.11+e2f47b01")));

            Assert.Equal(ChangeLevel.Inert, comparison.HighestLevel);

            var difference = Assert.Single(
                comparison.InertDifferences,
                candidate => candidate.Kind == "AssemblyInformationalVersion");

            Assert.Equal("10.0.10+f7d90799", difference.Left);
            Assert.Equal("10.0.11+e2f47b01", difference.Right);
        }

        [Fact]
        public void PrivateMemberChanged_MovesIlAndLeavesPublicSurfaceAlone()
        {
            // The distinction between "must retest" and "must tell consumers".
            const string source = """
                namespace Sample
                {
                    public static class Ops
                    {
                        public static int A() { return Helper() + 1; }

                        private static int Helper() { return 41; }
                    }
                }
                """;

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source),
                TinyAssemblyCompiler.Compile(source.Replace("return 41;", "return 42;")));

            Assert.False(comparison.IlIdentical);
            Assert.True(comparison.PublicApiIdentical, comparison.Report());
            Assert.Equal(ChangeLevel.InternalOnly, comparison.HighestLevel);

            var changed = Assert.Single(comparison.ChangedMethods);

            Assert.Contains("Sample.Ops.Helper(", changed, StringComparison.Ordinal);
        }

        [Fact]
        public void ParameterReordered_IsOneChangedMember_NotAnAdditionAndARemoval()
        {
            const string source = """
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void M(int a, string b) { }
                    }
                }
                """;

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source),
                TinyAssemblyCompiler.Compile(source.Replace("int a, string b", "string b, int a")));

            var difference = Assert.Single(comparison.ApiDifferences);

            Assert.Equal(DifferenceKind.Changed, difference.Kind);
            Assert.Contains("System.Int32, System.String", difference.Left.Signature, StringComparison.Ordinal);
            Assert.Contains("System.String, System.Int32", difference.Right.Signature, StringComparison.Ordinal);
        }

        [Fact]
        public void ParameterRetyped_IsOneChangedMember()
        {
            const string source = """
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void M(int a) { }
                    }
                }
                """;

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source),
                TinyAssemblyCompiler.Compile(source.Replace("int a", "long a")));

            var difference = Assert.Single(comparison.ApiDifferences);

            Assert.Equal(DifferenceKind.Changed, difference.Kind);
            Assert.Contains("System.Int32", difference.Left.Signature, StringComparison.Ordinal);
            Assert.Contains("System.Int64", difference.Right.Signature, StringComparison.Ordinal);
        }

        [Fact]
        public void ParameterRenamed_IsASurfaceChangeWithIdenticalIl()
        {
            // Binary-compatible and source-breaking for named arguments. A comparison that looked only at IL
            // would call this identical, which is the wrong advice to give a package consumer.
            const string source = """
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void M(int a) { }
                    }
                }
                """;

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source),
                TinyAssemblyCompiler.Compile(source.Replace("int a", "int value")));

            Assert.True(comparison.IlIdentical, comparison.Report());
            Assert.Equal(ChangeLevel.SourceBreaking, comparison.HighestLevel);

            var difference = Assert.Single(comparison.ApiDifferences);

            Assert.Equal(DifferenceKind.Changed, difference.Kind);
            Assert.Equal("a", difference.Left.ParameterNames);
            Assert.Equal("value", difference.Right.ParameterNames);
        }

        [Fact]
        public void DefaultValueChanged_IsASurfaceChangeWithIdenticalIl()
        {
            // A default lives in the Constant table and is baked into every CALLER, so recompiling this
            // assembly does not fix the ones already built. Identical IL here is exactly the trap.
            const string source = """
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void M(int a = 1) { }
                    }
                }
                """;

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source),
                TinyAssemblyCompiler.Compile(source.Replace("int a = 1", "int a = 2")));

            Assert.True(comparison.IlIdentical, comparison.Report());
            Assert.Equal(ChangeLevel.SilentBehaviourChange, comparison.HighestLevel);

            var difference = Assert.Single(comparison.ApiDifferences);

            Assert.Equal(DifferenceKind.Changed, difference.Kind);
            Assert.Contains("a = 1", difference.Left.Defaults, StringComparison.Ordinal);
            Assert.Contains("a = 2", difference.Right.Defaults, StringComparison.Ordinal);
        }

        [Fact]
        public void PublicMemberRemoved_IsReportedAsARemoval()
        {
            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(TwoMethods),
                TinyAssemblyCompiler.Compile("""
                    namespace Sample
                    {
                        public static class Ops
                        {
                            public static int A() { return 42; }
                        }
                    }
                    """));

            var difference = Assert.Single(comparison.ApiDifferences);

            Assert.Equal(DifferenceKind.Removed, difference.Kind);
            Assert.Equal("B", difference.Left.Name);
        }

        [Fact]
        public void PrivateTypeIsNotPartOfTheSurface()
        {
            // An internal type's whole existence is an implementation detail; reporting it would drown the
            // answer that matters in changes no consumer can observe.
            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(TwoMethods),
                TinyAssemblyCompiler.Compile(TwoMethods + """

                    namespace Sample
                    {
                        internal static class Hidden
                        {
                            public static int C() { return 3; }
                        }
                    }
                    """));

            Assert.True(comparison.PublicApiIdentical, comparison.Report());
            Assert.Equal(ChangeLevel.InternalOnly, comparison.HighestLevel);
            Assert.Contains(
                comparison.MethodDifferences,
                difference => difference.Method.Contains("Hidden.C(", StringComparison.Ordinal));
        }

        [Fact]
        public void RoslynOutputCarriesNoPrecompiledNativeCode()
        {
            // Pins the flag that carries this tool's largest limitation. If it ever reads true for plain IL
            // output the warning becomes noise and gets ignored — which is how the limit stops being visible.
            using var facts = AssemblyFacts.FromImage(TinyAssemblyCompiler.Compile(TwoMethods), "tiny");

            Assert.False(facts.HasPrecompiledNativeCode);
            Assert.True(facts.IsIlOnly);
            Assert.Empty(facts.UnresolvedTokenKinds);
            Assert.True(facts.TotalIlByteCount > 0);
        }

        [Fact]
        public void NonAssemblyInput_FailsInsteadOfComparingEqual()
        {
            var garbage = new byte[512];

            Assert.ThrowsAny<Exception>(() => AssemblyFacts.FromImage(garbage, "garbage"));
        }

        /// <summary>The raw, un-normalised IL of one method — used to prove a token-shift case really shifted.</summary>
        private static byte[] RawIl(byte[] image, string methodName)
        {
            using var peReader = new PEReader(ImmutableArray.Create(image));

            var reader = peReader.GetMetadataReader();

            // BOUND: one iteration per row of the MethodDef table.
            foreach (var handle in reader.MethodDefinitions)
            {
                var definition = reader.GetMethodDefinition(handle);

                if (reader.GetString(definition.Name) != methodName)
                {
                    continue;
                }

                return peReader.GetMethodBody(definition.RelativeVirtualAddress).GetILBytes();
            }

            throw new InvalidOperationException("no method named '" + methodName + "' in the image");
        }
    }
}
