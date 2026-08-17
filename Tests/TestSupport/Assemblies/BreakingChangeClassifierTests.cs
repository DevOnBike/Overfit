// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// One test per classification rule, on inputs whose answer is known by construction.
    ///
    /// <para><b>The expected levels are not this repository's opinion.</b> Each comes from
    /// <c>dotnet/runtime/docs/coding-guidelines/breaking-change-rules.md</c> or <i>.NET API changes that
    /// affect compatibility</i>, and the conditional rules are tested from <b>both</b> sides — sealing a type
    /// with an accessible constructor and sealing one without, adding an interface member with a default
    /// implementation and without. A classifier is only useful if it is right about the safe case too; one
    /// that reports every change as breaking is as useless as one that reports none.</para>
    ///
    /// <para>No fixture, same as <see cref="AssemblyComparerTests"/>: every input is compiled here.</para>
    /// </summary>
    public sealed class BreakingChangeClassifierTests
    {
        // ---------------------------------------------------------------- level 6: silent, no rebuild needed

        [Fact]
        public void ConstValueChanged_IsSilentBehaviourChange()
        {
            var change = Single("""
                namespace Sample
                {
                    public static class Limits
                    {
                        public const int MaxItems = 10;
                    }
                }
                """, "= 10", "= 20", ChangeLevel.SilentBehaviourChange);

            Assert.Equal("Sample.Limits.MaxItems", change.Target);
            Assert.Contains("10", change.Message, StringComparison.Ordinal);
            Assert.Contains("20", change.Message, StringComparison.Ordinal);

            // The whole reason this outranks a binary break: nothing fails and nothing has to be rebuilt.
            Assert.Contains("rebuild", change.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("already wrong", change.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void OptionalParameterDefaultChanged_IsSilentBehaviourChange_AndIsNotAParameterAddition()
        {
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static int Retry(int attempts = 3) { return attempts; }
                    }
                }
                """, "= 3", "= 5", ChangeLevel.SilentBehaviourChange);

            Assert.Equal("AC-DEFAULT-VALUE-CHANGED", change.RuleId);
            Assert.Contains("attempts", change.Message, StringComparison.Ordinal);

            // Distinct from AC-SIGNATURE-CHANGED, which is what adding a parameter reports. Same level-6
            // family, entirely different remedy.
            Assert.DoesNotContain("SIGNATURE", change.RuleId, StringComparison.Ordinal);
        }

        [Fact]
        public void EnumMemberValueChanged_IsSilentBehaviourChange()
        {
            var change = Single("""
                namespace Sample
                {
                    public enum Mode
                    {
                        Off = 0,
                        Fast = 1,
                    }
                }
                """, "Fast = 1", "Fast = 2", ChangeLevel.SilentBehaviourChange);

            Assert.Equal("Sample.Mode.Fast", change.Target);
            Assert.Contains("CP0011", change.RuleId, StringComparison.Ordinal);
            Assert.Contains("persisted", change.Message, StringComparison.OrdinalIgnoreCase);
        }

        // ------------------------------------------------------------------------------ level 5: binary break

        [Fact]
        public void InterfaceMemberAdded_IsBinaryBreaking_AndNamesTheInterfaceAndTheMember()
        {
            // The case the user named. It is NOT merely source-breaking: a type compiled against the old
            // interface fails to LOAD, because the CLR requires every interface method to be implemented.
            var change = Single("""
                namespace Sample
                {
                    public interface IStore
                    {
                        int Read();
                    }
                }
                """, "int Read();", "int Read();\n        int Write();", ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0006", change.RuleId, StringComparison.Ordinal);
            Assert.Contains("IStore", change.Message, StringComparison.Ordinal);
            Assert.Contains("Write", change.Message, StringComparison.Ordinal);
            Assert.Contains("TypeLoadException", change.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void InterfaceMemberAddedWithDefaultImplementation_IsNotBreaking()
        {
            // The cry-wolf guard. Every modern library adds default interface members; a tool that flags them
            // all as breaking gets switched off, and then it catches nothing.
            var change = Single("""
                namespace Sample
                {
                    public interface IStore
                    {
                        int Read();
                    }
                }
                """,
                "int Read();",
                "int Read();\n        int Write() { return 0; }",
                ChangeLevel.Additive);

            Assert.Equal("AC-INTERFACE-MEMBER-ADDED-WITH-DEFAULT", change.RuleId);
            Assert.Contains("ref struct", change.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void AbstractMemberAdded_IsBinaryBreaking_WhenTheTypeCanBeDerivedFrom()
        {
            var change = Single("""
                namespace Sample
                {
                    public abstract class Store
                    {
                        public abstract int Read();
                    }
                }
                """,
                "public abstract int Read();",
                "public abstract int Read();\n        public abstract int Write();",
                ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0005", change.RuleId, StringComparison.Ordinal);
        }

        [Fact]
        public void AbstractMemberAdded_IsAllowed_WhenNoConstructorIsAccessible()
        {
            // The conditional half. Nobody outside can have derived from it, so nobody outside can break.
            var change = Single("""
                namespace Sample
                {
                    public abstract class Store
                    {
                        internal Store() { }

                        public abstract int Read();
                    }
                }
                """,
                "public abstract int Read();",
                "public abstract int Read();\n        public abstract int Write();",
                ChangeLevel.Additive);

            Assert.Equal("AC-ABSTRACT-MEMBER-ADDED-SAFE", change.RuleId);
        }

        [Fact]
        public void SealingAType_IsBinaryBreaking_WhenAConstructorIsAccessible()
        {
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                    }
                }
                """, "public class Store", "public sealed class Store", ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0009", change.RuleId, StringComparison.Ordinal);
        }

        [Fact]
        public void SealingAType_IsAllowed_WhenNoConstructorIsAccessible()
        {
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                        internal Store() { }
                    }
                }
                """, "public class Store", "public sealed class Store", ChangeLevel.Additive);

            Assert.Equal("AC-TYPE-SEALED (CP0009)-SAFE", change.RuleId);
        }

        [Fact]
        public void ReturnTypeChanged_IsBinaryBreaking()
        {
            // This repository shipped exactly this shape: `void Embed(...)` became `Task EmbedAsync(...)` on a
            // public interface in a published package.
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void Embed(int a) { }
                    }
                }
                """,
                "public static void Embed(int a) { }",
                "public static int Embed(int a) { return a; }",
                ChangeLevel.BinaryBreaking);

            Assert.Equal("AC-RETURN-TYPE-CHANGED", change.RuleId);
            Assert.Contains("async", change.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void ParameterAddedWithADefault_IsBinaryBreaking_NotSourceCompatible()
        {
            // Looks harmless — every source caller still compiles — and it is still a different method in
            // metadata, so every already-compiled caller fails.
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void Send(int a) { }
                    }
                }
                """, "Send(int a)", "Send(int a, int b = 1)", ChangeLevel.BinaryBreaking);

            Assert.Equal("AC-SIGNATURE-CHANGED", change.RuleId);
            Assert.Contains("default value", change.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void MemberRemoved_IsBinaryBreaking()
        {
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static int A() { return 1; }

                        public static int B() { return 2; }
                    }
                }
                """, "public static int B() { return 2; }", string.Empty, ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0002", change.RuleId, StringComparison.Ordinal);
            Assert.Equal("Sample.Ops.B", change.Target);
        }

        [Fact]
        public void TypeRemoved_IsReportedOnce_NotOncePerMember()
        {
            // The usability rule. A removed type with twenty members must not produce twenty-one findings, or
            // a real package comparison becomes a wall nobody reads twice.
            var comparison = Compare("""
                namespace Sample
                {
                    public class Gone
                    {
                        public int A() { return 1; }

                        public int B() { return 2; }

                        public int C() { return 3; }
                    }
                }
                """, string.Empty);

            var change = Assert.Single(comparison.AtOrAbove(ChangeLevel.SourceBreaking));

            Assert.Contains("CP0001", change.RuleId, StringComparison.Ordinal);
            Assert.Equal("Sample.Gone", change.Target);
        }

        [Fact]
        public void VirtualRemoved_IsBinaryBreaking()
        {
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                        public virtual int Read() { return 1; }
                    }
                }
                """, "public virtual int", "public int", ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0012", change.RuleId, StringComparison.Ordinal);
        }

        [Fact]
        public void VirtualAdded_IsAlsoBinaryBreaking()
        {
            // Counter-intuitive, and in the published rules: CP0013.
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                        public int Read() { return 1; }
                    }
                }
                """, "public int Read", "public virtual int Read", ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0013", change.RuleId, StringComparison.Ordinal);
        }

        [Fact]
        public void StaticAdded_IsBinaryBreaking()
        {
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                        public int Read() { return 1; }
                    }
                }
                """, "public int Read", "public static int Read", ChangeLevel.BinaryBreaking);

            Assert.Equal("AC-STATIC-CHANGED", change.RuleId);
        }

        [Fact]
        public void VisibilityReduced_IsBinaryBreaking()
        {
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                        public int Read() { return 1; }
                    }
                }
                """, "public int Read", "protected int Read", ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0019", change.RuleId, StringComparison.Ordinal);
        }

        [Fact]
        public void FieldBecomesProperty_IsOneBinaryBreak_NotARemovalPlusAnAddition()
        {
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                        public int Count;
                    }
                }
                """, "public int Count;", "public int Count { get; set; }", ChangeLevel.BinaryBreaking);

            Assert.Equal("AC-FIELD-PROPERTY-SWAP", change.RuleId);
            Assert.Contains("ldfld", change.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void ReadonlyAddedToAField_IsBinaryBreaking()
        {
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                        public int Count;
                    }
                }
                """, "public int Count;", "public readonly int Count;", ChangeLevel.BinaryBreaking);

            Assert.Equal("AC-FIELD-READONLY-ADDED", change.RuleId);
        }

        [Fact]
        public void EnumUnderlyingTypeChanged_IsBinaryBreaking()
        {
            var change = Single("""
                namespace Sample
                {
                    public enum Mode : int
                    {
                        Off = 0,
                    }
                }
                """, "Mode : int", "Mode : long", ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0010", change.RuleId, StringComparison.Ordinal);
        }

        [Fact]
        public void ImplementedInterfaceRemoved_IsBinaryBreaking()
        {
            var change = Single("""
                namespace Sample
                {
                    public interface IReadable
                    {
                    }

                    public class Store : IReadable
                    {
                    }
                }
                """, "public class Store : IReadable", "public class Store", ChangeLevel.BinaryBreaking);

            Assert.Contains("CP0008", change.RuleId, StringComparison.Ordinal);
        }

        [Fact]
        public void InstanceFieldAddedToAStructWithOnlyPublicFields_IsBinaryBreaking()
        {
            var comparison = Compare("""
                namespace Sample
                {
                    public struct Point
                    {
                        public int X;
                    }
                }
                """, """
                namespace Sample
                {
                    public struct Point
                    {
                        public int X;

                        public int Y;
                    }
                }
                """);

            var change = Assert.Single(comparison.At(ChangeLevel.BinaryBreaking));

            Assert.Equal("AC-STRUCT-FIELD-ADDED", change.RuleId);
            Assert.Contains("SkipLocalsInit", change.Message, StringComparison.Ordinal);
        }

        // ------------------------------------------------------------------------------ level 4: source break

        [Fact]
        public void ParameterRenamed_IsSourceBreakingOnly()
        {
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void Send(int a) { }
                    }
                }
                """, "int a", "int value", ChangeLevel.SourceBreaking);

            Assert.Contains("CP0017", change.RuleId, StringComparison.Ordinal);
            Assert.Contains("named arguments", change.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void GenericConstraintTightened_IsSourceBreaking()
        {
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void Send<T>(T value) { }
                    }
                }
                """, "Send<T>(T value) { }", "Send<T>(T value) where T : class { }",
                ChangeLevel.SourceBreaking);

            Assert.Equal("AC-GENERIC-CONSTRAINT-CHANGED", change.RuleId);

            // Recorded because it is the clearest thing this tool does that ApiCompat does not.
            Assert.Contains("ApiCompat", change.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void NewOverloadAdded_IsReportedAsASourceRisk_NotAsAFinding()
        {
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void Send(int a) { }
                    }
                }
                """,
                "public static void Send(int a) { }",
                "public static void Send(int a) { }\n        public static void Send(string a) { }",
                ChangeLevel.SourceBreaking);

            Assert.Equal("AC-OVERLOAD-ADDED", change.RuleId);

            // It must say it is a risk. The tool cannot see consumer call sites, and a confident verdict it
            // has no evidence for is worse than a hedged one it does.
            Assert.Contains("RISK", change.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void DefaultValueRemoved_IsSourceBreaking_NotBinary()
        {
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void Send(int a = 1) { }
                    }
                }
                """, "int a = 1", "int a", ChangeLevel.SourceBreaking);

            Assert.Equal("AC-DEFAULT-REMOVED", change.RuleId);
        }

        // ----------------------------------------------------------------------------------- level 3 and below

        [Fact]
        public void NewTypeAdded_IsAdditive()
        {
            var change = Single("""
                namespace Sample
                {
                    public class Store
                    {
                    }
                }
                """, "public class Store\n    {\n    }",
                "public class Store\n    {\n    }\n\n    public class Cache\n    {\n    }",
                ChangeLevel.Additive);

            Assert.Equal("AC-TYPE-ADDED", change.RuleId);
        }

        [Fact]
        public void TheSnippetHelperIsIndependentOfHowThisFileWasCheckedOut()
        {
            // Every other test here takes its snippet from a raw string literal, so its line endings are the
            // FILE's — LF in git, CRLF on a Windows runner because .gitattributes does not pin *.cs. That made
            // NewTypeAdded_IsAdditive red on windows-latest and green on ubuntu-latest. The CRLF below is
            // written explicitly instead of inherited, so this pins the normalisation in Single() on either
            // checkout; without it, this test is the one that goes red.
            var change = Single(
                "namespace Sample\r\n{\r\n    public class Store\r\n    {\r\n    }\r\n}\r\n",
                "public class Store\n    {\n    }",
                "public class Store\n    {\n    }\n\n    public class Cache\n    {\n    }",
                ChangeLevel.Additive);

            Assert.Equal("AC-TYPE-ADDED", change.RuleId);
        }

        [Fact]
        public void DefaultValueAdded_IsAdditive()
        {
            var change = Single("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static void Send(int a) { }
                    }
                }
                """, "int a)", "int a = 1)", ChangeLevel.Additive);

            Assert.Equal("AC-DEFAULT-ADDED", change.RuleId);
        }

        [Fact]
        public void PrivateMethodBodyChanged_IsInternalOnly_WithNoClassifiedChange()
        {
            var comparison = Compare("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static int A() { return Helper(); }

                        private static int Helper() { return 41; }
                    }
                }
                """.Replace("41", "41"), """
                namespace Sample
                {
                    public static class Ops
                    {
                        public static int A() { return Helper(); }

                        private static int Helper() { return 42; }
                    }
                }
                """);

            Assert.Equal(ChangeLevel.InternalOnly, comparison.HighestLevel);
            Assert.Empty(comparison.Changes);

            // The pair that made this distinction worth having: nothing to tell consumers, and still
            // something to retest.
            Assert.True(comparison.BreaksNoConsumer);
            Assert.True(comparison.RequiresRetest);
        }

        [Fact]
        public void SameSourceCompiledTwice_IsLevelOneAndNothingElse()
        {
            // The negative that keeps the tool usable. If a package that changed nothing produces a wall of
            // level 4s, nobody runs it a second time.
            const string source = """
                namespace Sample
                {
                    public interface IStore
                    {
                        int Read();
                    }

                    public abstract class Store : IStore
                    {
                        public const int Max = 8;

                        public abstract int Read();

                        public virtual void Send(int a = 1, string b = "x") { }
                    }

                    public enum Mode
                    {
                        Off = 0,
                        Fast = 1,
                    }

                    public struct Point
                    {
                        public int X;
                    }
                }
                """;

            var comparison = AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(source),
                TinyAssemblyCompiler.Compile(source));

            Assert.Equal(ChangeLevel.Inert, comparison.HighestLevel);
            Assert.Empty(comparison.Changes);
            Assert.Empty(comparison.MethodDifferences);
            Assert.Empty(comparison.Warnings);
        }

        [Fact]
        public void AssemblyVersionBump_DoesNotRaiseTheLevelOrCountAsABlindSpot()
        {
            // Found by running the tool on real packages rather than by reasoning: the assembly version was
            // being reported as a WARNING, and a warning suppresses BreaksNoConsumer. Every release moves the
            // assembly version, so the property came back false for System.Numerics.Tensors 10.0.10 -> 10.0.11
            // — a comparison with zero findings at every level. A property that is false for every release is
            // noise, and noise is what people stop reading.
            const string source = """
                [assembly: System.Reflection.AssemblyVersion("1.0.0.0")]

                namespace Sample
                {
                    public static class Ops
                    {
                        public static int A() { return 1; }
                    }
                }
                """;

            var comparison = Compare(source, source.Replace("1.0.0.0", "1.0.1.0"));

            Assert.Equal(ChangeLevel.Inert, comparison.HighestLevel);
            Assert.Empty(comparison.Changes);
            Assert.Empty(comparison.Warnings);
            Assert.True(comparison.BreaksNoConsumer);
            Assert.False(comparison.RequiresRetest);

            var note = Assert.Single(comparison.Notes);

            Assert.Contains("1.0.0.0 -> 1.0.1.0", note, StringComparison.Ordinal);
        }

        [Fact]
        public void AssemblyNameChanged_IsBinaryBreaking_UnlikeAVersionBump()
        {
            var left = TinyAssemblyCompiler.Compile("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static int A() { return 1; }
                    }
                }
                """, "Before");

            var right = TinyAssemblyCompiler.Compile("""
                namespace Sample
                {
                    public static class Ops
                    {
                        public static int A() { return 1; }
                    }
                }
                """, "After");

            var comparison = AssemblyComparer.CompareImages(left, right);

            Assert.Equal(ChangeLevel.BinaryBreaking, comparison.HighestLevel);

            var change = Assert.Single(comparison.At(ChangeLevel.BinaryBreaking));

            Assert.Contains("CP0003", change.RuleId, StringComparison.Ordinal);
            Assert.Contains("the whole assembly", change.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void AnUnrecognisedChangeFailsClosed()
        {
            // The fallback is driven at the classifier's own seam rather than through compiled source, and that
            // is a property of metadata rather than a shortcut: every facet a reader can write arrives coupled
            // to one an existing rule already reads, so a rule claims the pair first. The nearest real
            // candidate is pinned by DefaultImplementationRemoved_IsClaimedByAnExistingRule below — when that
            // test goes red, this construction has stopped being hypothetical and this one should follow it.
            //
            // `default-impl` moving ALONE is what "a facet no rule names" means here: the only rule reading
            // HasDefaultImplementation fires on false -> true, so true -> false reaches no rule at all. In real
            // metadata `abstract` flips with it, which is exactly why the pair below cannot be compiled.
            var before = InterfaceMethod(hasDefaultImplementation: true);
            var after = InterfaceMethod(hasDefaultImplementation: false);

            // If these ever compare equal the difference would not exist and the test would pass vacuously.
            Assert.NotEqual(before.Descriptor, after.Descriptor);

            using var facts = AssemblyFacts.FromImage(
                TinyAssemblyCompiler.Compile("""
                    namespace Sample
                    {
                        public interface IStore
                        {
                            int Read();
                        }
                    }
                    """),
                "TinyAssembly.dll");

            var changes = BreakingChangeClassifier.Classify(
                facts,
                facts,
                new[] { new ApiDifference(DifferenceKind.Changed, before, after) });

            var change = Assert.Single(changes);

            Assert.Equal("AC-UNCLASSIFIED", change.RuleId);

            // The guarantee the name promises, and the only one that matters: a facet no rule understands is
            // reported at the breaking level, never assumed additive and waved through.
            Assert.Equal(ChangeLevel.BinaryBreaking, change.Level);

            // Both descriptors travel with the finding — with no rule to name what moved, the reader's only
            // next step is diffing them.
            Assert.Contains(before.Descriptor, change.Message, StringComparison.Ordinal);
            Assert.Contains(after.Descriptor, change.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void DefaultImplementationRemoved_IsClaimedByAnExistingRule()
        {
            // Load-bearing for the test above rather than a rule test of its own: it is the nearest change a
            // reader CAN write to the unclassified facet, and it asserts that a rule still claims it. Removing
            // the body flips `abstract` in the same edit, and AC-VIRTUAL-MADE-ABSTRACT reads that.
            var change = Single("""
                namespace Sample
                {
                    public interface IStore
                    {
                        int Read() { return 1; }
                    }
                }
                """, "int Read() { return 1; }", "int Read();", ChangeLevel.BinaryBreaking);

            Assert.Equal("AC-VIRTUAL-MADE-ABSTRACT", change.RuleId);
        }

        // --------------------------------------------------------------------------------------------- helpers

        /// <summary>
        /// One interface method, differing between the two calls in <c>default-impl</c> and nothing else.
        ///
        /// <para>Deliberately not compiled. A real interface method carries <c>abstract</c> in lockstep with the
        /// absence of a body, so this exact pair cannot come out of metadata — which is the point: it is the
        /// shape of a facet the classifier does not yet have a rule for, and the only way to reach the
        /// fail-closed fallback on purpose.</para>
        /// </summary>
        private static ApiMember InterfaceMethod(bool hasDefaultImplementation)
        {
            return new ApiMember
            {
                DeclaringType = "Sample.IStore",
                Kind = ApiMemberKind.Method,
                Name = "Read",
                Signature = "() : System.Int32",
                Accessibility = "Public",
                IsVirtual = true,
                HasDefaultImplementation = hasDefaultImplementation,
            };
        }

        private static AssemblyComparison Compare(string left, string right)
        {
            return AssemblyComparer.CompareImages(
                TinyAssemblyCompiler.Compile(left),
                TinyAssemblyCompiler.Compile(right));
        }

        /// <summary>
        /// Compiles <paramref name="source"/>, compiles it again with one substitution, and asserts that
        /// exactly one classified change came out at <paramref name="expected"/> — and that nothing came out
        /// above it.
        ///
        /// <para>Both halves matter. "There is a level-5 finding" is satisfied by a classifier that reports
        /// level 5 for everything; "and nothing worse, and only one at this level" is not.</para>
        /// </summary>
        private static ApiChange Single(
            string source,
            string find,
            string replace,
            ChangeLevel expected)
        {
            // `source` comes from a raw string literal, and a raw string literal carries whatever line endings
            // the FILE has — the compiler does not normalise them. This file is LF in git and .gitattributes
            // says nothing about *.cs, so a Windows runner checks it out as CRLF while `find` and `replace`
            // stay LF, being escaped literals with an explicit \n. That mismatch made the guard below fail on
            // windows-latest only. What is under test is the CONTENT of the snippet, not how it was checked
            // out, so normalise once here and let every caller match against one form.
            var snippet = source.Replace("\r\n", "\n", StringComparison.Ordinal);

            Assert.Contains(find, snippet, StringComparison.Ordinal);

            var comparison = Compare(snippet, snippet.Replace(find, replace, StringComparison.Ordinal));

            Assert.Equal(expected, comparison.HighestLevel);

            return Assert.Single(comparison.At(expected));
        }
    }
}
