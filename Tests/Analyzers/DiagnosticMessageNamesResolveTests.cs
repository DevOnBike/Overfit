// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Reflection;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.Analyzers;
using DevOnBike.Overfit.Inference;
using Microsoft.CodeAnalysis;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// Every Overfit-owned type name written into an analyzer diagnostic must still resolve against
    /// <c>DevOnBike.Overfit</c>. OVERFIT001 told developers to use <c>PooledArray</c> for three months
    /// after that type was deleted (2026-05-29), and it was found by accident while chasing something
    /// else; nothing in the build noticed. This is that nothing.
    ///
    /// <para><b>What this test does NOT cover — read this before treating a green run as "the advice a
    /// developer sees has been checked".</b> It sees only identifiers written as literal text in a
    /// descriptor, so it misses two of the four defects the 2026-08-15 sweep actually found:</para>
    ///
    /// <list type="bullet">
    /// <item>a name <b>composed at report time</b> — OVERFIT015 used to emit <c>CpuFeatures.Has{0}</c>
    /// from the containing type's name, so <c>Avx512F</c> produced <c>CpuFeatures.HasAvx512F</c> against
    /// a field called <c>HasAvx512</c>. No static string ever contains the composed name, so neither this
    /// test nor a text search can see it. The rule that catches that shape is structural and lives in
    /// <c>Sources/Analyzers/README.md</c>'s authoring notes: never assert a symbol exists without having
    /// resolved it;</item>
    /// <item><b>prose describing a mechanism</b> — OVERFIT002's description cited "the MSBuild guard"
    /// after that guard became OVERFIT033. There is no identifier to resolve;</item>
    /// <item><b>member names</b>, and therefore all of <c>BannedSymbols.txt</c> — the RS0030 message text
    /// named <c>PooledBuffer&lt;T&gt;.RentArray</c> long after that member was removed. Resolving members
    /// needs a different extractor and its prose names most of the BCL, which is the unbounded
    /// false-positive shape this test deliberately avoids.</item>
    /// </list>
    ///
    /// <para>Scope is the closed set of Overfit-owned identifiers rather than "every CamelCase word":
    /// descriptor text legitimately names <c>Span&lt;T&gt;</c>, <c>Interlocked</c>, <c>CancellationToken</c>,
    /// rule ids and format placeholders, and a general extractor needs an allow-list that grows with every
    /// new message — the kind of guard that eventually gets suppressed instead of read.</para>
    /// </summary>
    public sealed class DiagnosticMessageNamesResolveTests
    {
        /// <summary>
        /// Case-sensitive, and <c>Overfit</c> deliberately requires at least one further character: the bare
        /// word appears in descriptor text only as a namespace segment
        /// (<c>DevOnBike.Overfit.Intrinsics.CpuFeatures</c>) and as prose ("an Overfit buffer"), neither of
        /// which is a type name. Case sensitivity is also what keeps rule ids (<c>OVERFIT001</c>) and the
        /// editorconfig key <c>overfit_max_stackalloc_bytes</c> out without an exclusion list — an exclusion
        /// list is where a real check goes to hide. Generic arity needs no stripping either: the trailing
        /// <c>\b</c> stops before the <c>&lt;</c> in <c>PooledBuffer&lt;T&gt;</c>.
        /// </summary>
        private static readonly Regex OverfitOwnedName = new(
            @"\b(Overfit\w+|Pooled\w+|TensorStorage|FastTensor|CpuFeatures|ValueStringBuilder)\b",
            RegexOptions.CultureInvariant);

        private static readonly string[] NamesTheRuleSetActuallyUses =
        [
            "PooledBuffer", "TensorStorage", "OverfitParallel", "CpuFeatures",
            "OverfitEnvironment", "ValueStringBuilder", "OverfitSchemas", "OverfitHotPath"
        ];

        [Fact]
        public void EveryOverfitTypeNamedInADiagnosticResolves()
        {
            var descriptors = AllDescriptors();
            var libraryTypes = LibraryTypeNames();

            Assert.True(
                descriptors.Count >= 45,
                $"Only {descriptors.Count} descriptors were enumerated; the rule set has 47 and rules are only ever added. " +
                "A shrinking denominator means the reflection walk stopped seeing analyzers, not that the rules went away.");

            Assert.True(
                libraryTypes.Count >= 100,
                $"Only {libraryTypes.Count} types were read from DevOnBike.Overfit. With an empty or near-empty type list " +
                "every name would 'fail to resolve' and the result would say nothing about the messages.");

            var unresolved = new List<string>();

            foreach (var (owner, field, descriptor) in descriptors)
            {
                foreach (var named in ExtractNames(descriptor))
                {
                    if (!libraryTypes.Contains(ResolvedAs(named)))
                    {
                        unresolved.Add($"{descriptor.Id} ({owner}.{field}) names '{named}' — no such type in DevOnBike.Overfit");
                    }
                }
            }

            Assert.True(
                unresolved.Count == 0,
                "Analyzer diagnostics tell developers to use types that do not exist:" + Environment.NewLine +
                string.Join(Environment.NewLine, unresolved));
        }

        /// <summary>
        /// Pins the extractor's yield, so a regex or reflection change that stops matching fails loudly here
        /// instead of making <see cref="EveryOverfitTypeNamedInADiagnosticResolves"/> pass vacuously.
        /// </summary>
        [Fact]
        public void TheExtractorStillFindsTheNamesTheRuleSetUses()
        {
            var descriptors = AllDescriptors();

            Assert.True(descriptors.Count >= 45, $"Only {descriptors.Count} descriptors were enumerated.");

            var extracted = new SortedSet<string>(StringComparer.Ordinal);

            foreach (var (_, _, descriptor) in descriptors)
            {
                extracted.UnionWith(ExtractNames(descriptor));
            }

            foreach (var expected in NamesTheRuleSetActuallyUses)
            {
                Assert.Contains(expected, extracted);
            }
        }

        /// <summary>
        /// The negative control: the extractor and the resolver, run over a descriptor built here that names
        /// the type this whole test exists because of. It exercises the extractor only — it is NOT a
        /// substitute for the real assertion, which is the one that depends on the reflection walk finding
        /// anything at all.
        /// </summary>
        [Fact]
        public void ADescriptorNamingADeletedTypeIsReportedUnresolved()
        {
            var control = new DiagnosticDescriptor(
                "OVERFIT000",
                title: "negative control",
                messageFormat: "Allocates 'new {0}[]' on the heap in per-call code — use PooledArray",
                category: "Performance",
                defaultSeverity: DiagnosticSeverity.Warning,
                isEnabledByDefault: true);

            Assert.Contains("PooledArray", ExtractNames(control));
            Assert.DoesNotContain("PooledArray", LibraryTypeNames());
        }

        private static List<(string Owner, string Field, DiagnosticDescriptor Descriptor)> AllDescriptors()
        {
            var assembly = typeof(HeapArrayAllocationAnalyzer).Assembly;
            var found = new List<(string, string, DiagnosticDescriptor)>();

            // Reflection over static fields, NOT SupportedDiagnostics: OVERFIT035 and OVERFIT036 live on
            // OverfitSchemasGenerator, an IIncrementalGenerator with no SupportedDiagnostics at all, and
            // OVERFIT900 is an internal field on a static helper. An id-keyed enumeration misses three
            // rules silently; this one misses none. NonPublic is required for exactly that reason.
            //
            // GetTypes() is deliberately not wrapped in a catch that filters ex.Types for non-null: doing
            // that would shrink the denominator silently, which is the vacuous-pass shape this file guards
            // against. If it throws, the reference is what needs fixing.
            foreach (var type in assembly.GetTypes())
            {
                foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.DeclaredOnly))
                {
                    if (field.FieldType != typeof(DiagnosticDescriptor))
                    {
                        continue;
                    }

                    var descriptor = (DiagnosticDescriptor?)field.GetValue(null);

                    Assert.NotNull(descriptor);
                    found.Add((type.Name, field.Name, descriptor!));
                }
            }

            return found;
        }

        private static SortedSet<string> ExtractNames(DiagnosticDescriptor descriptor)
        {
            var names = new SortedSet<string>(StringComparer.Ordinal);
            var texts = new[]
            {
                descriptor.Title.ToString(),
                descriptor.MessageFormat.ToString(),
                descriptor.Description.ToString()
            };

            foreach (var text in texts)
            {
                foreach (Match match in OverfitOwnedName.Matches(text))
                {
                    names.Add(match.Value);
                }
            }

            return names;
        }

        private static HashSet<string> LibraryTypeNames()
        {
            var names = new HashSet<string>(StringComparer.Ordinal);

            foreach (var type in typeof(InferenceEngine).Assembly.GetTypes())
            {
                var name = type.Name;
                var arity = name.IndexOf('`');

                names.Add(arity < 0 ? name : name.Substring(0, arity));
            }

            return names;
        }

        /// <summary>An attribute is written <c>[OverfitHotPath]</c> in prose and declared with the
        /// <c>Attribute</c> suffix; that is a C# spelling convention, not message rot.</summary>
        private static string ResolvedAs(string named)
        {
            return named == "OverfitHotPath" ? "OverfitHotPathAttribute" : named;
        }
    }
}
