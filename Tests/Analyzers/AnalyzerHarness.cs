// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// Compiles a snippet in memory and runs one analyzer over it.
    ///
    /// <para><b>The first analyzer test harness this repository has had.</b> Thirty-two analyzers shipped
    /// with none: every OVERFIT rule was verified by building the tree and looking at what came out, which
    /// proves a rule does not fire on clean code and says nothing at all about whether it fires on dirty
    /// code. A rule that never reports is indistinguishable from a rule that is correct, and that is the
    /// same silence-versus-health confusion this codebase spends its time removing elsewhere.</para>
    ///
    /// <para>Hand-rolled rather than <c>Microsoft.CodeAnalysis.Testing</c>: the whole need is "compile a
    /// string, run one analyzer, return the ids", and the testing package brings a verifier framework, its
    /// own reference resolution and a second opinion about diagnostics formatting. Thirty lines here beat a
    /// dependency that has to be understood before the first test can be read.</para>
    /// </summary>
    internal static class AnalyzerHarness
    {
        /// <summary>
        /// Framework assemblies resolved out of the test host's own trusted-platform list.
        ///
        /// <para>Deliberately short. <c>System.Console.dll</c> earned its place on 2026-08-12 with
        /// OVERFIT040's console-writer exclusion, which cannot be tested without the real
        /// <c>Console.Out</c> property to bind against — a hand-written stand-in named <c>Console</c> would
        /// pin the test's own type name rather than the rule. Compared by file name rather than by suffix so
        /// that adding a short name cannot silently match a longer one.</para>
        /// </summary>
        private static readonly HashSet<string> Referenced = new(StringComparer.OrdinalIgnoreCase)
        {
            "System.Runtime.dll",
            "System.Console.dll",
        };

        /// <summary>Diagnostic ids the analyzer reports for <paramref name="source"/>, in source order.</summary>
        public static IReadOnlyList<string> Run(DiagnosticAnalyzer analyzer, string source)
        {
            ArgumentNullException.ThrowIfNull(analyzer);
            ArgumentNullException.ThrowIfNull(source);

            var tree = CSharpSyntaxTree.ParseText(source);

            // The reference set is deliberately minimal — these rules are about types written in the source,
            // not about anything resolved from the framework. A missing reference would surface as a CS0246
            // that the assertion below turns into a readable failure rather than an empty diagnostic list.
            var references = new List<MetadataReference>
            {
                MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
            };

            var runtime = AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string ?? string.Empty;

            foreach (var path in runtime.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
            {
                if (Referenced.Contains(Path.GetFileName(path)))
                {
                    references.Add(MetadataReference.CreateFromFile(path));
                }
            }

            var compilation = CSharpCompilation.Create(
                "AnalyzerHarness",
                [tree],
                references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

            // A snippet that does not compile makes every diagnostic list meaningless, so it fails loudly
            // here instead of quietly returning nothing.
            var syntaxErrors = compilation.GetDiagnostics()
                .Where(d => d.Severity == DiagnosticSeverity.Error && d.Id != "CS5001")
                .Select(d => d.ToString())
                .ToArray();

            Assert.True(
                syntaxErrors.Length == 0,
                "the test snippet does not compile: " + string.Join("; ", syntaxErrors));

            var withAnalyzer = compilation.WithAnalyzers(ImmutableArray.Create(analyzer));
            var results = withAnalyzer.GetAnalyzerDiagnosticsAsync().GetAwaiter().GetResult();

            return results
                .OrderBy(d => d.Location.SourceSpan.Start)
                .Select(d => d.Id)
                .ToArray();
        }
    }
}
