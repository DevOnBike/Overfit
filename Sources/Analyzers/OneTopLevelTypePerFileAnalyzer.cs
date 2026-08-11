// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT034 — one namespace-level type per file.
    ///
    /// <para><b>Replaces the <c>BanMultipleTopLevelTypes</c> MSBuild task</b>, whose regular expression
    /// required top-level types to be indented by exactly four spaces. That is true while every file uses
    /// block-scoped namespaces and false the moment one does not — and a rule that silently stops applying
    /// is worse than no rule, because the directory still looks guarded. This counts declarations in the
    /// syntax tree, so indentation and namespace style stop mattering.</para>
    ///
    /// <para><b>Nested types are fine</b> — the point is that a reader can find <c>Foo</c> in <c>Foo.cs</c>,
    /// not that files hold one type object. <b>Partial declarations of the same type are fine</b> too, and
    /// collapsing them by name is why this counts distinct names rather than declarations.</para>
    ///
    /// <para>Reported once per file, on the second declaration, so a file with five types produces one
    /// diagnostic naming the problem rather than four telling the same story.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class OneTopLevelTypePerFileAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT034";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "More than one top-level type in a file",
            messageFormat: "This file declares {0} namespace-level types ({1}) — keep one per file so the file name finds the type; nested types and same-name partials are fine",
            category: "Design",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A reader looking for a type should find it in the file named after it. Helper enums, records and option types belong in their own files.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxTreeAction(AnalyzeTree);
        }

        private static void AnalyzeTree(SyntaxTreeAnalysisContext context)
        {
            var root = context.Tree.GetRoot(context.CancellationToken);
            var names = new SortedSet<string>(StringComparer.Ordinal);
            Location? second = null;

            foreach (var node in root.DescendantNodes())
            {
                if (node is not BaseTypeDeclarationSyntax and not DelegateDeclarationSyntax)
                {
                    continue;
                }

                // Namespace level only: anything whose parent is another type is nested, and nesting is the
                // recommended alternative rather than the thing being banned.
                if (node.Parent is BaseTypeDeclarationSyntax)
                {
                    continue;
                }

                var name = node switch
                {
                    BaseTypeDeclarationSyntax type => type.Identifier.ValueText,
                    DelegateDeclarationSyntax del => del.Identifier.ValueText,
                    _ => string.Empty,
                };

                if (name.Length == 0)
                {
                    continue;
                }

                // Distinct NAMES, so `partial class Foo` across three declarations is one type.
                if (names.Add(name) && names.Count == 2)
                {
                    second = node.GetLocation();
                }
            }

            if (names.Count > 1 && second != null)
            {
                context.ReportDiagnostic(
                    Diagnostic.Create(Rule, second, names.Count, string.Join(", ", names)));
            }
        }
    }
}
