// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT013 — reading <c>.Count</c> on a <c>ConcurrentQueue&lt;T&gt;</c> or
    /// <c>ConcurrentBag&lt;T&gt;</c>. Unlike the indexed collections, these have no O(1) count: the
    /// property takes a snapshot and walks every internal segment under synchronization, so a
    /// <c>while (q.Count > 0)</c> loop is quadratic and contends with the producers. Use
    /// <c>IsEmpty</c> for the empty/non-empty test (O(1), lock-free) or maintain an approximate
    /// <c>Interlocked</c> counter alongside the collection. See <c>docs/performance-patterns.md</c> #80.
    ///
    /// One-time contexts (a count logged once at construction) and the exception path are exempt,
    /// like the other per-call rules.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ConcurrentCollectionCountAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT013";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: ".Count on a concurrent collection",
            messageFormat: "'{0}.Count' walks every internal segment under synchronization — use IsEmpty for the empty test, or keep an approximate Interlocked counter",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "ConcurrentQueue/ConcurrentBag .Count is a synchronized segment walk, not O(1). Use IsEmpty or an Interlocked counter on hot paths. One-time contexts and the exception path are exempt.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzePropertyReference, OperationKind.PropertyReference);
        }

        private static void AnalyzePropertyReference(OperationAnalysisContext context)
        {
            var operation = (IPropertyReferenceOperation)context.Operation;
            var property = operation.Property;

            if (property.Name != "Count" || !IsConcurrentSegmentCollection(property.ContainingType))
            {
                return;
            }

            if (OverfitPerfAnalysis.IsOneTimeAllocationContext(context.ContainingSymbol))
            {
                return;
            }

            if (OverfitPerfAnalysis.IsOnExceptionPath(operation))
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(
                Rule,
                operation.Syntax.GetLocation(),
                property.ContainingType.Name));
        }

        private static bool IsConcurrentSegmentCollection(INamedTypeSymbol? type)
        {
            if (type is not { Name: "ConcurrentQueue" or "ConcurrentBag" })
            {
                return false;
            }

            // System.Collections.Concurrent
            return type.ContainingNamespace is
            {
                Name: "Concurrent",
                ContainingNamespace: { Name: "Collections", ContainingNamespace: { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true } }
            };
        }
    }
}
