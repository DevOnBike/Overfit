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
    /// OVERFIT010 — per-call <c>new</c> of a growable BCL collection
    /// (<c>List</c>/<c>Dictionary</c>/<c>HashSet</c>/<c>Queue</c>/<c>Stack</c>/<c>StringBuilder</c>/
    /// <c>MemoryStream</c>) inside a method body. Each call allocates the wrapper plus a backing
    /// array that grows by doubling — exactly the per-call GC pressure the hot-path contract forbids.
    /// Pool the instance and <c>Clear()</c> it (<c>docs/performance-patterns.md</c> #1 ObjectPool),
    /// reach for a <c>ValueStringBuilder</c>/<c>stackalloc</c> span (#26), or a recyclable stream (#46p).
    ///
    /// Same noise policy as the other per-call rules: one-time contexts (field/property initializers,
    /// constructors) and the exception path are exempt — a pool's own backing collection is built in a
    /// ctor, and an error message may format freely. Justify a hot site with
    /// <c>#pragma warning disable OVERFIT010</c> + a reason.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class CollectionAllocationAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT010";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Growable collection allocation in per-call code",
            messageFormat: "Allocates 'new {0}' (wrapper + doubling backing array) in per-call code — pool and Clear() it, or use a ValueStringBuilder/stackalloc span",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Per-call construction of List/Dictionary/HashSet/Queue/Stack/StringBuilder/MemoryStream churns the GC on hot paths. Pool the instance, or use a value-type span builder. One-time contexts (field initializers, constructors) and the exception path are exempt.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeObjectCreation, OperationKind.ObjectCreation);
        }

        private static void AnalyzeObjectCreation(OperationAnalysisContext context)
        {
            var operation = (IObjectCreationOperation)context.Operation;

            if (operation.IsImplicit || operation.Type is not INamedTypeSymbol type)
            {
                return;
            }

            if (!IsTrackedCollection(type))
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
                type.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)));
        }

        /// <summary>The growable mutable collections whose per-call construction is worth a pool —
        /// matched by name AND namespace so a same-named user type never trips the rule.</summary>
        private static bool IsTrackedCollection(INamedTypeSymbol type)
        {
            var ns = type.ContainingNamespace;

            switch (type.Name)
            {
                case "List":
                case "Dictionary":
                case "HashSet":
                case "Queue":
                case "Stack":
                    return IsNamespace(ns, "System", "Collections", "Generic");

                case "StringBuilder":
                    return IsNamespace(ns, "System", "Text");

                case "MemoryStream":
                    return IsNamespace(ns, "System", "IO");

                default:
                    return false;
            }
        }

        private static bool IsNamespace(INamespaceSymbol? ns, string outer, string middle, string inner)
        {
            return ns is { Name: { } i } && i == inner
                && ns.ContainingNamespace is { Name: { } m } && m == middle
                && ns.ContainingNamespace.ContainingNamespace is { Name: { } o, ContainingNamespace.IsGlobalNamespace: true } && o == outer;
        }

        private static bool IsNamespace(INamespaceSymbol? ns, string outer, string inner)
        {
            return ns is { Name: { } i } && i == inner
                && ns.ContainingNamespace is { Name: { } o, ContainingNamespace.IsGlobalNamespace: true } && o == outer;
        }
    }
}
