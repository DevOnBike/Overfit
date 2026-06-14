// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT012 — a finalizer (<c>~T()</c>) is declared. Finalizable objects take a slower
    /// allocation path (they go on the finalization queue at construction) and survive at least one
    /// extra GC generation — the collector must run the finalizer on a separate pass before the
    /// memory is reclaimed. Overfit owns its native lifetimes through <c>IDisposable</c> + pooled
    /// <c>TensorStorage&lt;T&gt;</c>/<c>PooledBuffer&lt;T&gt;</c>, deterministically. If a finalizer
    /// is genuinely needed as a safety net, pair it with <c>GC.SuppressFinalize(this)</c> in
    /// <c>Dispose()</c> so the common path skips finalization. See <c>docs/performance-patterns.md</c> #48a.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class FinalizerAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT012";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Finalizer declared",
            messageFormat: "'{0}' declares a finalizer — finalizable objects allocate slower and survive an extra GC generation; release resources through IDisposable + GC.SuppressFinalize instead",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Finalizers put objects on the finalization queue at construction and delay reclamation by a GC generation. Prefer deterministic IDisposable cleanup; if a finalizer is a required safety net, suppress it in Dispose with GC.SuppressFinalize.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSymbolAction(AnalyzeMethod, SymbolKind.Method);
        }

        private static void AnalyzeMethod(SymbolAnalysisContext context)
        {
            var method = (IMethodSymbol)context.Symbol;

            if (method.MethodKind != MethodKind.Destructor)
            {
                return;
            }

            var location = method.Locations.Length > 0 ? method.Locations[0] : Location.None;

            context.ReportDiagnostic(Diagnostic.Create(
                Rule,
                location,
                method.ContainingType?.Name ?? method.Name));
        }
    }
}
