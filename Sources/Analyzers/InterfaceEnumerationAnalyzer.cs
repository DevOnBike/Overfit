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
    /// OVERFIT005 — <c>foreach</c> over an interface-typed collection in per-call code: when the
    /// static type is <c>IEnumerable&lt;T&gt;</c>/<c>IList&lt;T&gt;</c>/…, <c>GetEnumerator()</c>
    /// returns an interface — a heap-allocated enumerator per loop plus interface dispatch per
    /// element. Iterating the concrete type (<c>T[]</c>, <c>List&lt;T&gt;</c>, <c>Span&lt;T&gt;</c>)
    /// uses a struct enumerator or no enumerator at all.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class InterfaceEnumerationAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT005";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "foreach over an interface-typed collection in per-call code",
            messageFormat: "foreach over '{0}' allocates an enumerator and interface-dispatches every element — take the concrete type (T[], List<T>, Span<T>) or an indexed for loop",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Enumerating via an interface type boxes the enumerator and virtualizes MoveNext/Current. Hot code should iterate concrete types or use indexed loops.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeForEach, OperationKind.Loop);
        }

        private static void AnalyzeForEach(OperationAnalysisContext context)
        {
            if (context.Operation is not IForEachLoopOperation operation)
            {
                return;
            }

            if (OverfitPerfAnalysis.IsOneTimeAllocationContext(context.ContainingSymbol))
            {
                return;
            }

            // Unwrap the compiler's implicit conversion to IEnumerable around the collection.
            var collection = operation.Collection;

            while (collection is IConversionOperation { IsImplicit: true } conversion)
            {
                collection = conversion.Operand;
            }

            if (collection.Type is not { TypeKind: TypeKind.Interface } interfaceType)
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(
                Rule,
                collection.Syntax.GetLocation(),
                interfaceType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)));
        }
    }
}
