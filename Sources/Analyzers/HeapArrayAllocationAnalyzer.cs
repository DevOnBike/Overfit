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
    /// OVERFIT001 — heap array allocation in per-call code.
    ///
    /// The Overfit hot-path contract is "no hidden allocations in inference": scratch memory comes
    /// from <c>PooledBuffer&lt;T&gt;</c> / <c>PooledArray</c> (pooled), long-lived tensor data from
    /// <c>TensorStorage&lt;T&gt;</c>, and small fixed-size scratch from <c>stackalloc</c>. A bare
    /// <c>new T[n]</c> inside a method body defeats all of that and wakes the GC on every call.
    ///
    /// Noise policy (deliberate): ONE-TIME allocations are exempt — field/property initializers,
    /// instance constructors (layer weights live for the object's lifetime), static constructors,
    /// and compiler-generated (implicit) creations such as <c>params</c> arrays. What remains
    /// flagged is exactly the per-call surface: method bodies, accessors, lambdas, local functions.
    /// Suppress a justified site with <c>#pragma warning disable OVERFIT001</c> + a comment.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class HeapArrayAllocationAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT001";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Heap array allocation in per-call code",
            messageFormat: "Allocates 'new {0}[]' on the heap in per-call code — use PooledBuffer<T>/PooledArray (pooled scratch), TensorStorage<T> (tensor data), or stackalloc (small fixed-size)",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Per-call heap array allocations cause GC pressure on hot paths. Rent pooled memory or stackalloc instead; one-time allocations (field initializers, constructors, static constructors) are not flagged.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule, OverfitPerfAnalysis.HotPathRule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeArrayCreation, OperationKind.ArrayCreation);
            context.RegisterOperationAction(AnalyzeCollectionExpression, OperationKind.CollectionExpression);
        }

        private static void AnalyzeArrayCreation(OperationAnalysisContext context)
        {
            var operation = (IArrayCreationOperation)context.Operation;

            if (operation.Type is not IArrayTypeSymbol arrayType)
            {
                return;
            }

            ReportIfPerCall(context, operation, arrayType.ElementType);
        }

        private static void AnalyzeCollectionExpression(OperationAnalysisContext context)
        {
            // `[a, b, c]` only allocates a heap array when its TARGET type is an array
            // (Span/ReadOnlySpan targets stackalloc or inline data — those are fine).
            // An EMPTY `[]` lowers to the cached Array.Empty<T>() — zero allocation, skip.
            var operation = (ICollectionExpressionOperation)context.Operation;

            if (operation.Type is not IArrayTypeSymbol arrayType || operation.Elements.IsEmpty)
            {
                return;
            }

            ReportIfPerCall(context, operation, arrayType.ElementType);
        }

        private static void ReportIfPerCall(OperationAnalysisContext context, IOperation operation, ITypeSymbol elementType)
        {
            // Compiler-generated creations (params arrays, query rewrites) — not the author's call to fix here.
            if (operation.IsImplicit)
            {
                return;
            }

            // Jagged arrays get the more specific OVERFIT002 (JaggedArrayAllocationAnalyzer) — don't double-report.
            if (elementType is IArrayTypeSymbol)
            {
                return;
            }

            if (IsOneTimeAllocationContext(context.ContainingSymbol))
            {
                return;
            }

            OverfitPerfAnalysis.Report(
                context,
                Rule,
                operation.Syntax.GetLocation(),
                elementType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));
        }

        /// <summary>One-time-per-lifetime contexts that may allocate freely: field and property
        /// initializers (the containing symbol is the field/property itself), any constructor
        /// (instance ctors build long-lived state like layer weights; static ctors run once).</summary>
        private static bool IsOneTimeAllocationContext(ISymbol containingSymbol)
        {
            switch (containingSymbol)
            {
                case IFieldSymbol:
                case IPropertySymbol:
                    return true;

                case IMethodSymbol method:
                    return method.MethodKind is MethodKind.Constructor or MethodKind.StaticConstructor;

                default:
                    return false;
            }
        }
    }
}
