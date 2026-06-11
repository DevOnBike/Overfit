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
    /// OVERFIT002 — jagged array allocation (<c>new T[n][]</c>) in per-call code.
    ///
    /// A jagged array is N+1 heap allocations plus a pointer-chase per row — cache-hostile and
    /// GC-heavy, which is why <c>float[][]</c> is a build ERROR in Sources/Main outright (the
    /// MSBuild guard). This rule covers the remaining element types (<c>int[][]</c>,
    /// <c>byte[][]</c>, …): allowed as one-time structures (lookup tables, model topology), but a
    /// per-call jagged allocation should be a flat <c>T[]</c> Span-sliced per row (one allocation,
    /// cache-friendly) or an Overfit buffer. Same one-time exemptions as OVERFIT001:
    /// field/property initializers, constructors, static constructors, implicit creations.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class JaggedArrayAllocationAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT002";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Jagged array allocation in per-call code",
            messageFormat: "Allocates a jagged '{0}[][]' (N+1 heap objects, pointer-chase per row) in per-call code — use a flat '{0}[]' Span-sliced per row, or PooledBuffer<T>/TensorStorage<T>",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Jagged arrays cost one allocation per row plus the outer array and defeat cache locality. Per-call code should use a flat array sliced per row; one-time structures (field initializers, constructors) are not flagged. float[][] is banned in Sources/Main entirely by the MSBuild guard.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

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

            if (operation.Type is IArrayTypeSymbol { ElementType: IArrayTypeSymbol inner })
            {
                ReportIfPerCall(context, operation, InnermostElement(inner));
            }
        }

        private static void AnalyzeCollectionExpression(OperationAnalysisContext context)
        {
            var operation = (ICollectionExpressionOperation)context.Operation;

            if (operation.Type is IArrayTypeSymbol { ElementType: IArrayTypeSymbol inner })
            {
                ReportIfPerCall(context, operation, InnermostElement(inner));
            }
        }

        private static ITypeSymbol InnermostElement(IArrayTypeSymbol array)
        {
            var current = array;

            while (current.ElementType is IArrayTypeSymbol deeper)
            {
                current = deeper;
            }

            return current.ElementType;
        }

        private static void ReportIfPerCall(OperationAnalysisContext context, IOperation operation, ITypeSymbol elementType)
        {
            if (operation.IsImplicit)
            {
                return;
            }

            if (IsOneTimeAllocationContext(context.ContainingSymbol))
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(
                Rule,
                operation.Syntax.GetLocation(),
                elementType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)));
        }

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
