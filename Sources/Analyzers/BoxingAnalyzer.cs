// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT003 — boxing conversion in per-call code: a value type converted to
    /// <c>object</c>/interface allocates a box on every conversion. In math code this hides in
    /// <c>string.Format</c> args, non-generic collections, and interface dispatch on structs.
    /// Exempt: one-time contexts and the exception path (building a throw message may box).
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class BoxingAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT003";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Boxing conversion in per-call code",
            messageFormat: "Boxes '{0}' to '{1}' — every conversion allocates; use a generic API, a struct-constrained generic, or restructure to avoid the object/interface hop",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Boxing allocates a heap object per conversion and costs an indirection on every later use. Exempt: one-time contexts (field initializers, constructors) and exception construction.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeConversion, OperationKind.Conversion);
        }

        private static void AnalyzeConversion(OperationAnalysisContext context)
        {
            var operation = (IConversionOperation)context.Operation;

            if (!operation.GetConversion().IsBoxing)
            {
                return;
            }

            // Constant boxes (e.g. a const enum in an attribute-ish spot) and compiler plumbing are noise.
            if (operation.IsImplicit && operation.ConstantValue.HasValue)
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

            var from = operation.Operand.Type?.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat) ?? "?";
            var to = operation.Type?.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat) ?? "?";
            context.ReportDiagnostic(Diagnostic.Create(Rule, operation.Syntax.GetLocation(), from, to));
        }
    }
}
