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
    /// OVERFIT006 — string building (interpolation / concatenation) in per-call code. Usually a
    /// leftover log/debug line on a hot path — every pass allocates. Exempt: one-time contexts,
    /// the exception path (throw messages SHOULD be informative), <c>ToString</c> overrides
    /// (their job is to build a string), and compile-time-constant folds.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class HotPathStringAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT006";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "String allocation in per-call code",
            messageFormat: "Builds a string ({0}) in per-call code — move it off the hot path, gate it behind a flag, or precompute it",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "String interpolation/concatenation allocates per execution. Exempt: exception construction, ToString overrides, one-time contexts, constant folds.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeInterpolatedString, OperationKind.InterpolatedString);
            context.RegisterOperationAction(AnalyzeBinaryConcat, OperationKind.Binary);
        }

        private static void AnalyzeInterpolatedString(OperationAnalysisContext context)
        {
            ReportIfHot(context, context.Operation, "interpolation");
        }

        private static void AnalyzeBinaryConcat(OperationAnalysisContext context)
        {
            var operation = (IBinaryOperation)context.Operation;

            if (operation.OperatorKind != BinaryOperatorKind.Add || operation.Type?.SpecialType != SpecialType.System_String)
            {
                return;
            }

            // Only flag the OUTERMOST + of a concat chain (one diagnostic per expression, not per segment).
            if (operation.Parent is IBinaryOperation { OperatorKind: BinaryOperatorKind.Add, Type.SpecialType: SpecialType.System_String })
            {
                return;
            }

            ReportIfHot(context, operation, "concatenation");
        }

        private static void ReportIfHot(OperationAnalysisContext context, IOperation operation, string kind)
        {
            if (operation.ConstantValue.HasValue)
            {
                return;
            }

            if (OverfitPerfAnalysis.IsOneTimeAllocationContext(context.ContainingSymbol))
            {
                return;
            }

            if (IsToStringOverride(context.ContainingSymbol))
            {
                return;
            }

            if (OverfitPerfAnalysis.IsOnExceptionPath(operation))
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(Rule, operation.Syntax.GetLocation(), kind));
        }

        private static bool IsToStringOverride(ISymbol containingSymbol)
        {
            return containingSymbol is IMethodSymbol { Name: nameof(ToString) };
        }
    }
}
