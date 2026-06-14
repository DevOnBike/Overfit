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
    /// OVERFIT014 — case-insensitive string comparison written as
    /// <c>a.ToLower() == b.ToLower()</c> (or <c>ToUpper</c>/<c>…Invariant</c>). Each
    /// <c>ToLower()</c> allocates a whole new string just to throw it away after the compare —
    /// two transient allocations per comparison, plus the culture-dependent default of the
    /// non-Invariant overloads. Use <c>string.Equals(a, b, StringComparison.OrdinalIgnoreCase)</c>,
    /// which compares in place with zero allocation. See <c>docs/performance-patterns.md</c> #46o.
    ///
    /// One-time contexts and the exception path are exempt, like the other per-call rules.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class CaseFoldComparisonAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT014";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Case-folding string equality allocates",
            messageFormat: "'{0}()' inside an equality comparison allocates a throwaway string per call — use string.Equals(a, b, StringComparison.OrdinalIgnoreCase)",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Comparing strings via ToLower()/ToUpper() allocates a new string on each side. Use string.Equals with StringComparison.OrdinalIgnoreCase for an allocation-free, culture-explicit comparison. One-time contexts and the exception path are exempt.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule, OverfitPerfAnalysis.HotPathRule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeBinary, OperationKind.Binary);
        }

        private static void AnalyzeBinary(OperationAnalysisContext context)
        {
            var operation = (IBinaryOperation)context.Operation;

            if (operation.OperatorKind is not (BinaryOperatorKind.Equals or BinaryOperatorKind.NotEquals))
            {
                return;
            }

            var offender = CaseFoldCall(operation.LeftOperand) ?? CaseFoldCall(operation.RightOperand);

            if (offender is null)
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

            OverfitPerfAnalysis.Report(
                context,
                Rule,
                operation.Syntax.GetLocation(),
                offender.Name);
        }

        /// <summary>The <c>System.String</c> case-folding method invoked by <paramref name="operand"/>
        /// (after unwrapping conversions), or null if it is not such a call.</summary>
        private static IMethodSymbol? CaseFoldCall(IOperation operand)
        {
            while (operand is IConversionOperation conversion)
            {
                operand = conversion.Operand;
            }

            if (operand is not IInvocationOperation { TargetMethod: { } method })
            {
                return null;
            }

            if (method.Name is not ("ToLower" or "ToLowerInvariant" or "ToUpper" or "ToUpperInvariant"))
            {
                return null;
            }

            return method.ContainingType is { SpecialType: SpecialType.System_String } ? method : null;
        }
    }
}
