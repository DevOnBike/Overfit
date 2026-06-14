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
    /// OVERFIT007 — calling a <c>params</c> method in per-call code: the compiler materialises a
    /// hidden array for the arguments on every call (an EMPTY params call uses the cached
    /// <c>Array.Empty</c> and stays silent). Add a fixed-arity overload or pass a reused buffer.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ParamsArrayCallAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT007";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "params call allocates a hidden array in per-call code",
            messageFormat: "Call to params method '{0}' allocates a hidden array per call — add a fixed-arity overload, pass an existing array, or take a ReadOnlySpan<T> params (C# 13)",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Expanded-form params calls allocate an array each time. Empty params calls (Array.Empty) and one-time contexts are not flagged.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule, OverfitPerfAnalysis.HotPathRule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeInvocation, OperationKind.Invocation);
        }

        private static void AnalyzeInvocation(OperationAnalysisContext context)
        {
            var operation = (IInvocationOperation)context.Operation;

            foreach (var argument in operation.Arguments)
            {
                if (argument.ArgumentKind != ArgumentKind.ParamArray)
                {
                    continue;
                }

                // Expanded form materialises an implicit array; empty expansions use Array.Empty (free).
                if (argument.Value is not IArrayCreationOperation { IsImplicit: true } creation ||
                    creation.Initializer is not { ElementValues.Length: > 0 })
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
                    context, Rule, operation.Syntax.GetLocation(), operation.TargetMethod.Name);
                return;
            }
        }
    }
}
