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
    /// OVERFIT009 — <c>.ToArray()</c> in per-call code (one of the explicit hot-path bans in
    /// Sources/Main/README.md). <c>Span.ToArray()</c>/<c>ToArray()</c> copies the whole sequence
    /// into a fresh heap array on every call — slice the existing buffer (<c>Span</c>/<c>Memory</c>),
    /// write into a caller-owned/pooled buffer, or keep the original storage. One-time contexts and
    /// the exception path are exempt, like the other per-call rules.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ToArrayAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT009";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: ".ToArray() in per-call code",
            messageFormat: "'{0}.ToArray()' copies into a fresh heap array per call — slice the existing Span/Memory, write into a caller-owned or pooled buffer, or keep the original storage",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "ToArray allocates and copies on every call. Hot paths should operate on existing buffers (Span slicing, PooledBuffer<T>). One-time contexts and exception construction are exempt.");

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

            if (operation.TargetMethod.Name != "ToArray" || operation.TargetMethod.Parameters.Length > (operation.TargetMethod.IsExtensionMethod ? 1 : 0))
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

            var receiverType = (operation.Instance ?? (operation.Arguments.Length > 0 ? operation.Arguments[0].Value : null))?.Type;
            var display = receiverType?.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat) ?? "sequence";

            OverfitPerfAnalysis.Report(context, Rule, operation.Syntax.GetLocation(), display);
        }
    }
}
