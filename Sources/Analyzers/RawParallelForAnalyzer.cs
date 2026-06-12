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
    /// OVERFIT008 — raw TPL <c>Parallel.For/ForEach/Invoke</c> in library code. Two measured reasons
    /// (docs/mnist-cnn-training-audit.md, docs/llamacpp-cpu-analysis.md):
    /// (1) raw TPL ignores <c>OverfitParallel.SuppressParallelismOnCurrentThread</c>, so inside a
    /// DataParallelTrainer replica it oversubscribes the box (R replicas × N threads — measured
    /// 1.8× slower end-to-end); (2) sustained dispatch measured 4.5× slower with ~925 KB/token of
    /// TPL allocations vs 0 B on the OverfitParallel pool. Use <c>OverfitParallel.For</c> (the
    /// suppress-aware TPL wrapper) or <c>OverfitParallel.ForDecode</c> (zero-alloc spin pool).
    /// Measured exceptions (e.g. Conv2D's dispatch, where migration cost +13% wall) keep raw TPL
    /// behind a <c>#pragma warning disable OVERFIT008</c> with the measurement cited.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class RawParallelForAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT008";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Raw Parallel.For/ForEach/Invoke — use OverfitParallel",
            messageFormat: "Raw 'Parallel.{0}' ignores SuppressParallelismOnCurrentThread (oversubscription inside data-parallel replicas) and allocates per dispatch — use OverfitParallel.For (suppress-aware) or OverfitParallel.ForDecode (zero-alloc)",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "System.Threading.Tasks.Parallel is not suppress-aware and allocates on every dispatch. Route library parallelism through OverfitParallel; keep a measured exception behind #pragma with the benchmark cited.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeInvocation, OperationKind.Invocation);
        }

        private static void AnalyzeInvocation(OperationAnalysisContext context)
        {
            var operation = (IInvocationOperation)context.Operation;
            var method = operation.TargetMethod;

            if (method.Name is not ("For" or "ForEach" or "Invoke"))
            {
                return;
            }

            // No one-time exemption on purpose: a suppress leak in a constructor breaks
            // data-parallel replicas exactly the same way.
            if (method.ContainingType is
                {
                    Name: "Parallel",
                    ContainingNamespace:
                    {
                        Name: "Tasks",
                        ContainingNamespace:
                        {
                            Name: "Threading",
                            ContainingNamespace: { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true }
                        }
                    }
                })
            {
                context.ReportDiagnostic(Diagnostic.Create(Rule, operation.Syntax.GetLocation(), method.Name));
            }
        }
    }
}
