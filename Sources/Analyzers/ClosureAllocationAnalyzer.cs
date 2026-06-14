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
    /// OVERFIT004 — delegate allocation in per-call code. Two shapes:
    /// (1) a lambda that CAPTURES enclosing locals/parameters/<c>this</c> — allocates a closure
    /// object + delegate per call (non-capturing lambdas are cached by the compiler and stay
    /// silent); (2) an instance method-group conversion — allocates a delegate per call (static
    /// method groups are cached since C# 11 and stay silent). This is exactly why the decode hot
    /// path uses <c>OverfitParallel</c> function pointers instead of delegates.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ClosureAllocationAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT004";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Closure/delegate allocation in per-call code",
            messageFormat: "{0} — allocates per call; hoist to a static non-capturing lambda, cache the delegate in a field, or use function pointers (OverfitParallel-style)",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Capturing lambdas allocate a closure + delegate on every execution of the creating statement; instance method groups allocate a delegate. Non-capturing lambdas and static method groups are compiler-cached and not flagged.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule, OverfitPerfAnalysis.HotPathRule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeDelegateCreation, OperationKind.DelegateCreation);
        }

        private static void AnalyzeDelegateCreation(OperationAnalysisContext context)
        {
            var operation = (IDelegateCreationOperation)context.Operation;

            if (OverfitPerfAnalysis.IsOneTimeAllocationContext(context.ContainingSymbol))
            {
                return;
            }

            switch (operation.Target)
            {
                case IAnonymousFunctionOperation lambda:
                    if (CapturesEnclosingState(lambda))
                    {
                        OverfitPerfAnalysis.Report(
                            context, Rule, operation.Syntax.GetLocation(), "Lambda captures enclosing state (closure)");
                    }

                    return;

                case IMethodReferenceOperation { Method.IsStatic: false } methodReference:
                    OverfitPerfAnalysis.Report(
                        context, Rule, operation.Syntax.GetLocation(),
                        $"Instance method group '{methodReference.Method.Name}' converted to a delegate");
                    return;

                default:
                    return;
            }
        }

        /// <summary>A lambda captures when it references a local/parameter declared OUTSIDE itself,
        /// or touches <c>this</c>. Symbol containment decides — references to the lambda's own
        /// parameters/locals (including nested lambdas' own state) are not captures.</summary>
        private static bool CapturesEnclosingState(IAnonymousFunctionOperation lambda)
        {
            var lambdaSymbol = lambda.Symbol;

            foreach (var descendant in lambda.Descendants())
            {
                switch (descendant)
                {
                    case IInstanceReferenceOperation { ReferenceKind: InstanceReferenceKind.ContainingTypeInstance }:
                        return true;

                    case ILocalReferenceOperation localReference:
                        if (!IsDeclaredWithin(localReference.Local.ContainingSymbol, lambdaSymbol))
                        {
                            return true;
                        }

                        break;

                    case IParameterReferenceOperation parameterReference:
                        if (!IsDeclaredWithin(parameterReference.Parameter.ContainingSymbol, lambdaSymbol))
                        {
                            return true;
                        }

                        break;
                }
            }

            return false;
        }

        private static bool IsDeclaredWithin(ISymbol declaringSymbol, IMethodSymbol lambdaSymbol)
        {
            for (ISymbol? current = declaringSymbol; current is IMethodSymbol method; current = method.ContainingSymbol)
            {
                if (SymbolEqualityComparer.Default.Equals(method, lambdaSymbol))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
