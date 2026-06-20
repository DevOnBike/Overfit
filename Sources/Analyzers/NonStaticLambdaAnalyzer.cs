// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT019 — a lambda that captures nothing but is not declared <c>static</c>. A non-capturing
    /// lambda is already compiler-cached (no per-call allocation), so this is a safety, not an
    /// allocation, rule: marking it <c>static</c> makes the no-capture guarantee explicit and turns a
    /// future accidental capture (which would silently start allocating a closure on every call) into
    /// a compile error. See <c>docs/performance-patterns.md</c> #46h. Advisory (suggestion); does not
    /// escalate. Capturing lambdas are OVERFIT004's job, not this one's.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NonStaticLambdaAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT019";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Non-capturing lambda without static",
            messageFormat: "This lambda captures nothing — mark it 'static' so a future accidental capture (which would start allocating a closure per call) becomes a compile error",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A non-capturing lambda is compiler-cached, but without 'static' a later edit can silently introduce a capture and per-call closure allocation. 'static' makes the no-capture contract enforced. Advisory (suggestion).");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeAnonymousFunction, OperationKind.AnonymousFunction);
        }

        private static void AnalyzeAnonymousFunction(OperationAnalysisContext context)
        {
            var operation = (IAnonymousFunctionOperation)context.Operation;

            // Already static, or it genuinely captures (OVERFIT004 covers that) — nothing to suggest.
            if (HasStaticModifier(operation.Syntax) || OverfitPerfAnalysis.LambdaCapturesEnclosingState(operation))
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(Rule, operation.Syntax.GetLocation()));
        }

        private static bool HasStaticModifier(SyntaxNode syntax)
        {
            var modifiers = syntax switch
            {
                AnonymousFunctionExpressionSyntax anonymous => anonymous.Modifiers,
                _ => default
            };

            return modifiers.Any(SyntaxKind.StaticKeyword);
        }
    }
}
