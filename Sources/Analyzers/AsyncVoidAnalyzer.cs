// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT027 — an <c>async void</c> method or lambda.
    ///
    /// <para><b>The failure mode is a process kill, not an error.</b> An <c>async Task</c> that throws parks the
    /// exception in the returned task, where the caller can await it, catch it, log it. An <c>async void</c>
    /// has nowhere to park one: the exception is rethrown on whatever context captured the continuation — a
    /// thread-pool thread, in a library like this — and an unhandled exception there terminates the process.
    /// There is no <c>catch</c> that helps, because the frame the exception surfaces on is not the caller's.
    /// This is the same class of failure as <see cref="StackallocSizeAnalyzer"/> and the recursion and
    /// unbounded-loop rules: the host application dies with no stack trace anyone can act on.</para>
    ///
    /// <para><b>Also caught: async void lambdas</b>, which are the commoner accident. <c>Task.Run(async () =>
    /// ...)</c> is fine — that binds to <c>Func&lt;Task&gt;</c>. But an async lambda attached to a plain
    /// <c>Action</c> or an event, <c>timer.Elapsed += async (s, e) =&gt; ...</c>, is async void with the syntax
    /// hidden. The rule resolves the converted delegate type rather than trusting the shape.</para>
    ///
    /// <para><b>Zero sites at introduction (2026-07-24)</b>, across every project in the solution — which is
    /// why it goes straight to <c>error</c> in <c>Sources/Main</c> rather than through the usual ratchet, on
    /// the same criteria as OVERFIT008/015: no backlog, unconditional rule, trivial fix (return
    /// <c>Task</c>), high miss-cost.</para>
    ///
    /// <para>The one legitimate use is a framework-mandated event handler whose signature cannot return a
    /// task. Those carry <c>#pragma warning disable OVERFIT027</c> with a comment stating that the body cannot
    /// throw — usually because it is a <c>try</c>/<c>catch</c> wrapper around the real work.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class AsyncVoidAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT027";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "async void",
            messageFormat: "'{0}' is async void — an exception has no task to surface in and is rethrown on the captured context, terminating the host process; return Task instead",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "An async void method cannot report failure to its caller: the exception is rethrown on the continuation's context, and an unhandled exception there kills the process. Return Task so the caller can await, catch and log. Framework-mandated event handlers are the only exception and need an explicit pragma stating why the body cannot throw.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeMethod, SyntaxKind.MethodDeclaration);
            context.RegisterSyntaxNodeAction(
                AnalyzeLambda,
                SyntaxKind.ParenthesizedLambdaExpression,
                SyntaxKind.SimpleLambdaExpression,
                SyntaxKind.AnonymousMethodExpression);
        }

        private static void AnalyzeMethod(SyntaxNodeAnalysisContext context)
        {
            var method = (MethodDeclarationSyntax)context.Node;

            if (!method.Modifiers.Any(SyntaxKind.AsyncKeyword))
            {
                return;
            }

            if (method.ReturnType is not PredefinedTypeSyntax predefined
                || !predefined.Keyword.IsKind(SyntaxKind.VoidKeyword))
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(Rule, method.Identifier.GetLocation(), method.Identifier.Text));
        }

        private static void AnalyzeLambda(SyntaxNodeAnalysisContext context)
        {
            var node = context.Node;

            if (!HasAsyncModifier(node))
            {
                return;
            }

            // The syntax says nothing here — `async () => ...` is a Func<Task> against Task.Run and an async
            // void against an Action. Only the delegate it converts to settles it.
            if (context.SemanticModel.GetTypeInfo(node, context.CancellationToken).ConvertedType
                is not INamedTypeSymbol { DelegateInvokeMethod: { } invoke })
            {
                return;
            }

            if (!invoke.ReturnsVoid)
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(Rule, node.GetLocation(), "async lambda"));
        }

        private static bool HasAsyncModifier(SyntaxNode node)
        {
            if (node is AnonymousFunctionExpressionSyntax anonymous)
            {
                return anonymous.AsyncKeyword.IsKind(SyntaxKind.AsyncKeyword);
            }

            return false;
        }
    }
}
