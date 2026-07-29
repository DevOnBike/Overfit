// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT031 — an <c>async</c> method whose only <c>await</c> is its final statement, so the state
    /// machine buys nothing and the task can simply be returned.
    ///
    /// <code>
    /// private async Task DoAsync()          private Task DoAsync()
    /// {                                     {
    ///     var a = "a";                          var a = "a";
    ///     await Task.CompletedTask;   →         return Task.CompletedTask;
    /// }                                     }
    /// </code>
    ///
    /// <para><b>What the keyword costs when it earns nothing.</b> <c>async</c> makes the compiler build a
    /// state machine, box it on the first suspension, allocate a builder and an <c>Action</c> continuation,
    /// and add a frame that shows up in every stack trace through the call. On a path that suspends once at
    /// the very end, all of that exists to forward a task the caller could have been handed directly.</para>
    ///
    /// <para><b>The trap this rule is built around, and why the guards are not optional.</b> Eliding is
    /// <i>wrong</i> whenever the awaited call sits inside something that must outlive it. A
    /// <c>using</c> declaration is the dangerous one, because it creates a <c>try/finally</c> that is
    /// invisible in the syntax:</para>
    /// <code>
    /// using var scope = Open();
    /// await Work(scope);      // returning this task disposes `scope` before Work has finished
    /// </code>
    /// <para>So the rule fires only when the method body has no <c>using</c> declaration or statement, no
    /// <c>try</c> around the await, exactly one <c>await</c> of its own, and an awaited type identical to the
    /// declared return type. Anything less certain is left alone — a false positive here produces a
    /// use-after-dispose, which is a far worse outcome than a missed state machine.</para>
    ///
    /// <para><b>One behavioural difference remains, by design.</b> In an <c>async</c> method every exception —
    /// including one thrown by argument validation before the await — is captured into the returned task. Once
    /// elided, those throw synchronously to the caller instead. For a caller that awaits immediately, and that
    /// is nearly all of them, the two are indistinguishable; for a caller that stores the task and awaits it
    /// later, the throw arrives earlier. Check that before applying the fix, and suppress with a reason if the
    /// deferred throw was the point.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class RedundantAsyncAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT031";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Redundant async state machine",
            messageFormat: "'{0}' is async only to await its last statement — return the task directly and drop the state machine",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "An async method whose single await is its final statement builds a state machine, a builder and a continuation to forward a task the caller could receive directly, and adds a frame to every stack trace through it. Removing async/await is safe here because the rule already excluded the cases where it is not: a using declaration or statement, a try around the await, more than one await, or an awaited type that differs from the return type. Note that exceptions thrown before the await will then surface synchronously rather than inside the returned task.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.MethodDeclaration);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var declaration = (MethodDeclarationSyntax)context.Node;

            if (!declaration.Modifiers.Any(SyntaxKind.AsyncKeyword))
            {
                return;
            }

            var awaited = FindElidableAwait(declaration);

            if (awaited is null)
            {
                return;
            }

            if (context.SemanticModel.GetDeclaredSymbol(declaration, context.CancellationToken)
                is not IMethodSymbol method)
            {
                return;
            }

            // The awaited expression has to already be exactly what the signature promises. Anything needing a
            // conversion — ValueTask against Task, a derived task type — is a rewrite, not a keyword removal.
            var awaitedType = context.SemanticModel.GetTypeInfo(Unwrap(awaited.Expression), context.CancellationToken).Type;

            if (awaitedType is null
                || awaitedType.TypeKind == TypeKind.Error
                || !SymbolEqualityComparer.Default.Equals(awaitedType, method.ReturnType))
            {
                return;
            }

            context.ReportDiagnostic(
                Diagnostic.Create(Rule, declaration.Identifier.GetLocation(), method.Name));
        }

        /// <summary>
        /// The method's sole <c>await</c> when it sits in the last statement and nothing in the body would be
        /// torn down early by returning the task; otherwise <c>null</c>.
        /// </summary>
        private static AwaitExpressionSyntax? FindElidableAwait(MethodDeclarationSyntax declaration)
        {
            var body = (SyntaxNode?)declaration.Body ?? declaration.ExpressionBody?.Expression;

            if (body is null)
            {
                return null;
            }

            var awaits = OwnAwaits(body);

            if (awaits.Count != 1)
            {
                return null;
            }

            var awaited = awaits[0];

            // An `await using` is a disposal that must follow completion, and a `using` declaration compiles
            // to a try/finally that no ancestor node reveals — both make the task unsafe to hand back.
            if (declaration.Body is not null && HasDisposalScope(declaration.Body))
            {
                return null;
            }

            if (declaration.ExpressionBody is not null)
            {
                return ReferenceEquals(declaration.ExpressionBody.Expression, awaited) ? awaited : null;
            }

            var statements = declaration.Body!.Statements;

            if (statements.Count == 0)
            {
                return null;
            }

            // Must be the whole of the final statement: `return await X;` or, for a plain Task, `await X;`.
            var last = statements[statements.Count - 1];

            var isTail = last switch
            {
                ReturnStatementSyntax { Expression: { } returned } => ReferenceEquals(returned, awaited),
                ExpressionStatementSyntax { Expression: { } expression } => ReferenceEquals(expression, awaited),
                _ => false
            };

            return isTail ? awaited : null;
        }

        /// <summary>
        /// Awaits belonging to this method, not to a lambda or local function nested inside it — those have
        /// their own state machines and say nothing about this one.
        /// </summary>
        private static List<AwaitExpressionSyntax> OwnAwaits(SyntaxNode body)
        {
            var result = new List<AwaitExpressionSyntax>();

            foreach (var node in body.DescendantNodesAndSelf(
                         descendIntoChildren: n => n is not (AnonymousFunctionExpressionSyntax or LocalFunctionStatementSyntax)))
            {
                if (node is AwaitExpressionSyntax await)
                {
                    result.Add(await);
                }
            }

            return result;
        }

        /// <summary>
        /// Whether the body holds anything whose teardown must wait for the awaited work: a <c>using</c> in
        /// either form, or a <c>try</c>. Lambdas and local functions are skipped — their scopes are their own.
        /// </summary>
        private static bool HasDisposalScope(BlockSyntax body)
        {
            foreach (var node in body.DescendantNodesAndSelf(
                         descendIntoChildren: n => n is not (AnonymousFunctionExpressionSyntax or LocalFunctionStatementSyntax)))
            {
                if (node is UsingStatementSyntax or TryStatementSyntax)
                {
                    return true;
                }

                if (node is LocalDeclarationStatementSyntax local
                    && !local.UsingKeyword.IsKind(SyntaxKind.None))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Strips a trailing <c>ConfigureAwait(...)</c> so the underlying task's type is what gets compared —
        /// the wrapper is what <c>await</c> consumes, not what a caller would be handed.
        /// </summary>
        private static ExpressionSyntax Unwrap(ExpressionSyntax expression)
        {
            if (expression is InvocationExpressionSyntax
                {
                    Expression: MemberAccessExpressionSyntax
                    {
                        Name.Identifier.ValueText: "ConfigureAwait"
                    } access
                })
            {
                return access.Expression;
            }

            return expression;
        }
    }
}
