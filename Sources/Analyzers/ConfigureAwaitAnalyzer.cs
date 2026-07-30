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
    /// OVERFIT032 — an <c>await</c> in library code that does not say <c>ConfigureAwait(false)</c>.
    ///
    /// <para><b>What the omission costs, and why "the host has no context" is not an answer.</b> Without it,
    /// the continuation is posted back to whatever <see cref="System.Threading.SynchronizationContext"/> the
    /// caller happened to be on. In a console app or ASP.NET Core there is none, so nothing happens — which
    /// is exactly what makes this dangerous to leave to judgement. A library does not choose its host. The
    /// same code awaited from WPF, WinForms, MAUI or a legacy ASP.NET request marshals every continuation
    /// through a single dispatcher, and a caller that blocks on the result deadlocks outright. This project
    /// ships a WPF demo and an Android app, so the hosts that have a context are not hypothetical.</para>
    ///
    /// <para>Even where it cannot deadlock it is not free: posting a continuation costs a queue hop per
    /// suspension, and on a decode loop that suspends per token that is the kind of cost this codebase
    /// measures rather than accepts.</para>
    ///
    /// <para><b>Why an analyzer rather than review.</b> Coverage in <c>Sources/Main</c> was 17 of 19 await
    /// sites, maintained by hand — and the two gaps were found by grepping for them, not by anyone noticing.
    /// One was real. A convention that holds 89% of the time is not a convention, it is a habit, and this is
    /// what a build error is for.</para>
    ///
    /// <para><b>The shapes it must not flag, each of which cost something to learn:</b></para>
    /// <list type="bullet">
    /// <item><c>await Task.Yield()</c> — <see cref="System.Runtime.CompilerServices.YieldAwaitable"/> is not a
    /// task and has no such overload. Worth knowing that it always resumes on the captured context, so it is
    /// the one construct here that genuinely cannot be configured away.</item>
    /// <item>An expression already ending in <c>ConfigureAwait(...)</c>, in any of its forms.</item>
    /// <item>Awaiting something with no <c>ConfigureAwait</c> member at all — a custom awaitable, or
    /// <c>IAsyncEnumerable</c> in a context where the extension is not in scope. Reporting there would demand
    /// a fix that does not compile.</item>
    /// </list>
    ///
    /// <para><b>And the one it flags with a caveat in the message.</b> On an <c>await using</c>
    /// <i>declaration</i>, the obvious fix does not compile: <c>await using var s = Open().ConfigureAwait(false)</c>
    /// makes <c>s</c> a <see cref="System.Runtime.CompilerServices.ConfiguredAsyncDisposable"/> rather than
    /// the stream, so every later use of it fails. The handle and the scope have to be separated —
    /// <c>var s = Open(); await using (s.ConfigureAwait(false)) { … }</c> — and the message says so, because
    /// the first attempt at exactly this fix in this repository did not build.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ConfigureAwaitAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT032";

        private const string ConfigureAwaitName = "ConfigureAwait";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "await without ConfigureAwait(false)",
            messageFormat: "await here captures the caller's synchronization context — add ConfigureAwait(false){0}",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A library does not choose its host. Without ConfigureAwait(false) every continuation is posted back to the caller's SynchronizationContext, which is absent in a console or ASP.NET Core app and present in WPF, WinForms, MAUI and legacy ASP.NET — where a caller that blocks on the result deadlocks. It also costs a queue hop per suspension. Task.Yield() is exempt because YieldAwaitable has no such overload; so is anything already configured, and anything with no ConfigureAwait member to call. On an `await using` declaration the fix must separate the handle from the scope, because configuring the declaration changes the variable's type.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();

            context.RegisterSyntaxNodeAction(AnalyzeAwait, SyntaxKind.AwaitExpression);
            context.RegisterSyntaxNodeAction(AnalyzeUsingStatement, SyntaxKind.UsingStatement);
            context.RegisterSyntaxNodeAction(AnalyzeUsingDeclaration, SyntaxKind.LocalDeclarationStatement);
            context.RegisterSyntaxNodeAction(AnalyzeForEach, SyntaxKind.ForEachStatement);
        }

        /// <summary>Ordinary <c>await expr</c>, including the one inside an <c>await foreach</c> source.</summary>
        private static void AnalyzeAwait(SyntaxNodeAnalysisContext context)
        {
            var await = (AwaitExpressionSyntax)context.Node;

            Check(context, await.Expression, await.AwaitKeyword.GetLocation(), string.Empty);
        }

        /// <summary><c>await using (expr) { … }</c> — the scoped form, where the plain fix works.</summary>
        private static void AnalyzeUsingStatement(SyntaxNodeAnalysisContext context)
        {
            var statement = (UsingStatementSyntax)context.Node;

            if (statement.AwaitKeyword.IsKind(SyntaxKind.None) || statement.Expression is null)
            {
                return;
            }

            Check(context, statement.Expression, statement.AwaitKeyword.GetLocation(), string.Empty);
        }

        /// <summary>
        /// <c>await using var x = expr;</c> — the declaration form. Flagged, but with the caveat that the
        /// obvious fix changes the variable's type and will not compile.
        /// </summary>
        private static void AnalyzeUsingDeclaration(SyntaxNodeAnalysisContext context)
        {
            var statement = (LocalDeclarationStatementSyntax)context.Node;

            if (statement.AwaitKeyword.IsKind(SyntaxKind.None) || statement.UsingKeyword.IsKind(SyntaxKind.None))
            {
                return;
            }

            const string Caveat =
                " — on an `await using` declaration, configure the handle rather than the declaration "
                + "(`var x = Open(); await using (x.ConfigureAwait(false)) { … }`), because configuring the "
                + "declaration makes x a ConfiguredAsyncDisposable";

            foreach (var declarator in statement.Declaration.Variables)
            {
                if (declarator.Initializer?.Value is { } initialiser)
                {
                    Check(context, initialiser, statement.AwaitKeyword.GetLocation(), Caveat);
                }
            }
        }

        /// <summary><c>await foreach (… in expr)</c>.</summary>
        private static void AnalyzeForEach(SyntaxNodeAnalysisContext context)
        {
            var statement = (ForEachStatementSyntax)context.Node;

            if (statement.AwaitKeyword.IsKind(SyntaxKind.None))
            {
                return;
            }

            Check(context, statement.Expression, statement.AwaitKeyword.GetLocation(), string.Empty);
        }

        private static void Check(
            SyntaxNodeAnalysisContext context,
            ExpressionSyntax expression,
            Location location,
            string caveat)
        {
            if (IsAlreadyConfigured(expression))
            {
                return;
            }

            // Task.Yield() is the one construct that cannot be configured: YieldAwaitable is not a task and
            // exposes no such overload. Recognised by shape as well as by symbol, so the rule still behaves
            // in a file whose semantic model is incomplete.
            if (IsTaskYield(expression))
            {
                return;
            }

            // If the awaited type has no ConfigureAwait to call, reporting would demand a fix that does not
            // compile. A custom awaitable is a deliberate choice, not an oversight.
            if (!HasConfigureAwait(context, expression))
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(Rule, location, caveat));
        }

        private static bool IsAlreadyConfigured(ExpressionSyntax expression)
        {
            return expression is InvocationExpressionSyntax
            {
                Expression: MemberAccessExpressionSyntax
                {
                    Name.Identifier.ValueText: ConfigureAwaitName
                }
            };
        }

        private static bool IsTaskYield(ExpressionSyntax expression)
        {
            return expression is InvocationExpressionSyntax
            {
                ArgumentList.Arguments.Count: 0,
                Expression: MemberAccessExpressionSyntax
                {
                    Name.Identifier.ValueText: "Yield"
                }
            };
        }

        private static bool HasConfigureAwait(SyntaxNodeAnalysisContext context, ExpressionSyntax expression)
        {
            var type = context.SemanticModel.GetTypeInfo(expression, context.CancellationToken).Type;

            if (type is null || type.TypeKind == TypeKind.Error)
            {
                // An unresolved type is not evidence of anything. Staying silent keeps the rule from turning
                // an unrelated compile error into a page of spurious diagnostics.
                return false;
            }

            foreach (var member in type.GetMembers(ConfigureAwaitName))
            {
                if (member is IMethodSymbol { Parameters.Length: 1 })
                {
                    return true;
                }
            }

            // IAsyncDisposable and IAsyncEnumerable<T> get theirs from an extension method, which is not on
            // the type. Both are configurable, and both are exactly what `await using` and `await foreach`
            // operate on, so they are recognised by name.
            return Implements(type, "IAsyncDisposable") || Implements(type, "IAsyncEnumerable");
        }

        private static bool Implements(ITypeSymbol type, string name)
        {
            if (type.Name == name)
            {
                return true;
            }

            foreach (var @interface in type.AllInterfaces)
            {
                if (@interface.Name == name)
                {
                    return true;
                }
            }

            return false;
        }
    }
}
