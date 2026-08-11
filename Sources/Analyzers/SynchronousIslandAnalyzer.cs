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
    /// OVERFIT040 — a synchronous method whose body calls APIs that have asynchronous siblings: the method
    /// itself is the thing to change, not the call.
    ///
    /// <para><b>This is the gap CA1849 cannot see, and it was found by measurement.</b> CA1849 ("call async
    /// methods when in an async method") tests whether the ENCLOSING SYMBOL returns a task. A synchronous
    /// helper reached from async code therefore never trips it, however much I/O it does — and that helper
    /// is exactly where a synchronous island hides. Audited 2026-08-11 across `Sources`: CA1849 produced 8
    /// hits and none of them real, while the two costly sites it structurally could not reach were a sink
    /// holding a Kestrel request thread for a whole generation and a file store writing on a pool thread
    /// once per guard cycle.</para>
    ///
    /// <para><b>The report lands on the method, not on the call, and that is the point.</b> Swapping one
    /// call for its async sibling inside a synchronous method is impossible; the fix is to make the method
    /// return a task and let its callers await it. That is a refactor with a blast radius, which is why the
    /// diagnostic names the method and leaves the decision to a person.</para>
    ///
    /// <para><b>The cost this catches is a held thread, not a deadlock.</b> Blocking ON a task is
    /// <see cref="SyncOverAsyncAnalyzer"/> (OVERFIT039) and is the more dangerous shape — it can starve the
    /// pool into a stall. A synchronous island merely occupies a thread for the duration of the I/O. The two
    /// are separate rules because they call for different fixes and carry different urgency.</para>
    ///
    /// <para><b>Deliberately synchronous designs exist here and take a pragma with a reason.</b> The
    /// redaction gateway's request path is synchronous ON PURPOSE — <c>HttpClient.Send</c>,
    /// <c>AllowSynchronousIO</c>, a terminal middleware returning <c>Task.CompletedTask</c> — because the
    /// exchange it drives streams tokens from a synchronous callback. A rule that could not be told
    /// "this one is a decision" would be wrong there, so <c>#pragma warning disable OVERFIT040</c> with a
    /// stated reason is part of the contract, as with OVERFIT022, OVERFIT023 and OVERFIT039.</para>
    ///
    /// <para><b>Console writers are excluded</b>, measured rather than assumed: <c>Console.Out</c> and
    /// <c>Console.Error</c> are <c>TextWriter.Synchronized</c> wrappers whose <c>WriteLineAsync</c> performs
    /// the same synchronous write and returns a completed task, so "fixing" one is a literal no-op. Eight of
    /// the eight CA1849 hits in this repository were exactly that, which is how the exclusion earned its
    /// place.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class SynchronousIslandAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT040";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "synchronous method doing work that has an asynchronous form",
            messageFormat:
                "'{0}' is synchronous but calls '{1}', which has an asynchronous sibling '{1}Async' — a "
                + "synchronous method reached from async code holds a thread for the whole operation; make "
                + "the method return a task, or state with a pragma why it is synchronous by design",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "A synchronous helper called from asynchronous code is invisible to CA1849, which only "
                + "inspects methods that return a task. The fix is the method, not the call: one "
                + "synchronous call cannot be swapped for its async sibling without making the whole method "
                + "asynchronous.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeMethod, SyntaxKind.MethodDeclaration);
        }

        private static void AnalyzeMethod(SyntaxNodeAnalysisContext context)
        {
            var method = (MethodDeclarationSyntax)context.Node;

            if (method.Body == null && method.ExpressionBody == null)
            {
                return;
            }

            // Already asynchronous, or already returning a task: that is CA1849's ground and reporting it
            // here would double up on a rule the SDK ships.
            if (method.Modifiers.Any(SyntaxKind.AsyncKeyword))
            {
                return;
            }

            var declared = context.SemanticModel.GetDeclaredSymbol(method, context.CancellationToken);

            if (declared == null || ReturnsAwaitable(declared.ReturnType))
            {
                return;
            }

            SyntaxNode body = method.Body ?? (SyntaxNode?)method.ExpressionBody ?? method;

            foreach (var node in body.DescendantNodes())
            {
                if (node is not InvocationExpressionSyntax invocation)
                {
                    continue;
                }

                // A lambda inside this method has its own body and its own asynchrony; judging it as part
                // of the enclosing method would blame the wrong declaration.
                if (IsInsideNestedFunction(node, body))
                {
                    continue;
                }

                if (context.SemanticModel.GetSymbolInfo(invocation, context.CancellationToken).Symbol
                    is not IMethodSymbol called)
                {
                    continue;
                }

                if (!HasAsyncSibling(called) || IsExcluded(called))
                {
                    continue;
                }

                context.ReportDiagnostic(Diagnostic.Create(
                    Rule, method.Identifier.GetLocation(), method.Identifier.Text, called.Name));

                // One report per method: the diagnostic is about the method's shape, and a dozen copies of
                // it on one declaration is noise that gets a rule suppressed rather than obeyed.
                return;
            }
        }

        private static bool ReturnsAwaitable(ITypeSymbol type)
        {
            var name = type.OriginalDefinition.ToDisplayString();

            return name is "System.Threading.Tasks.Task"
                or "System.Threading.Tasks.Task<TResult>"
                or "System.Threading.Tasks.ValueTask"
                or "System.Threading.Tasks.ValueTask<TResult>"
                or "System.Collections.Generic.IAsyncEnumerable<T>";
        }

        /// <summary>
        /// Whether the called method's own type offers a genuine asynchronous version of THIS call.
        ///
        /// <para><b>Name matching alone was measured and rejected.</b> The first version asked only whether
        /// <c>{Name}Async</c> existed and produced 155 hits, most of them wrong: <c>JsonSerializer.Serialize</c>
        /// has a <c>SerializeAsync</c>, but one returns a string and the other writes to a stream;
        /// <c>SHA256.HashData</c> has a <c>HashDataAsync</c> that takes a <c>Stream</c>. Those are different
        /// operations that share a prefix, and a rule that cannot tell them apart is a rule nobody keeps on.
        /// </para>
        ///
        /// <para>So the sibling must also ACCEPT THE SAME ARGUMENTS — allowing one extra
        /// <see cref="System.Threading.CancellationToken"/>, which is how the BCL spells its async pairs —
        /// and return a task of the same thing. That is what makes <c>File.ReadAllText(path)</c> /
        /// <c>ReadAllTextAsync(path, ct)</c> a pair and <c>Serialize</c> / <c>SerializeAsync</c> not one.</para>
        /// </summary>
        private static bool HasAsyncSibling(IMethodSymbol called)
        {
            if (called.Name.EndsWith("Async", System.StringComparison.Ordinal))
            {
                return false;
            }

            foreach (var member in called.ContainingType.GetMembers(called.Name + "Async"))
            {
                if (member is IMethodSymbol sibling
                    && ReturnsAwaitable(sibling.ReturnType)
                    && ParametersMatch(called, sibling)
                    && ResultMatches(called, sibling))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>Same parameters, optionally plus a trailing cancellation token.</summary>
        private static bool ParametersMatch(IMethodSymbol sync, IMethodSymbol async)
        {
            var syncParameters = sync.Parameters;
            var asyncParameters = async.Parameters;
            var extra = asyncParameters.Length - syncParameters.Length;

            if (extra is < 0 or > 1)
            {
                return false;
            }

            if (extra == 1
                && asyncParameters[asyncParameters.Length - 1].Type.ToDisplayString()
                    != "System.Threading.CancellationToken")
            {
                return false;
            }

            for (var i = 0; i < syncParameters.Length; i++)
            {
                if (!SymbolEqualityComparer.Default.Equals(syncParameters[i].Type, asyncParameters[i].Type))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// A void method pairs with <c>Task</c>; a method returning <c>T</c> pairs with <c>Task&lt;T&gt;</c>.
        /// Anything else is a different operation — which is what separates a real async sibling from a
        /// method that merely starts with the same word.
        /// </summary>
        private static bool ResultMatches(IMethodSymbol sync, IMethodSymbol async)
        {
            if (async.ReturnType is not INamedTypeSymbol returned)
            {
                return false;
            }

            if (sync.ReturnsVoid)
            {
                return returned.TypeArguments.Length == 0;
            }

            return returned.TypeArguments.Length == 1
                && SymbolEqualityComparer.Default.Equals(returned.TypeArguments[0], sync.ReturnType);
        }

        /// <summary>
        /// Console writers only. <c>Console.Out</c> and <c>Console.Error</c> are synchronised wrappers whose
        /// async members write synchronously and hand back a completed task, so switching changes nothing
        /// but the syntax — measured, and the reason eight of eight CA1849 hits here were noise.
        /// </summary>
        private static bool IsExcluded(IMethodSymbol called)
        {
            // Disposal is IDisposable versus IAsyncDisposable — a lifetime contract, not a slow operation,
            // and turning `Dispose` into `DisposeAsync` is a different decision with its own rule.
            if (called.Name is "Dispose")
            {
                return true;
            }

            return called.ContainingType.ToDisplayString() is "System.Console";
        }

        private static bool IsInsideNestedFunction(SyntaxNode node, SyntaxNode body)
        {
            for (var current = node.Parent; current != null && current != body; current = current.Parent)
            {
                if (current is LambdaExpressionSyntax
                    or AnonymousMethodExpressionSyntax
                    or LocalFunctionStatementSyntax)
                {
                    return true;
                }
            }

            return false;
        }
    }
}
