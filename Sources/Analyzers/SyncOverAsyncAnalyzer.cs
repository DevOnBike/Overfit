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
    /// OVERFIT039 — blocking on a task: <c>.GetAwaiter().GetResult()</c>, <c>.Result</c>, or
    /// <c>Task.Wait()</c>.
    ///
    /// <para><b>The failure mode is a hang under load, not an error.</b> Blocking a thread on a task
    /// occupies that thread until the task finishes, while the task's own continuation may need a thread to
    /// finish at all. Under a saturated pool the two wait for each other and the process stops serving,
    /// with no exception and no stack anybody can act on — the same silent-death class as OVERFIT022
    /// (recursion), OVERFIT023 (unbounded loop) and OVERFIT027 (async void), and the reason those are in
    /// this tier.</para>
    ///
    /// <para><b>Why it exists, found rather than anticipated.</b> The xunit v3 migration on 2026-08-11
    /// brought an analyzer that flags this <i>in tests</i>, and it found seven sites. Auditing the product
    /// for the same shape then found twelve more in <c>Sources</c> — including two inside the redaction
    /// gateway's REQUEST PATH, where an upstream response body was read with
    /// <c>ReadAsStringAsync().GetAwaiter().GetResult()</c> on the thread serving the caller. Nothing in this
    /// repository was checking for it; the rule is the check.</para>
    ///
    /// <para><b>What it deliberately does NOT flag, and this gate is what makes it usable.</b>
    /// <c>SemaphoreSlim.Wait()</c>, <c>CountdownEvent.Wait()</c>, <c>ManualResetEventSlim.Wait()</c> and
    /// their kin are blocking SYNCHRONISATION PRIMITIVES, not tasks — blocking is their entire purpose and
    /// there is no continuation to deadlock against. <c>Runtime/OverfitParallel.cs</c> has three such calls
    /// in its decode loop and every one is correct; a rule that flagged them would be turned off within a
    /// day. Only a receiver whose type is <c>Task</c>, <c>Task&lt;T&gt;</c>, <c>ValueTask</c> or
    /// <c>ValueTask&lt;T&gt;</c> counts.</para>
    ///
    /// <para><b>Legitimate uses exist and take a pragma with a reason.</b> A <c>Main</c> that bridges to
    /// async, a <c>Dispose</c> that cannot be async, an entry point before any synchronisation context is
    /// installed — all are safe because no pool thread is waiting on the result. Each carries
    /// <c>#pragma warning disable OVERFIT039</c> stating which of those it is, on the same "name the
    /// constraint" contract as OVERFIT022 and OVERFIT023.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class SyncOverAsyncAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT039";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "blocking on a task",
            messageFormat:
                "'{0}' blocks a thread on a task — under a saturated thread pool the waiter and the task's "
                + "continuation deadlock and the process stops serving silently; await it, or state why "
                + "blocking is safe here with a pragma",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "Blocking on a task holds a thread until the task completes, while the task may need a "
                + "thread to complete. The result is a hang with no exception. Blocking synchronisation "
                + "primitives (SemaphoreSlim.Wait and friends) are not flagged — they are not tasks and "
                + "have no continuation to starve.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeInvocation, SyntaxKind.InvocationExpression);
            context.RegisterSyntaxNodeAction(AnalyzeMemberAccess, SyntaxKind.SimpleMemberAccessExpression);
        }

        /// <summary>Catches <c>x.Wait()</c> and <c>x.GetAwaiter().GetResult()</c>.</summary>
        private static void AnalyzeInvocation(SyntaxNodeAnalysisContext context)
        {
            var invocation = (InvocationExpressionSyntax)context.Node;

            if (invocation.Expression is not MemberAccessExpressionSyntax member)
            {
                return;
            }

            var name = member.Name.Identifier.Text;

            if (name == "Wait")
            {
                // The receiver decides. SemaphoreSlim.Wait() and CountdownEvent.Wait() are the reason this
                // rule checks the type instead of the method name.
                if (IsAwaitable(context, member.Expression))
                {
                    Report(context, invocation, "Wait()");
                }

                return;
            }

            if (name != "GetResult")
            {
                return;
            }

            // `.GetAwaiter().GetResult()` — walk one link back and ask what GetAwaiter was called on, so an
            // unrelated GetResult() on some other type is left alone.
            if (member.Expression is InvocationExpressionSyntax inner
                && inner.Expression is MemberAccessExpressionSyntax awaiter
                && awaiter.Name.Identifier.Text == "GetAwaiter"
                && IsAwaitable(context, awaiter.Expression))
            {
                Report(context, invocation, "GetAwaiter().GetResult()");
            }
        }

        /// <summary>
        /// Catches <c>x.Result</c>, but only when it is READ rather than assigned — and only on a task.
        /// Plenty of types in this tree carry an unrelated <c>Result</c> property.
        /// </summary>
        private static void AnalyzeMemberAccess(SyntaxNodeAnalysisContext context)
        {
            var member = (MemberAccessExpressionSyntax)context.Node;

            if (member.Name.Identifier.Text != "Result")
            {
                return;
            }

            // `.Result` inside `.GetAwaiter().GetResult()` chains and similar is handled above; skip the
            // case where this member access is itself the target of an invocation.
            if (member.Parent is InvocationExpressionSyntax invocation && invocation.Expression == member)
            {
                return;
            }

            if (IsAwaitable(context, member.Expression))
            {
                Report(context, member, "Result");
            }
        }

        /// <summary>
        /// Whether the expression is a task. <b>This is the whole rule.</b> Name-matching would flag every
        /// <c>SemaphoreSlim.Wait()</c> in the decode loop, and a rule with false positives on correct code
        /// gets suppressed wholesale rather than obeyed.
        /// </summary>
        private static bool IsAwaitable(SyntaxNodeAnalysisContext context, ExpressionSyntax expression)
        {
            var type = context.SemanticModel.GetTypeInfo(expression, context.CancellationToken).Type;

            if (type is null)
            {
                return false;
            }

            for (var current = type; current is not null; current = current.BaseType)
            {
                var name = current.OriginalDefinition.ToDisplayString();

                if (name is "System.Threading.Tasks.Task"
                    or "System.Threading.Tasks.Task<TResult>"
                    or "System.Threading.Tasks.ValueTask"
                    or "System.Threading.Tasks.ValueTask<TResult>")
                {
                    return true;
                }
            }

            return false;
        }

        private static void Report(SyntaxNodeAnalysisContext context, SyntaxNode node, string what)
        {
            context.ReportDiagnostic(Diagnostic.Create(Rule, node.GetLocation(), what));
        }
    }
}
