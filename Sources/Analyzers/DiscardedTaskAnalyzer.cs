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
    /// OVERFIT046 — a task thrown away through an explicit discard, <c>_ = SomethingAsync()</c>.
    ///
    /// <para><b>This rule exists because it is a measured hole in one the repository already promotes to
    /// error.</b> <c>CS4014</c> is in <c>WarningsAsErrors</c> in <c>Directory.Build.props</c>, so an
    /// un-awaited task written as a bare statement fails the build. It does <b>not</b> fire on an explicit
    /// discard. Verified rather than assumed, by compiling both shapes in one file: <c>Work();</c> reports
    /// <c>CS4014</c> and <c>_ = Work();</c> reports nothing — one diagnostic across the two. So <c>_ =</c> is
    /// a silent, one-character opt-out of a build error, and nothing else in the tree watched it.</para>
    ///
    /// <para><b>What the discard costs.</b> The task's exceptions are never observed: a faulted
    /// fire-and-forget completes, its <c>AggregateException</c> is collected, and the caller's own logs show
    /// nothing at all. That is the failure shape the rest of this repository is organised against — a guard
    /// that reports nothing and a guard that is switched off produce identical output. A long-running one is
    /// worse than a short one, because the process keeps serving and looks healthy.</para>
    ///
    /// <para><b>The defect that motivated this rule — FIXED 2026-08-11, and worth keeping written down
    /// because the discard was not the bug, it was what the bug hid.</b>
    /// <c>Cli/GuardMetricsEndpoint.cs</c> started its whole HTTP serve loop with <c>_ = ServeAsync()</c>,
    /// and that loop's <c>catch</c> clauses were FILTERED — <c>HttpListenerException</c>,
    /// <c>ObjectDisposedException</c>, <c>IOException</c>, the transport faults. A concurrent
    /// <c>/suppressions</c> scrape reading guard state while a cycle mutated it throws
    /// <c>InvalidOperationException</c>, which matches neither filter, so it left the loop, faulted the task
    /// nobody held, and ended the metrics channel for the life of the process. An unobserved exception has
    /// not crashed anything since .NET 4.5, so the guard kept running and kept looking healthy while its own
    /// observability was gone. <b>Two silences composed</b>: the discard hid the fault, and a guard with
    /// nothing to report is indistinguishable from a guard that has stopped reporting. It now holds the task
    /// in a field, catches unfiltered, and <c>Dispose</c> reads <c>IsFaulted</c> back; the backstop it grew
    /// says out loud that the endpoint "has STOPPED serving and will not recover".</para>
    ///
    /// <para><b>The site that remains is the harder shape, and the reason it is harder is the rule's whole
    /// point.</b> At <c>Server.AspNet/OverfitAspNetServer.cs:75</c> the discard is inside a
    /// <c>CancellationToken.Register</c> callback, which takes an <c>Action</c> — there is nowhere to await
    /// and <c>async void</c> is banned by OVERFIT027, so the discard is genuinely the least-bad spelling and
    /// it keeps a pragma naming the callback signature. What it must NOT keep is its original justification:
    /// the comment there says the <c>_ =</c> is present to stop <c>CS4014</c> tripping. <b>The escape hatch
    /// had become the reason</b> — a compiler diagnostic was silenced because it was in the way, and the
    /// silencing is the thing this rule now reports. A pragma that names the callback signature is a
    /// constraint; a discard that names the diagnostic it defeated is a workaround wearing a comment.</para>
    ///
    /// <para>So this rule ships with <b>no live defect to catch</b> — every remaining site is deliberate.
    /// That is the same posture as OVERFIT027, which went to error with zero sites: the value is a tripwire
    /// on a shape that has already produced one real defect here, not a backlog to burn down.</para>
    ///
    /// <para><b>The gate is the TYPE, not the discard</b>, which is what makes the rule usable. Explicit
    /// discards are common here for a completely different reason — <c>_ = someParameter;</c> silences an
    /// unused parameter, and <c>Sources</c> holds seventeen of those (<c>GPT1Model</c>,
    /// <c>MultiHeadAttentionLayer</c>, <c>TrainableLlamaModel</c>, <c>TensorStorage</c> and others). Every
    /// one must stay silent, and a test asserts exactly that. The check reuses OVERFIT039's awaitable gate
    /// (<see cref="AwaitableType"/>) rather than a second copy, so the two rules cannot drift apart about
    /// what a task is; <c>ValueTask</c> and <c>ValueTask&lt;T&gt;</c> count, as they do there.</para>
    ///
    /// <para><b>The discard is identified semantically</b>, through <see cref="IDiscardSymbol"/>, not by
    /// matching the identifier <c>_</c>. A local genuinely named <c>_</c> is legal C# and assigning to it is
    /// an ordinary assignment that keeps the value reachable — the compiler already knows the difference, so
    /// asking it costs nothing and removes a false positive nobody would have predicted.</para>
    ///
    /// <para><b>The escape is a pragma naming the reason</b>, on the same "name the constraint" contract as
    /// OVERFIT022, OVERFIT023, OVERFIT038 and OVERFIT039. Fire-and-forget is sometimes exactly right — the
    /// two paragraphs above are the worked example of both halves, a site where it was wrong and a site
    /// where it is correct. What the rule takes away is not the ability to write one; it is the ability to
    /// write one by accident and have nobody review it.</para>
    ///
    /// <para><b>Where this rule can and cannot see, which is not the whole solution.</b> The analyzers are
    /// wired centrally in <c>Directory.Build.props</c> with an exclusion list, so OVERFIT046 never runs in
    /// <c>Analyzers</c>, <c>Tests</c>, <c>AotSmokeTest</c>, <c>Benchmarks</c>,
    /// <c>DevOnBike.Overfit.Templates</c> or <c>OverfitChat</c>. Inventoried at introduction: <b>five</b>
    /// task discards exist and <b>all five are inside projects the rule covers</b> — one in
    /// <c>Server.AspNet/OverfitAspNetServer.cs</c>, one in <c>Demo/LabLoadDriver</c> and three in
    /// <c>Demo/LabWorkload</c>'s fault-injection endpoints, which must return before the fault lands.
    /// <b>Every one of the five already observes its own exceptions</b> — the load driver's
    /// <c>SendAsync</c> is wrapped end to end in a <c>catch</c> that counts the failure, and each
    /// <c>LabWorkload</c> body logs from a <c>catch</c> (one of them carrying the note "Observed, not
    /// swallowed. The previous version's throw vanished and the endpoint stayed silent"). So all five want a
    /// pragma naming that, and none is better fixed than suppressed — though only the <c>Server.AspNet</c>
    /// one is pragma'd now; the four under <c>Demo</c> stay at warning until that directory has an owner
    /// again, which is the other half of why the severity below is not yet <c>error</c>. Nothing is hidden
    /// from the rule today, but <c>Tests</c> holds 121 discards of the unused-parameter kind and would be
    /// invisible if any of them ever became a task.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class DiscardedTaskAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT046";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "task discarded with '_ ='",
            messageFormat:
                "'_ =' throws away a {0} — CS4014 does not fire on an explicit discard, so a failure in this "
                + "call is never observed and never logged; await it, hand it to something that awaits it, or "
                + "opt out with '#pragma warning disable OVERFIT046' saying why it is fire-and-forget",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "An explicit discard silently opts out of CS4014, which is an error repo-wide. The discarded "
                + "task's exceptions are never observed, so the failure is invisible at the call site and in "
                + "the logs. Fire-and-forget is legitimate; writing it by accident is not, which is why the "
                + "escape is a pragma that states the reason.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.SimpleAssignmentExpression);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var assignment = (AssignmentExpressionSyntax)context.Node;

            // `_ = x` only. A compound assignment cannot target a discard, so SimpleAssignment is the whole
            // surface; asking the compiler for IDiscardSymbol keeps a local actually named `_` out of it.
            if (context.SemanticModel.GetSymbolInfo(assignment.Left, context.CancellationToken).Symbol
                is not IDiscardSymbol)
            {
                return;
            }

            if (!AwaitableType.IsAwaitable(
                    context.SemanticModel, assignment.Right, context.CancellationToken))
            {
                return;
            }

            var type = context.SemanticModel
                .GetTypeInfo(assignment.Right, context.CancellationToken).Type;

            context.ReportDiagnostic(Diagnostic.Create(
                Rule, assignment.GetLocation(), type?.ToDisplayString() ?? "task"));
        }
    }
}
