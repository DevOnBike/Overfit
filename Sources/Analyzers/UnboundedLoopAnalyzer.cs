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
    /// OVERFIT023 — a loop with no exit condition in its header (<c>while (true)</c>, <c>for (;;)</c>,
    /// <c>do … while (true)</c>). This is NASA's Power of 10 rule 2: every loop must have a bound a reviewer can
    /// establish without simulating the program.
    ///
    /// <para>The failure mode is the sibling of <see cref="RecursionAnalyzer"/>'s. Unbounded recursion kills the
    /// host process outright; an unbounded loop <b>hangs</b> it — which is worse to diagnose, because there is no
    /// exception, no stack trace and no log line, just a process that stops responding. Overfit runs inside
    /// someone else's application, and several of these loops sit on paths driven by external input: protobuf
    /// field scanning (ONNX), token generation, agent loops. A malformed file or a model that never emits a stop
    /// token must produce a reportable error, not a spin.</para>
    ///
    /// <para><b>What this does and does not catch.</b> It flags loop headers that are syntactically infinite —
    /// the exit lives somewhere in the body as a <c>break</c>/<c>return</c>/<c>throw</c> that the header does not
    /// promise. It does NOT attempt to prove that a conditioned loop terminates (<c>while (i &lt; n)</c> where the
    /// body forgets to advance <c>i</c> is still a hang, and no analyzer of this size will catch that). So this
    /// is a forcing function for stating the bound, not a termination prover.</para>
    ///
    /// <para><b>How to satisfy it.</b> Preferably give the loop a real bound in the header — a counter, a length,
    /// a budget. Where the shape genuinely wants <c>while (true)</c> (a worker parked on a semaphore, a scan
    /// whose terminator is data-driven), suppress the site with <c>#pragma warning disable OVERFIT023</c> and
    /// state in the comment <b>what makes it terminate and what bounds it</b>. If that sentence cannot be
    /// written truthfully, the loop is a latent hang.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class UnboundedLoopAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT023";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Loop with no exit condition in its header",
            messageFormat: "'{0}' has no exit condition in its header — give it a bound (counter/length/budget), or suppress this site with OVERFIT023 and state what terminates it",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "NASA Power of 10 rule 2: every loop needs an upper bound a reviewer can establish from the header. An unbounded loop hangs the host process with no exception and no stack trace — the hardest failure to diagnose. Where an infinite header is genuinely right, suppress the site and document the terminating condition.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(
                Analyze, SyntaxKind.WhileStatement, SyntaxKind.ForStatement, SyntaxKind.DoStatement);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var (condition, keyword, label) = context.Node switch
            {
                WhileStatementSyntax w => (w.Condition, w.WhileKeyword, "while (true)"),
                DoStatementSyntax d => (d.Condition, d.DoKeyword, "do … while (true)"),
                ForStatementSyntax f => (f.Condition, f.ForKeyword, "for (;;)"),
                _ => (null, default, string.Empty),
            };

            // `for` with an omitted condition is infinite by construction; while/do need a constant `true`.
            var infinite = condition == null
                ? context.Node.IsKind(SyntaxKind.ForStatement)
                : IsAlwaysTrue(context, condition);

            if (!infinite)
            {
                return;
            }

            // Report on the keyword only — underlining the whole loop would bury a long body in squiggle.
            context.ReportDiagnostic(Diagnostic.Create(Rule, keyword.GetLocation(), label));
        }

        /// <summary>
        /// True when the condition is a compile-time constant <c>true</c>. Uses the semantic model rather than
        /// matching the literal, so a <c>const bool Forever = true</c> loop condition is caught too.
        /// </summary>
        private static bool IsAlwaysTrue(SyntaxNodeAnalysisContext context, ExpressionSyntax condition)
        {
            var constant = context.SemanticModel.GetConstantValue(condition, context.CancellationToken);

            return constant.HasValue && constant.Value is true;
        }
    }
}
