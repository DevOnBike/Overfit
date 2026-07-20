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
    /// OVERFIT021 — <c>else</c> / <c>else if</c>. An <c>else</c> buries the primary path one level deeper than it
    /// needs to be and turns a sequence of independent conditions into a chain the reader has to hold in their
    /// head. Nearly every occurrence has a flatter form that reads better:
    ///
    /// <list type="bullet">
    ///   <item>validation or an early exit → <b>guard clause</b> (<c>if (bad) { return …; }</c> then continue at
    ///     the outer level)</item>
    ///   <item>picking one of two values → <b>ternary</b></item>
    ///   <item><c>else if</c> chain over one value → <b>switch expression</b> (also gets exhaustiveness checking)</item>
    ///   <item>branching inside a loop → <c>continue</c></item>
    /// </list>
    ///
    /// <para>This is a readability rule, not a performance one: the compiler emits the same branches either way.
    /// It is reported per <c>else</c> clause, so an <c>if / else if / else</c> chain reports twice.</para>
    ///
    /// <para><b>Severity is deliberately staged.</b> The rule was introduced against a large body of existing
    /// code, so it ships as a warning repo-wide and is escalated to <c>error</c> per directory in
    /// <c>.editorconfig</c> as each area is cleaned. That ratchet is the point — a blanket error on day one would
    /// have forced a single mechanical rewrite across ~125 files, including hot inference paths where a
    /// behaviour-preserving-but-unmeasured refactor is exactly what this repo's discipline forbids.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ElseClauseAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT021";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "else / else if",
            messageFormat: "'{0}' — invert the condition into a guard clause with an early return/continue, or use a ternary or switch expression",
            category: "Style",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "else nests the primary path and chains conditions the reader must track. Prefer a guard clause + early return, continue inside loops, a ternary for two-way value selection, or a switch expression for a chain over one value.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeElseClause, SyntaxKind.ElseClause);
        }

        private static void AnalyzeElseClause(SyntaxNodeAnalysisContext context)
        {
            var clause = (ElseClauseSyntax)context.Node;

            // `else if` is one ElseClause whose statement is another IfStatement — name it accurately so the
            // message points at the switch-expression fix rather than the guard-clause one.
            var isElseIf = clause.Statement.IsKind(SyntaxKind.IfStatement);

            // Report on the keyword only. Covering the whole clause would underline the entire branch body,
            // which in a long method is pages of squiggle for a one-token problem.
            context.ReportDiagnostic(Diagnostic.Create(
                Rule,
                clause.ElseKeyword.GetLocation(),
                isElseIf ? "else if" : "else"));
        }
    }
}
