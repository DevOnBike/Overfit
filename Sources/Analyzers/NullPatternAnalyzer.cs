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
    /// OVERFIT044 — the null patterns <c>x is null</c> and <c>x is not null</c>.
    ///
    /// <para><b>A readability decision by the project's maintainer</b>, on the same terms as OVERFIT042 and
    /// OVERFIT043: <c>== null</c> and <c>!= null</c> read better here and are what this codebase uses. 412
    /// sites at the time the decision was taken, which was known before it was taken.</para>
    ///
    /// <para><b>The two are NOT always equivalent, and this is the one place the rule could bite.</b>
    /// <c>is null</c> always tests reference (or default) equality; <c>== null</c> dispatches to a
    /// user-defined <c>operator ==</c> whenever the operand's type declares one, and no compiler warning
    /// marks the difference. A well-written operator short-circuits on a null operand, so the two agree —
    /// that is the case to expect, and it is what must be established before a site is swept rather than
    /// assumed. Where an operator does not short-circuit, the rewrite this rule asks for can change
    /// behaviour or throw. <b>When they disagree, the spelling that preserves the pattern's meaning is
    /// <c>(object?)x == null</c> or <c>ReferenceEquals(x, null)</c></b> — both bypass the operator. One of
    /// those, not the pattern, is the answer at such a site.</para>
    ///
    /// <para><b>The overload does not have to live in this repository, and that is the easy mistake.</b>
    /// Surveyed in <c>Sources</c>: <b>exactly one</b> user-defined <c>operator ==</c> exists —
    /// <c>Anomalies/Contracts/GuardCycleOutcome.cs</c>, on a value type, where a null comparison does not
    /// arise. That survey is necessary and <b>not sufficient</b>, because a REFERENCED type supplies
    /// operators too, and one already reaches a swept site: <c>Microsoft.CodeAnalysis.Location</c> declares
    /// <c>operator ==</c> and is the declared type of a local in <c>OneTopLevelTypePerFileAnalyzer</c>. It
    /// was exercised against the pattern on both a null and a non-null operand, in both directions, before
    /// that site was converted; it short-circuits, so the two agree and the swap is behaviour-preserving.
    /// <b>The check belongs to the operand's TYPE, not to the repository</b> — read the operator, or run it
    /// against the pattern, and only then swap.</para>
    ///
    /// <para>Only null patterns are flagged. <c>is string s</c>, <c>is > 0</c>, <c>is { Count: 0 }</c> and
    /// the rest of pattern matching are untouched — the objection was to this spelling of a null check, not
    /// to patterns.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class NullPatternAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT044";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "null pattern",
            messageFormat:
                "'is {0}null' is not the spelling used in this codebase — write '{1} null'",
            category: "Design",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "A readability decision by the project's maintainer. Note the one semantic difference: a "
                + "pattern bypasses a user-defined operator ==, which a comparison calls. Only one such "
                + "overload exists here and it is on a value type, so the swap is behaviour-preserving "
                + "today — re-check if that changes.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.IsPatternExpression);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var expression = (IsPatternExpressionSyntax)context.Node;
            var pattern = expression.Pattern;
            var negated = false;

            // `is not null` is a unary pattern wrapping the constant one.
            if (pattern is UnaryPatternSyntax unary && unary.OperatorToken.IsKind(SyntaxKind.NotKeyword))
            {
                pattern = unary.Pattern;
                negated = true;
            }

            if (pattern is not ConstantPatternSyntax constant
                || !constant.Expression.IsKind(SyntaxKind.NullLiteralExpression))
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(
                Rule, expression.GetLocation(), negated ? "not " : string.Empty, negated ? "!=" : "=="));
        }
    }
}
