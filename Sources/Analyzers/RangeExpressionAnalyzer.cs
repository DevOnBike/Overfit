// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT043 — the range operator, <c>span[1..]</c> / <c>span[..count]</c> / <c>span[a..b]</c>.
    ///
    /// <para><b>A readability decision by the project's maintainer</b>, the sibling of OVERFIT042 and
    /// recorded on the same terms: the range operator reads slowly here and is not wanted, and the honest
    /// justification is that preference rather than an invented measurement. <c>span[1..]</c> and
    /// <c>span.Slice(1)</c> compile to the same call.</para>
    ///
    /// <para><b>This is the expensive one and the number was known before the decision.</b> 177 sites in
    /// <c>Sources</c> at the time it was taken, against 52 for the from-end operator — the range form is
    /// what most of this span-based codebase is written in. <c>Slice</c> is the replacement everywhere:
    /// <c>x[a..b]</c> becomes <c>x.Slice(a, b - a)</c>, <c>x[a..]</c> becomes <c>x.Slice(a)</c>, and
    /// <c>x[..b]</c> becomes <c>x.Slice(0, b)</c> — the last of which is the one worth writing out, because
    /// it is where the reader most often has to stop and work out which end is which.</para>
    ///
    /// <para><b>Strings take <c>Substring</c>, and arrays need <c>AsSpan</c> first</b> — an array has no
    /// <c>Slice</c>, so <c>array[1..]</c> becomes <c>array.AsSpan(1)</c> when a span will do and
    /// <c>array.Skip</c> is unavailable here because LINQ is banned in <c>Sources/Main</c>. That difference
    /// is why this cannot be a blind find-and-replace.</para>
    ///
    /// <para><b><c>Sources/Analyzers</c> cannot accrue a site for this rule</b>, and that is a structural
    /// guarantee rather than an observation about today's code. It targets <c>netstandard2.0</c> — alone in
    /// this solution — whose reference set contains neither <c>System.Index</c> nor <c>System.Range</c>, so
    /// a range expression does not compile there at all; verified by compiling <c>s[1..]</c>, which fails
    /// with two <c>CS0518</c> errors naming both missing predefined types. <c>LangVersion</c> is
    /// <c>latest</c>, so the compiler accepts the syntax and then fails on the absent supporting types,
    /// which is why the error does not mention the target framework. "Zero sites" and "cannot have sites"
    /// are the same output and different facts; this is the second.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class RangeExpressionAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT043";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "range operator",
            messageFormat:
                "'..' slices by range and is not used in this codebase — use Slice(start, length), "
                + "Slice(start), Substring for strings, or AsSpan(start) for arrays",
            category: "Design",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "A readability decision by the project's maintainer. The range operator compiles to the "
                + "same Slice call, so only legibility is at stake — and the explicit form names which end "
                + "is which at the point where the reader is asking.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.RangeExpression);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            context.ReportDiagnostic(Diagnostic.Create(Rule, context.Node.GetLocation()));
        }
    }
}
