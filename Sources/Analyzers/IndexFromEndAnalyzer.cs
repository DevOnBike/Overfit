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
    /// OVERFIT042 — the index-from-end operator, <c>collection[^1]</c>.
    ///
    /// <para><b>This is a readability rule and it is owned by the maintainer, not by a measurement.</b> The
    /// project's owner finds <c>[^1]</c> hard to read and does not want it in this codebase. That is a
    /// sufficient reason on its own and is recorded as such rather than dressed up as a performance or
    /// correctness argument — <c>x[^1]</c> and <c>x[x.Length - 1]</c> compile to the same thing, and
    /// claiming otherwise would be inventing evidence for a decision that does not need any.</para>
    ///
    /// <para><b>Why a rule rather than a habit.</b> Code that its maintainer reads slowly is code that gets
    /// reviewed worse, and this repository leans hard on review for the things tests cannot hold — the
    /// copy-before-dispose ordering in <c>ValueStringBuilder.Grow</c>, the unreachable-branch invariant in
    /// <c>IncrementalDetokenizer</c>, the floor-versus-unit reasoning throughout the anomaly guard. A
    /// construct that costs a beat every time it is read taxes exactly that. The same reasoning is behind
    /// OVERFIT037 (comments in English) and OVERFIT021 (no <c>else</c>).</para>
    ///
    /// <para><b>Ranges are NOT flagged.</b> <c>span[1..]</c> and <c>span[..count]</c> are 251 sites here
    /// against 51 for <c>^</c>, they are the idiom the whole span-based codebase is written in, and they
    /// were not what was objected to. Only the from-end operator is.</para>
    ///
    /// <para>The replacement is <c>x[x.Length - 1]</c> for arrays and lists, <c>x[x.Count - 1]</c>, or a
    /// named local where the expression is long. XOR is untouched: <c>a ^ b</c> is a binary operator and a
    /// different syntax node, so <c>_checksum ^= ids[ids.Length - 1]</c> stays legal.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class IndexFromEndAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT042";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "index-from-end operator",
            messageFormat:
                "'^' indexes from the end and is not used in this codebase — write the arithmetic out, "
                + "'x[x.Length - 1]' or 'x[x.Count - 1]', or name the index in a local",
            category: "Design",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "A readability decision by the project's maintainer: the from-end operator reads slowly "
                + "here, and code that is read slowly is reviewed worse. It compiles identically to the "
                + "explicit arithmetic, so nothing but legibility is at stake. Ranges (x[1..], x[..n]) are "
                + "unaffected — they are the idiom this span-based codebase is written in.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();

            // IndexExpression is the `^x` form specifically. Binary XOR is a different node, so this cannot
            // fire on `a ^ b` — which matters, because the two share a character and one of them is in the
            // benchmark checksums.
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.IndexExpression);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            context.ReportDiagnostic(Diagnostic.Create(Rule, context.Node.GetLocation()));
        }
    }
}
