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
    /// OVERFIT045 — a primary constructor on a <c>class</c> or <c>struct</c>.
    ///
    /// <para><b>A readability decision by the project's maintainer</b>, and the cheapest of the set: a
    /// handful of sites against 412 for the null patterns. Write an ordinary constructor and assign the
    /// fields, so the parameters have a visible lifetime and the reader can see what is captured.</para>
    ///
    /// <para><b>Records are deliberately NOT flagged, and this is a judgement worth stating rather than
    /// burying.</b> For a positional record the parameter list is not a constructor shorthand — it declares
    /// the members, the deconstructor and the equality contract; it IS the type. 106 files here declare
    /// records, most of them positional, and flagging them would not be applying the maintainer's rule but
    /// proposing a different codebase. If records are meant to go too, that is a separate decision with a
    /// separate number attached to it.</para>
    ///
    /// <para><b>The trap a primary constructor sets, beyond legibility.</b> A parameter captured by a method
    /// body becomes a hidden field with no declaration to read, no <c>readonly</c> to enforce and no name in
    /// the type's field list — so two things that look like the same value, the parameter and a field
    /// assigned from it, are different storage. That is invisible at the point of use, which is the property
    /// this repository's rules are generally organised against.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class PrimaryConstructorAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT045";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "primary constructor on a class or struct",
            messageFormat:
                "'{0}' declares a primary constructor — write an ordinary constructor and assign the "
                + "fields, so what is captured has a declaration to read",
            category: "Design",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "A readability decision by the project's maintainer. A captured primary-constructor "
                + "parameter is a field with no declaration, no readonly and no entry in the type's field "
                + "list. Positional records are exempt: there the parameter list declares the members and "
                + "the equality contract rather than shortening a constructor.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(
                Analyze, SyntaxKind.ClassDeclaration, SyntaxKind.StructDeclaration);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            // RecordDeclarationSyntax derives from TypeDeclarationSyntax and is not registered above, so a
            // positional record cannot reach this — but a `record class` shares the ClassDeclaration kind in
            // some shapes, so the type is checked rather than trusted to the registration.
            if (context.Node is RecordDeclarationSyntax)
            {
                return;
            }

            if (context.Node is not TypeDeclarationSyntax declaration
                || declaration.ParameterList == null)
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(
                Rule, declaration.ParameterList.GetLocation(), declaration.Identifier.Text));
        }
    }
}
