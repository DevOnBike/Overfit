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
    /// OVERFIT033 — the jagged type <c>float[][]</c> itself, anywhere it is written.
    ///
    /// <para><b>Replaces the <c>BanJaggedFloatArrays</c> MSBuild task</b>, which was 20 lines of C# embedded
    /// in two csproj files and matched with a regular expression over raw text. That approach has holes this
    /// one does not: a block comment on a code line (<c>/* float[][] */</c>) fooled it, and it saw only what
    /// the regex spelled — a <c>float[][]</c> arriving as a generic argument, a type alias or an inferred
    /// local was invisible to it. This walks the syntax tree and asks the semantic model, so the rule is
    /// about the type rather than about the characters.</para>
    ///
    /// <para><b>Distinct from <see cref="JaggedArrayAllocationAnalyzer"/> (OVERFIT002), which stays.</b>
    /// That one flags <i>allocating</i> any jagged array in per-call code and is a warning with one-time
    /// exemptions. This one bans the <c>float[][]</c> <i>type</i> outright — a field, a parameter, a return
    /// type or a local — because in this codebase float rows are always either a flat <c>float[]</c>
    /// Span-sliced per row (one allocation, cache-friendly) or an Overfit buffer. Other element types
    /// (<c>int[][]</c>, <c>Parameter[][]</c>) remain allowed and are OVERFIT002's business.</para>
    ///
    /// <para>Severity is set per directory in <c>.editorconfig</c>, as with every other rule here: an error
    /// where the old MSBuild guard used to fail the build, and off elsewhere.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class JaggedFloatArrayTypeAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT033";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Jagged float[][] type",
            messageFormat: "'float[][]' is banned here — use a flat 'float[]' Span-sliced per row (one allocation, cache-friendly) or an Overfit buffer (PooledBuffer<float>, TensorStorage<float>)",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A jagged float array is one heap object per row plus the outer array, and a pointer chase per access. Every float matrix in this codebase is a flat array sliced per row or a pooled buffer. Other element types are covered by OVERFIT002 as a per-call allocation concern rather than banned outright.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();

            // The written type, wherever it appears: fields, parameters, returns, locals, generic arguments.
            context.RegisterSyntaxNodeAction(AnalyzeArrayType, SyntaxKind.ArrayType);
        }

        private static void AnalyzeArrayType(SyntaxNodeAnalysisContext context)
        {
            var syntax = (ArrayTypeSyntax)context.Node;

            // Only the OUTER type of a nested pair — `float[][]` parses as an ArrayType whose element type is
            // itself an ArrayType, and reporting both would produce two diagnostics for one mistake.
            if (syntax.Parent is ArrayTypeSyntax)
            {
                return;
            }

            var type = context.SemanticModel.GetTypeInfo(syntax, context.CancellationToken).Type;

            // Jagged at all — at least one array of arrays — and float at the bottom however deep it goes.
            //
            // The first version compared `inner.ElementType` directly and therefore missed `float[][][]`,
            // where the second level is still an array rather than the element. Caught by
            // `ANestedTypeProducesExactlyOneDiagnostic`, which is the whole argument for these tests: the
            // rule had shipped, the tree was clean, and nothing else would ever have said so.
            if (type is IArrayTypeSymbol { ElementType: IArrayTypeSymbol inner }
                && InnermostElement(inner).SpecialType == SpecialType.System_Single)
            {
                context.ReportDiagnostic(Diagnostic.Create(Rule, syntax.GetLocation()));
            }
        }

        private static ITypeSymbol InnermostElement(IArrayTypeSymbol array)
        {
            var current = array;

            while (current.ElementType is IArrayTypeSymbol deeper)
            {
                current = deeper;
            }

            return current.ElementType;
        }
    }
}
