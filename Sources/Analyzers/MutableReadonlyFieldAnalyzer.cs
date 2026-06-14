// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT018 — a <c>readonly</c> field whose type is a <b>mutable</b> struct (a non-readonly
    /// struct that carries non-readonly instance fields, e.g. an enumerator or a hand-rolled mutable
    /// value). The <c>readonly</c> modifier forces a fresh defensive copy on <i>every</i> access, and
    /// — the real footgun (<c>docs/performance-patterns.md</c> #90) — any call to a mutating member
    /// runs against that throwaway copy, so the mutation is silently lost. Either make the struct type
    /// <c>readonly struct</c> (if it is actually immutable), or drop <c>readonly</c> from the field
    /// (if you need to mutate it in place). Advisory (suggestion); does not escalate.
    ///
    /// The mutability signal is deliberately tight — the struct must expose a non-readonly instance
    /// field — so immutable BCL values with private readonly state (<c>DateTime</c>, <c>Guid</c>,
    /// <c>TimeSpan</c>, …) and the <c>Span</c>/<c>Vector</c> readonly structs never trip it.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class MutableReadonlyFieldAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT018";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "readonly field of a mutable struct",
            messageFormat: "Field '{0}' is a readonly '{1}', a mutable struct — every access defensively copies it and any mutation is silently lost; make '{1}' a readonly struct, or drop 'readonly' from the field",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A readonly field of a mutable struct is re-copied on every access and silently discards in-place mutations (the classic mutable-struct trap). Mark the struct readonly or remove readonly from the field. Advisory (suggestion).");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSymbolAction(AnalyzeField, SymbolKind.Field);
        }

        private static void AnalyzeField(SymbolAnalysisContext context)
        {
            var field = (IFieldSymbol)context.Symbol;

            // Source readonly fields only — skip consts and compiler-generated backing fields.
            if (!field.IsReadOnly || field.IsConst || field.IsImplicitlyDeclared)
            {
                return;
            }

            if (field.Type is not INamedTypeSymbol { TypeKind: TypeKind.Struct, IsReadOnly: false } structType)
            {
                return;
            }

            // Effectively-immutable structs that carry historically non-readonly fields: Nullable<T>
            // (its `value` field is not readonly) and value tuples (public mutable Item fields). Neither
            // is the #90 trap — both are used immutably — so don't flag readonly fields of them.
            if (structType.IsTupleType ||
                structType is { OriginalDefinition: { Name: "Nullable", ContainingNamespace: { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true } } })
            {
                return;
            }

            if (!HasMutableInstanceField(structType))
            {
                return;
            }

            var location = field.Locations.Length > 0 ? field.Locations[0] : Location.None;

            context.ReportDiagnostic(Diagnostic.Create(
                Rule,
                location,
                field.Name,
                structType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat)));
        }

        /// <summary>True when the struct holds at least one non-readonly instance field — the metadata
        /// signal that it carries genuinely mutable state (immutable BCL structs keep their state in
        /// private readonly fields and so are excluded).</summary>
        private static bool HasMutableInstanceField(INamedTypeSymbol structType)
        {
            foreach (var member in structType.GetMembers())
            {
                if (member is IFieldSymbol { IsStatic: false, IsConst: false, IsReadOnly: false })
                {
                    return true;
                }
            }

            return false;
        }
    }
}
