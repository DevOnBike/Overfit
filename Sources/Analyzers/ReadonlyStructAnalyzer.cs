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
    /// OVERFIT017 — a <c>struct</c> whose every instance field is already <c>readonly</c> (and which
    /// exposes no settable property) but the type itself is not declared <c>readonly struct</c>.
    /// Without the type-level <c>readonly</c>, the compiler must take a <b>defensive copy</b> every
    /// time the struct is read through a <c>readonly</c> field, an <c>in</c> parameter, or a
    /// <c>readonly</c>-context member access — re-copying data that can never change. Marking the
    /// type <c>readonly struct</c> removes every one of those copies for free. See
    /// <c>docs/performance-patterns.md</c> #2. Advisory (suggestion); does not escalate.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ReadonlyStructAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT017";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Struct could be readonly",
            messageFormat: "Struct '{0}' has only readonly fields but is not 'readonly struct' — mark it readonly to drop the defensive copies on every readonly-context access",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A struct with exclusively readonly fields and no settable members is effectively immutable; declaring it 'readonly struct' lets the compiler skip the defensive copy it otherwise makes on every readonly-context read. Advisory (suggestion).");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSymbolAction(AnalyzeNamedType, SymbolKind.NamedType);
        }

        private static void AnalyzeNamedType(SymbolAnalysisContext context)
        {
            var type = (INamedTypeSymbol)context.Symbol;

            // Plain structs only: skip classes/enums, already-readonly structs, records (own semantics),
            // and compiler-synthesised types.
            if (type.TypeKind != TypeKind.Struct || type.IsReadOnly || type.IsRecord || type.IsImplicitlyDeclared)
            {
                return;
            }

            var hasInstanceField = false;

            foreach (var member in type.GetMembers())
            {
                switch (member)
                {
                    // A settable instance property (auto or manual) means a mutable surface — not readonly-able.
                    case IPropertySymbol { IsStatic: false, SetMethod: { } }:
                        return;

                    case IFieldSymbol { IsStatic: false, IsConst: false } field:
                        if (!field.IsReadOnly)
                        {
                            return;
                        }

                        hasInstanceField = true;
                        break;
                }
            }

            // Empty/marker structs: no copies to save, and "make me readonly" is noise.
            if (!hasInstanceField)
            {
                return;
            }

            var location = type.Locations.Length > 0 ? type.Locations[0] : Location.None;

            context.ReportDiagnostic(Diagnostic.Create(Rule, location, type.Name));
        }
    }
}
