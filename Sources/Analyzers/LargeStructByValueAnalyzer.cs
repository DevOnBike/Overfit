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
    /// OVERFIT016 — a large <c>struct</c> (estimated &gt; 64 B, one cache line) passed by value.
    /// Every call copies the whole struct onto the stack; passing it as <c>in</c> hands over a
    /// readonly reference instead (no copy, and the JIT can still enregister small reads). See
    /// <c>docs/performance-patterns.md</c> #56. Heuristic and advisory — the size is estimated from
    /// the field layout (sum without padding, so it under-counts and only the clearly-large structs
    /// trip), and this rule stays a suggestion: it never escalates under <c>[OverfitHotPath]</c>.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class LargeStructByValueAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT016";

        private const int CacheLineBytes = 64;
        private const int MaxFieldDepth = 4;

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Large struct passed by value",
            messageFormat: "Parameter '{0}' is a large struct ('{1}', ~{2} B) passed by value — copied on every call; pass it as 'in {1}' for a readonly reference",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A struct larger than a cache line copies in full on each by-value pass. Use an 'in' parameter to pass a readonly reference. The size is estimated from the field layout; this rule is advisory (suggestion).");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSymbolAction(AnalyzeMethod, SymbolKind.Method);
        }

        private static void AnalyzeMethod(SymbolAnalysisContext context)
        {
            var method = (IMethodSymbol)context.Symbol;

            // Skip compiler-synthesised members and anything without source parameters of our own.
            if (method.IsImplicitlyDeclared)
            {
                return;
            }

            foreach (var parameter in method.Parameters)
            {
                // Only by-value parameters; in/ref/out already avoid the copy.
                if (parameter.RefKind != RefKind.None)
                {
                    continue;
                }

                if (parameter.Type is not INamedTypeSymbol { TypeKind: TypeKind.Struct } structType)
                {
                    continue;
                }

                if (IsIntrinsicVectorType(structType))
                {
                    continue;
                }

                var size = EstimateStructSize(structType, 0);

                if (size <= CacheLineBytes)
                {
                    continue;
                }

                var location = parameter.Locations.Length > 0 ? parameter.Locations[0] : Location.None;

                context.ReportDiagnostic(Diagnostic.Create(
                    Rule,
                    location,
                    parameter.Name,
                    structType.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat),
                    size));
            }
        }

        /// <summary>Sum of instance-field sizes (no alignment padding — a deliberate under-estimate so
        /// only clearly-large structs trip). Recursion is depth-capped against cyclic/huge layouts.</summary>
        private static int EstimateStructSize(ITypeSymbol type, int depth)
        {
            if (depth > MaxFieldDepth)
            {
                return 0;
            }

            var total = 0;

            foreach (var member in type.GetMembers())
            {
                if (member is IFieldSymbol { IsStatic: false, IsConst: false } field)
                {
                    total += EstimateFieldSize(field.Type, depth);
                }
            }

            return total;
        }

        private static int EstimateFieldSize(ITypeSymbol type, int depth)
        {
            switch (type.SpecialType)
            {
                case SpecialType.System_Boolean:
                case SpecialType.System_Byte:
                case SpecialType.System_SByte:
                    return 1;

                case SpecialType.System_Char:
                case SpecialType.System_Int16:
                case SpecialType.System_UInt16:
                    return 2;

                case SpecialType.System_Int32:
                case SpecialType.System_UInt32:
                case SpecialType.System_Single:
                    return 4;

                case SpecialType.System_Int64:
                case SpecialType.System_UInt64:
                case SpecialType.System_Double:
                case SpecialType.System_IntPtr:
                case SpecialType.System_UIntPtr:
                    return 8;

                case SpecialType.System_Decimal:
                    return 16;
            }

            if (type.IsReferenceType || type.TypeKind is TypeKind.Pointer or TypeKind.FunctionPointer)
            {
                return 8;
            }

            if (type.TypeKind == TypeKind.Enum && type is INamedTypeSymbol { EnumUnderlyingType: { } underlying })
            {
                return EstimateFieldSize(underlying, depth);
            }

            if (type.TypeKind == TypeKind.Struct)
            {
                // Nested struct — recurse; an unresolved/empty layout counts as at least one slot.
                return System.Math.Max(1, EstimateStructSize(type, depth + 1));
            }

            // Type parameters and anything unknown: assume a pointer-sized slot (under-counts large ones).
            return 8;
        }

        /// <summary>SIMD vector types (<c>Vector128/256/512</c>, <c>Vector&lt;T&gt;</c>) are register-backed
        /// and copy cheaply — flagging them as "large structs" is noise.</summary>
        private static bool IsIntrinsicVectorType(INamedTypeSymbol type)
        {
            for (var ns = type.ContainingNamespace; ns is { IsGlobalNamespace: false }; ns = ns.ContainingNamespace)
            {
                if (ns.Name is "Intrinsics" or "Numerics"
                    && ns.ContainingNamespace is { Name: "Runtime" or nameof(System) })
                {
                    return true;
                }
            }

            return false;
        }
    }
}
