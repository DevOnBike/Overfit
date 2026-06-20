// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT011 — a custom <c>struct</c> used as a <c>Dictionary&lt;K,V&gt;</c> / <c>HashSet&lt;K&gt;</c>
    /// key without implementing <c>IEquatable&lt;K&gt;</c>. The default <c>ValueType.Equals</c>/
    /// <c>GetHashCode</c> for a struct that doesn't override them falls back to a
    /// <b>reflection-based</b>, field-by-field comparison (boxing each field) — invoked on every
    /// lookup, insert and resize. Implement <c>IEquatable&lt;K&gt;</c> (or make it a
    /// <c>record struct</c>, which the compiler implements for you), or pass an
    /// <c>IEqualityComparer&lt;K&gt;</c>. See <c>docs/performance-patterns.md</c> #46g.
    ///
    /// This is a design rule, not a per-call rule: a dictionary keyed on a bad struct is slow no
    /// matter where it is built (the common case is a field initialised in a constructor), so the
    /// one-time-context exemption does NOT apply. Passing a comparer overload silences it.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class StructDictionaryKeyAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT011";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Struct dictionary/set key without IEquatable",
            messageFormat: "Struct '{0}' is a {1} key but does not implement IEquatable<{0}> — the default struct Equals/GetHashCode is reflection-based and boxes per lookup; make it a record struct, implement IEquatable<{0}>, or pass an IEqualityComparer",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A struct key without IEquatable<T> triggers the reflection-based ValueType.Equals/GetHashCode on every dictionary/set operation. Implement IEquatable<T>, use a record struct, or supply an IEqualityComparer<T>.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule, OverfitPerfAnalysis.HotPathRule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeObjectCreation, OperationKind.ObjectCreation);
        }

        private static void AnalyzeObjectCreation(OperationAnalysisContext context)
        {
            var operation = (IObjectCreationOperation)context.Operation;

            if (operation.Type is not INamedTypeSymbol type || type.TypeArguments.Length == 0)
            {
                return;
            }

            var kind = CollectionKind(type);

            if (kind is null)
            {
                return;
            }

            // An explicit IEqualityComparer<K> overload replaces the default equality — no reflection.
            if (HasComparerParameter(operation.Constructor))
            {
                return;
            }

            var key = type.TypeArguments[0];

            if (!IsReflectionEqualityStruct(key))
            {
                return;
            }

            OverfitPerfAnalysis.Report(
                context,
                Rule,
                operation.Syntax.GetLocation(),
                key.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat),
                kind);
        }

        /// <summary>"dictionary" / "set" if the constructed type is the BCL keyed collection, else null.</summary>
        private static string? CollectionKind(INamedTypeSymbol type)
        {
            var ns = type.ContainingNamespace;

            if (ns is not { Name: "Generic", ContainingNamespace: { Name: "Collections", ContainingNamespace: { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true } } })
            {
                return null;
            }

            return type.Name switch
            {
                "Dictionary" => "dictionary",
                "HashSet" => "set",
                _ => null
            };
        }

        private static bool HasComparerParameter(IMethodSymbol? ctor)
        {
            if (ctor is null)
            {
                return false;
            }

            foreach (var p in ctor.Parameters)
            {
                if (p.Type.Name == "IEqualityComparer")
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>A user struct that will hit the reflection-based <c>ValueType</c> equality:
        /// a value type of kind Struct (not an enum), not <c>Nullable&lt;T&gt;</c>, and not already
        /// implementing <c>IEquatable&lt;self&gt;</c> (which record structs, primitives and tuples do).</summary>
        private static bool IsReflectionEqualityStruct(ITypeSymbol key)
        {
            if (key.TypeKind != TypeKind.Struct)
            {
                return false;
            }

            // Value tuples ((A, B), ValueTuple<...>) always get structural equality from ValueTuple,
            // which implements IEquatable and a real GetHashCode — never the reflection fallback.
            // (The IEquatable scan below misses them because the interface is typed on the unnamed
            // underlying tuple, so exclude them up front.)
            if (key is INamedTypeSymbol { IsTupleType: true })
            {
                return false;
            }

            if (key is INamedTypeSymbol { OriginalDefinition: { Name: "Nullable", ContainingNamespace: { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true } } })
            {
                return false;
            }

            foreach (var iface in key.AllInterfaces)
            {
                if (iface is { Name: "IEquatable", TypeArguments.Length: 1 }
                    && SymbolEqualityComparer.Default.Equals(iface.TypeArguments[0], key))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
