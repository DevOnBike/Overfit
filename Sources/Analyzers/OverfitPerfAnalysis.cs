// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>Shared predicates for the OVERFIT performance rules — one noise policy, applied
    /// uniformly: one-time-per-lifetime contexts may allocate, exceptional paths may build strings.</summary>
    internal static class OverfitPerfAnalysis
    {
        /// <summary>OVERFIT900 — a per-call performance rule fired inside a member (or type) marked
        /// <c>[OverfitHotPath]</c>, which pins that code to the zero-allocation contract regardless of
        /// directory. The per-call rule escalates from its configured severity to this hard error; the
        /// underlying rule id is carried in the message.</summary>
        internal static readonly DiagnosticDescriptor HotPathRule = new(
            "OVERFIT900",
            title: "Performance-rule violation inside an [OverfitHotPath] member",
            messageFormat: "{0} fired inside an [OverfitHotPath] member — this path is declared zero-allocation, so the rule is an error here: fix the violation, or remove the attribute / suppress this site with a reason",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "[OverfitHotPath] escalates every per-call OVERFIT performance rule to a build error within the marked method, property or type — the per-member form of the per-directory editorconfig ratchet. The message names the underlying rule id.");

        /// <summary>Report a per-call diagnostic, escalating it to <see cref="HotPathRule"/> (error)
        /// when the surrounding member or type is marked <c>[OverfitHotPath]</c>, else reporting the
        /// rule itself at its configured severity. Every analyzer that calls this must list
        /// <see cref="HotPathRule"/> in its <c>SupportedDiagnostics</c>.</summary>
        internal static void Report(OperationAnalysisContext context, DiagnosticDescriptor rule, Location location, params object[] messageArgs)
        {
            if (IsInHotPath(context.ContainingSymbol))
            {
                context.ReportDiagnostic(Diagnostic.Create(HotPathRule, location, rule.Id));
            }
            else
            {
                context.ReportDiagnostic(Diagnostic.Create(rule, location, messageArgs));
            }
        }

        /// <summary>True when the operation lives in a member, accessor, or (containing) type carrying
        /// the <c>[OverfitHotPath]</c> marker — walks the symbol chain up to, but not including, the
        /// namespace, and checks an accessor's associated property as well.</summary>
        internal static bool IsInHotPath(ISymbol? symbol)
        {
            for (var current = symbol; current is not null and not INamespaceSymbol; current = current.ContainingSymbol)
            {
                if (HasHotPathAttribute(current))
                {
                    return true;
                }

                if (current is IMethodSymbol { AssociatedSymbol: { } associated } && HasHotPathAttribute(associated))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool HasHotPathAttribute(ISymbol symbol)
        {
            foreach (var attribute in symbol.GetAttributes())
            {
                // DevOnBike.Overfit.Diagnostics.OverfitHotPathAttribute — match by name + namespace chain.
                if (attribute.AttributeClass is
                    {
                        Name: "OverfitHotPathAttribute",
                        ContainingNamespace: { Name: "Diagnostics", ContainingNamespace: { Name: "Overfit", ContainingNamespace: { Name: "DevOnBike", ContainingNamespace.IsGlobalNamespace: true } } }
                    })
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>Field/property initializers and constructors (instance + static) allocate once
        /// per lifetime — exempt from per-call rules.</summary>
        internal static bool IsOneTimeAllocationContext(ISymbol containingSymbol)
        {
            switch (containingSymbol)
            {
                case IFieldSymbol:
                case IPropertySymbol:
                    return true;

                case IMethodSymbol method:
                    return method.MethodKind is MethodKind.Constructor or MethodKind.StaticConstructor;

                default:
                    return false;
            }
        }

        /// <summary>True when the operation feeds an exception being constructed or thrown — the
        /// exceptional path is allowed to allocate (messages, argument formatting).</summary>
        internal static bool IsOnExceptionPath(IOperation operation)
        {
            for (var current = operation.Parent; current != null; current = current.Parent)
            {
                if (current is IThrowOperation)
                {
                    return true;
                }

                if (current is IObjectCreationOperation { Type: { } created } && DerivesFromException(created))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool DerivesFromException(ITypeSymbol type)
        {
            for (ITypeSymbol? current = type; current != null; current = current.BaseType)
            {
                if (current is { Name: nameof(Exception), ContainingNamespace: { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true } })
                {
                    return true;
                }
            }

            return false;
        }
    }
}
