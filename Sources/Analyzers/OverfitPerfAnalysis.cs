// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Operations;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>Shared predicates for the OVERFIT performance rules — one noise policy, applied
    /// uniformly: one-time-per-lifetime contexts may allocate, exceptional paths may build strings.</summary>
    internal static class OverfitPerfAnalysis
    {
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
