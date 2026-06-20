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
    /// OVERFIT020 — a method parameter typed as a primitive array (<c>float[]</c>/<c>int[]</c>/
    /// <c>byte[]</c>/…) that is only ever read/indexed and never escapes could be a
    /// <c>ReadOnlySpan&lt;T&gt;</c> (or <c>Span&lt;T&gt;</c> if it writes elements). A span parameter
    /// accepts an array, an array slice, a <c>stackalloc</c>, a pooled buffer's span or a
    /// <c>Memory&lt;T&gt;.Span</c> with no copy, and a <c>ReadOnlySpan&lt;T&gt;</c> also makes the
    /// read-only intent explicit. See <c>docs/performance-patterns.md</c> (span-friendly APIs).
    ///
    /// This is advisory (suggestion) and deliberately conservative — it does NOT escalate under
    /// <c>[OverfitHotPath]</c>, and it fires ONLY when the array provably does not escape the method:
    /// no constructors (weight arrays are stored in fields by design), no async/iterator methods (a
    /// ref struct can't cross <c>await</c>/<c>yield</c>), no public/protected surface (that change is
    /// breaking), and every use of the parameter is an index, <c>.Length</c>, <c>foreach</c>,
    /// <c>.AsSpan()</c>, or an argument to a span parameter. Any field/return/array-argument/lambda
    /// capture of the parameter suppresses the suggestion.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class ArrayParameterToSpanAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT020";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Array parameter could be a span",
            messageFormat: "Parameter '{0}' ({1}[]) is only {2} and never escapes — take a {3}<{1}> so callers can pass an array, a slice or stackalloc without copying",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A primitive-array parameter that is read/indexed and never stored, returned, captured or passed on as an array can be a ReadOnlySpan<T> (or Span<T> if it writes elements), accepting slices and stackalloc without a copy. Advisory (suggestion); fires only when the array provably does not escape.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationBlockAction(AnalyzeBlock);
        }

        private enum Usage
        {
            None,
            ReadOnly,
            Writes,
            Escape
        }

        private static void AnalyzeBlock(OperationBlockAnalysisContext context)
        {
            if (context.OwningSymbol is not IMethodSymbol method)
            {
                return;
            }

            // Ordinary methods only (skip ctors/accessors/operators/local functions), non-public surface,
            // synchronous, non-sequence-returning (iterators can't use ref structs).
            if (method.MethodKind != MethodKind.Ordinary
                || method.IsAsync
                || method.DeclaredAccessibility is Accessibility.Public or Accessibility.Protected or Accessibility.ProtectedOrInternal or Accessibility.ProtectedAndInternal
                || ReturnsSequence(method.ReturnType))
            {
                return;
            }

            foreach (var parameter in method.Parameters)
            {
                if (parameter.RefKind != RefKind.None || parameter.IsParams)
                {
                    continue;
                }

                if (parameter.Type is not IArrayTypeSymbol { Rank: 1, ElementType: { } element } || !IsBlittablePrimitive(element))
                {
                    continue;
                }

                var usage = ClassifyParameter(parameter, context.OperationBlocks);

                if (usage is Usage.None or Usage.Escape)
                {
                    continue;
                }

                var location = parameter.Locations.Length > 0 ? parameter.Locations[0] : Location.None;
                var elementName = element.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat);

                context.ReportDiagnostic(Diagnostic.Create(
                    Rule,
                    location,
                    parameter.Name,
                    elementName,
                    usage == Usage.Writes ? "read and written elementwise" : "read",
                    usage == Usage.Writes ? "Span" : "ReadOnlySpan"));
            }
        }

        private static Usage ClassifyParameter(IParameterSymbol parameter, ImmutableArray<IOperation> blocks)
        {
            var result = Usage.None;

            foreach (var block in blocks)
            {
                foreach (var operation in block.DescendantsAndSelf())
                {
                    if (operation is not IParameterReferenceOperation reference
                        || !SymbolEqualityComparer.Default.Equals(reference.Parameter, parameter))
                    {
                        continue;
                    }

                    var usage = ClassifyUsage(reference);

                    if (usage == Usage.Escape)
                    {
                        return Usage.Escape;
                    }

                    if (usage == Usage.Writes)
                    {
                        result = Usage.Writes;
                    }
                    else if (result == Usage.None)
                    {
                        result = Usage.ReadOnly;
                    }
                }
            }

            return result;
        }

        private static Usage ClassifyUsage(IParameterReferenceOperation reference)
        {
            // Captured by a nested lambda/local function → a span can't be captured → escape.
            for (var ancestor = reference.Parent; ancestor != null; ancestor = ancestor.Parent)
            {
                if (ancestor is IAnonymousFunctionOperation or ILocalFunctionOperation)
                {
                    return Usage.Escape;
                }
            }

            switch (reference.Parent)
            {
                // p[i] — write if it is the target of an assignment, else read.
                case IArrayElementReferenceOperation element when ReferenceEquals(element.ArrayReference, reference):
                    return IsAssignmentTarget(element) ? Usage.Writes : Usage.ReadOnly;

                // p.Length
                case IPropertyReferenceOperation { Property.Name: "Length" }:
                    return Usage.ReadOnly;

                // foreach (var x in p)
                case IForEachLoopOperation loop when ReferenceEquals(loop.Collection, reference):
                    return Usage.ReadOnly;

                // Implicit array→span conversion (passing p to a Span/ReadOnlySpan parameter): can't be retained.
                case IConversionOperation conversion:
                    return IsSpanType(conversion.Type) ? Usage.ReadOnly : Usage.Escape;

                // p.AsSpan()/AsMemory(...) — produces a view, the array itself is not retained.
                case IArgumentOperation { Parent: IInvocationOperation invocation }
                    when ReferenceEquals(invocation.Instance, reference) && invocation.TargetMethod.Name is "AsSpan" or "AsMemory":
                    return Usage.ReadOnly;

                case IInvocationOperation directInvocation
                    when ReferenceEquals(directInvocation.Instance, reference) && directInvocation.TargetMethod.Name is "AsSpan" or "AsMemory":
                    return Usage.ReadOnly;

                // Passed as an argument to an array/object parameter — it could be stored → escape.
                default:
                    return Usage.Escape;
            }
        }

        private static bool IsAssignmentTarget(IOperation operation)
            => operation.Parent is IAssignmentOperation assignment && ReferenceEquals(assignment.Target, operation);

        private static bool IsSpanType(ITypeSymbol? type)
            => type is INamedTypeSymbol { Name: "Span" or "ReadOnlySpan", ContainingNamespace: { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true } };

        private static bool ReturnsSequence(ITypeSymbol type)
            => type is INamedTypeSymbol { Name: "IEnumerable" or "IEnumerator" or "IAsyncEnumerable" };

        private static bool IsBlittablePrimitive(ITypeSymbol type)
        {
            switch (type.SpecialType)
            {
                case SpecialType.System_Boolean:
                case SpecialType.System_Char:
                case SpecialType.System_SByte:
                case SpecialType.System_Byte:
                case SpecialType.System_Int16:
                case SpecialType.System_UInt16:
                case SpecialType.System_Int32:
                case SpecialType.System_UInt32:
                case SpecialType.System_Int64:
                case SpecialType.System_UInt64:
                case SpecialType.System_Single:
                case SpecialType.System_Double:
                case SpecialType.System_Decimal:
                case SpecialType.System_IntPtr:
                case SpecialType.System_UIntPtr:
                    return true;

                default:
                    return false;
            }
        }
    }
}
