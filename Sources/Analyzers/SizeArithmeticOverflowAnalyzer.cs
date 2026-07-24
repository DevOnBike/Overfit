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
    /// OVERFIT028 — an array length computed by 32-bit multiplication.
    ///
    /// <para><b>Why this one is worse than an ordinary bug.</b> C# arithmetic is unchecked by default, so
    /// <c>layers * kvStride</c> does not fail when it exceeds <see cref="int.MaxValue"/> — it wraps. Two
    /// outcomes, and the survivable one is the rarer:</para>
    /// <list type="bullet">
    /// <item>wraps <b>negative</b> → <c>new T[n]</c> throws <c>OverflowException</c>. Noisy, catchable,
    /// harmless.</item>
    /// <item>wraps <b>positive</b> → an array that is silently far too small. Every subsequent write is a
    /// bounds check away from an exception in safe code — and in this library the consumers are pinned
    /// pointers and <c>Span</c>s over <c>fixed</c> buffers, where there is no bounds check at all. The result
    /// is heap corruption or an AccessViolationException: uncatchable, unlogged, and attributed to whatever
    /// code happens to run next rather than to the multiplication that caused it.</item>
    /// </list>
    ///
    /// <para><b>The fix is one cast.</b> <c>new float[(long)layers * kvStride]</c> performs the multiplication
    /// in 64 bits, so a genuine overflow becomes a clean <c>OutOfMemoryException</c> at the allocation instead
    /// of silent corruption later. It costs nothing: the widening happens at compile time and the array
    /// creation already takes a 64-bit length.</para>
    ///
    /// <para><b>Not reported</b> when the compiler has already settled the question — a constant-folded size
    /// (the compiler errors on overflow itself), an expression already evaluated as <c>long</c>, or a
    /// compilation built with <c>CheckForOverflowUnderflow</c>, where the wrap would throw anyway.</para>
    ///
    /// <para>Scope is array creation only. The same arithmetic feeding pointer offsets is the larger hazard in
    /// this codebase, but it cannot be judged syntactically — an offset is only wrong relative to a length the
    /// analyzer cannot see.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class SizeArithmeticOverflowAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT028";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Array length computed by 32-bit multiplication",
            messageFormat: "'{0}' sizes an array with 32-bit multiplication — on overflow it wraps silently to a too-small buffer, not an exception; widen one operand, e.g. '(long){1}'",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Unchecked 32-bit multiplication wraps instead of throwing. A positive wrap yields an undersized array whose overruns land in pinned/unsafe consumers as heap corruption or an uncatchable AccessViolationException, blamed on unrelated code. Casting one operand to long moves the failure to a clean OutOfMemoryException at the allocation.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.ArrayCreationExpression);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            // A compilation built /checked turns every wrap into an OverflowException, which is the outcome
            // this rule exists to restore. Nothing to say.
            if (context.Compilation.Options is CSharpCompilationOptions { CheckOverflow: true })
            {
                return;
            }

            var creation = (ArrayCreationExpressionSyntax)context.Node;

            if (creation.Type.RankSpecifiers.Count == 0)
            {
                return;
            }

            foreach (var size in creation.Type.RankSpecifiers[0].Sizes)
            {
                if (size is OmittedArraySizeExpressionSyntax)
                {
                    continue;
                }

                // The compiler folds and diagnoses constant overflow itself.
                if (context.SemanticModel.GetConstantValue(size, context.CancellationToken).HasValue)
                {
                    continue;
                }

                if (IsWideEnough(context, size))
                {
                    continue;
                }

                var product = FindNarrowProduct(context, size, size.SpanStart);
                if (product is null)
                {
                    continue;
                }

                context.ReportDiagnostic(Diagnostic.Create(
                    Rule,
                    product.GetLocation(),
                    size.ToString(),
                    product.Left.ToString()));
            }
        }

        /// <summary>
        /// Whether the nearest enclosing overflow context is <c>checked</c>. An explicit <c>unchecked</c>
        /// closer in wins, and so does running out of enclosing scopes (the compilation default).
        /// </summary>
        private static bool IsInCheckedContext(SyntaxNode node)
        {
            for (var current = node.Parent; current is not null; current = current.Parent)
            {
                if (current.IsKind(SyntaxKind.CheckedExpression) || current.IsKind(SyntaxKind.CheckedStatement))
                {
                    return true;
                }

                if (current.IsKind(SyntaxKind.UncheckedExpression) || current.IsKind(SyntaxKind.UncheckedStatement))
                {
                    return false;
                }

                if (current is MemberDeclarationSyntax)
                {
                    return false;
                }
            }

            return false;
        }

        /// <summary>True when the size is already computed in 64 bits, so the product cannot wrap.</summary>
        private static bool IsWideEnough(SyntaxNodeAnalysisContext context, ExpressionSyntax size)
        {
            var type = context.SemanticModel.GetTypeInfo(size, context.CancellationToken).Type;

            return type?.SpecialType is SpecialType.System_Int64
                or SpecialType.System_UInt64
                or SpecialType.System_IntPtr
                or SpecialType.System_UIntPtr;
        }

        /// <summary>
        /// Finds a multiplication (or left shift, which is a multiplication in disguise) evaluated in 32 bits
        /// anywhere inside the size expression, skipping any subtree already widened by a cast.
        /// </summary>
        private static BinaryExpressionSyntax? FindNarrowProduct(
            SyntaxNodeAnalysisContext context,
            SyntaxNode node,
            int _)
        {
            foreach (var descendant in node.DescendantNodesAndSelf())
            {
                if (descendant is not BinaryExpressionSyntax binary)
                {
                    continue;
                }

                if (!binary.IsKind(SyntaxKind.MultiplyExpression) && !binary.IsKind(SyntaxKind.LeftShiftExpression))
                {
                    continue;
                }

                if (IsWideEnough(context, binary))
                {
                    continue;
                }

                var type = context.SemanticModel.GetTypeInfo(binary, context.CancellationToken).Type;
                if (type?.SpecialType is not (SpecialType.System_Int32 or SpecialType.System_UInt32))
                {
                    continue;
                }

                // A local `checked(...)` or `checked { }` already turns the wrap into an OverflowException,
                // which is the outcome this rule exists to restore — the author solved it another way.
                if (IsInCheckedContext(binary))
                {
                    continue;
                }

                return binary;
            }

            return null;
        }
    }
}
