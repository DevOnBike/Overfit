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
    /// OVERFIT025 — a <c>stackalloc</c> whose size in <b>bytes</b> exceeds the configured budget, and
    /// OVERFIT026 — a <c>stackalloc</c> whose element count is not a compile-time constant.
    ///
    /// <para><b>Where the default comes from.</b> Microsoft's C# reference for <c>stackalloc</c> uses
    /// <c>const int MaxStackLimit = 1024;</c> as its worked example and adds "because the amount of memory
    /// available on the stack depends on the environment in which the code runs, be conservative when you
    /// define the actual limit value". The BCL is stricter than its own documentation:
    /// <c>System.String</c>'s manipulation paths cap at <c>StackallocCharBufferSizeLimit = 256</c> chars and
    /// <c>StackallocIntBufferSizeLimit = 128</c> ints — <b>512 bytes each</b>.</para>
    ///
    /// <para>The 512-byte default lands on the BCL's own figure, and a census of this codebase says it cuts
    /// in the right place. Of the 63 constant-size <c>stackalloc</c> sites in <c>Sources/Main</c>
    /// (2026-07-24), <b>49 are at or under 512 B</b> — 48 of them at or under 256 B, plus a single
    /// <c>int[128]</c> that matches <c>StackallocIntBufferSizeLimit</c> exactly — and <b>nothing at all</b>
    /// sits between 512 B and 1 KB. Everything above the gap is a deliberate tile or block buffer. The
    /// distribution is bimodal with an empty middle, so the threshold separates "ordinary scratch" from "a
    /// buffer someone sized on purpose and should therefore justify", and anywhere in 512…1024 B flags the
    /// same 14 sites.</para>
    ///
    /// <para><b>Sizes are counted in bytes of memory, never elements.</b> <c>char[256]</c>,
    /// <c>int[128]</c>, <c>double[64]</c> and <c>Vector256&lt;float&gt;[16]</c> all come to 512 B and are all
    /// treated identically — an element-count rule would wave through a <c>Vector512&lt;T&gt;[64]</c> at
    /// 4 KB while stopping a <c>byte[300]</c>.</para>
    ///
    /// <para><b>Why a size rule and not just CA2014.</b> The shipped analyzer for <c>stackalloc</c> (CA2014)
    /// only catches allocation inside a loop. Nothing in the standard set bounds the <i>size</i>, yet that is
    /// the failure that matters here: a <see cref="System.StackOverflowException"/> cannot be caught, cannot be
    /// logged, and takes the whole process down — in a library that other people host, on threads whose stack
    /// size this code does not choose.</para>
    ///
    /// <para><b>Configuration.</b> Set <c>overfit_max_stackalloc_bytes</c> in <c>.editorconfig</c>, scoped per
    /// directory like the rest of the OVERFIT ladder — a kernel that has measured its way to a larger buffer
    /// raises its own budget rather than the whole tree's.</para>
    ///
    /// <para><b>OVERFIT026 — a variable element count is not allowed at all.</b> <c>stackalloc T[n]</c> with a
    /// non-constant <c>n</c> defeats the entire point of a byte budget: the size cannot be read off the line,
    /// so no rule and no reviewer can bound it, and it is the shape that actually overflows in production.
    /// This repository holds a live example — <c>stackalloc float[vocab]</c>, which on a 152k-token
    /// vocabulary is <b>~608 KB</b>, over half a default thread stack, in one call.</para>
    ///
    /// <para>The documented <c>n &lt;= Limit ? stackalloc T[n] : new T[n]</c> guard is genuinely safe and is
    /// still reported, deliberately: a guard is only as good as the constant it compares against and the
    /// direction of the comparison, neither of which a mechanical check can confirm. The escape hatch is the
    /// one this repository already uses for OVERFIT022/023 — an explicit
    /// <c>#pragma warning disable OVERFIT026</c> whose comment names what bounds the length. Opting out is
    /// then a visible, reviewable act rather than a pattern the analyzer silently blesses.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class StackallocSizeAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT025";
        public const string UnboundedDiagnosticId = "OVERFIT026";

        /// <summary>
        /// Stack budget in <b>bytes of memory</b> — element count times element width, so
        /// <c>char[256]</c>, <c>int[128]</c> and <c>Vector256&lt;float&gt;[16]</c> are all exactly at it.
        /// 512 B is what the BCL's own <c>StackallocCharBufferSizeLimit</c> and
        /// <c>StackallocIntBufferSizeLimit</c> come to, and half Microsoft's documented example limit.
        /// </summary>
        public const int DefaultMaxBytes = 512;

        private const string MaxBytesOption = "overfit_max_stackalloc_bytes";

        private static readonly DiagnosticDescriptor SizeRule = new(
            DiagnosticId,
            title: "stackalloc exceeds the stack budget",
            messageFormat: "'stackalloc {0}[{1}]' takes {2} B of stack, over the {3} B budget — rent a PooledBuffer<T>, shrink the buffer, or raise 'overfit_max_stackalloc_bytes' for this directory with a reason",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A stack overflow cannot be caught or logged and terminates the host process. Microsoft's stackalloc reference uses 1024 bytes as its example limit and advises being conservative; the BCL's own string paths cap at 512. Large or long-lived scratch belongs in PooledBuffer<T>.");

        private static readonly DiagnosticDescriptor UnboundedRule = new(
            UnboundedDiagnosticId,
            title: "stackalloc element count is not a compile-time constant",
            messageFormat: "'stackalloc {0}[...]' has a variable element count — its stack cost cannot be read off the line; use a constant length, rent a PooledBuffer<T>, or opt out with '#pragma warning disable OVERFIT026' stating what bounds the length",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A stackalloc whose length comes from the input cannot be bounded by reading the code, and is the shape that actually overflows in production. Even the documented 'n <= Limit ? stackalloc : new' guard is reported: a guard is only as good as the constant it compares against, which no mechanical check can confirm. Opt out per site with an explicit pragma naming the bound.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [SizeRule, UnboundedRule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(Analyze, SyntaxKind.StackAllocArrayCreationExpression);
        }

        private static void Analyze(SyntaxNodeAnalysisContext context)
        {
            var node = (StackAllocArrayCreationExpressionSyntax)context.Node;

            if (node.Type is not ArrayTypeSyntax arrayType || arrayType.RankSpecifiers.Count == 0)
            {
                return;
            }

            var sizes = arrayType.RankSpecifiers[0].Sizes;

            if (sizes.Count != 1)
            {
                return;
            }

            var elementType = context.SemanticModel.GetTypeInfo(arrayType.ElementType, context.CancellationToken).Type;
            var elementName = arrayType.ElementType.ToString();
            var constant = context.SemanticModel.GetConstantValue(sizes[0], context.CancellationToken);

            if (!constant.HasValue || constant.Value is not int count)
            {
                // An omitted size ("stackalloc int[] { ... }") is an initializer: the length IS the element
                // list, written out in the source, so it is constant by construction and not the shape this
                // rule is about.
                if (sizes[0] is OmittedArraySizeExpressionSyntax)
                {
                    return;
                }

                context.ReportDiagnostic(Diagnostic.Create(UnboundedRule, node.GetLocation(), elementName));

                return;
            }

            var elementBytes = SizeOf(elementType);

            if (elementBytes <= 0)
            {
                // A generic or unresolved element type has no knowable width; guessing would produce noise on
                // exactly the code that is hardest to reason about.
                return;
            }

            var budget = ResolveBudget(context);
            var total = (long)count * elementBytes;

            if (total <= budget)
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(
                SizeRule,
                node.GetLocation(),
                elementName,
                count,
                total,
                budget));
        }

        private static int ResolveBudget(SyntaxNodeAnalysisContext context)
        {
            var options = context.Options.AnalyzerConfigOptionsProvider.GetOptions(context.Node.SyntaxTree);

            if (options.TryGetValue(MaxBytesOption, out var raw)
                && int.TryParse(raw, out var configured)
                && configured > 0)
            {
                return configured;
            }

            return DefaultMaxBytes;
        }

        private static int SizeOf(ITypeSymbol? type)
        {
            if (type == null)
            {
                return 0;
            }

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

            if (type.TypeKind is TypeKind.Pointer or TypeKind.FunctionPointer)
            {
                return 8;
            }

            if (type.TypeKind == TypeKind.Enum && type is INamedTypeSymbol { EnumUnderlyingType: { } underlying })
            {
                return SizeOf(underlying);
            }

            return 0;
        }
    }
}
