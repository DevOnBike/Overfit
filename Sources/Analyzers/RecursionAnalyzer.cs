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
    /// OVERFIT022 — direct recursion (a method that calls itself). This is NASA's Power of 10 rule 1 applied to
    /// the parts of Overfit where it actually buys something: with recursion the maximum stack depth is a
    /// function of the input, so it cannot be bounded by review, and in .NET a
    /// <c>StackOverflowException</c> <b>cannot be caught</b> — it terminates the process immediately, taking any
    /// host application down with it.
    ///
    /// <para>That failure mode is why this rule exists here rather than as a style preference. Overfit parses
    /// <b>untrusted, externally-authored binary input</b> — GGUF metadata, ONNX protobuf, safetensors headers,
    /// JSON schemas for constrained decoding. A recursive parser turns a malformed or hostile file into an
    /// uncatchable process kill, and Overfit's whole premise is being embedded in someone else's application.
    /// An explicit stack (a <c>Stack&lt;T&gt;</c> or an index-based worklist plus a depth cap that throws a
    /// normal, catchable exception) makes the bound reviewable and the failure recoverable.</para>
    ///
    /// <para><b>This detects direct recursion only</b> — a method invoking its own symbol. Mutual recursion
    /// (<c>A</c> → <c>B</c> → <c>A</c>) needs whole-compilation call-graph analysis and is NOT covered; treat
    /// that as a review item, not something the build catches. The check compares symbols, not names, so
    /// <c>Dispose()</c> calling <c>_field.Dispose()</c> or <c>Add()</c> calling <c>TensorPrimitives.Add()</c> is
    /// correctly ignored — a name-based check reports hundreds of those and is worse than no check at all.</para>
    ///
    /// <para><b>Tail-recursive shapes are still reported.</b> The .NET JIT does not guarantee tail-call
    /// optimisation, so a "tail call" is a real stack frame here.</para>
    ///
    /// <para>Severity is per-directory in <c>.editorconfig</c>, like the other OVERFIT rules: error where input
    /// is untrusted or the stack budget is tight, advisory elsewhere. Where recursion is genuinely the right
    /// tool over a small, provably-bounded structure, suppress the specific site with
    /// <c>#pragma warning disable OVERFIT022</c> plus the bound as the reason.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class RecursionAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT022";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Direct recursion",
            messageFormat: "'{0}' calls itself — recursion depth is input-dependent and a .NET stack overflow cannot be caught; use an explicit Stack<T>/worklist with a depth cap that throws a catchable exception",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "Recursion makes the maximum stack depth a function of the input rather than something reviewable, and StackOverflowException is uncatchable in .NET — it kills the host process. Prefer an explicit stack or worklist with an explicit depth limit, especially on paths that parse untrusted input (GGUF, ONNX, safetensors, JSON schema).");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeInvocation, OperationKind.Invocation);
        }

        private static void AnalyzeInvocation(OperationAnalysisContext context)
        {
            var operation = (IInvocationOperation)context.Operation;

            // The enclosing method, walking out through local functions and lambdas: a local function that
            // calls itself is just as unbounded as a method that does.
            var enclosing = context.ContainingSymbol as IMethodSymbol;

            if (enclosing is null)
            {
                return;
            }

            var target = operation.TargetMethod;

            // Compare ORIGINAL DEFINITIONS so a generic method calling itself at another type argument
            // (Foo<int> -> Foo<string>) still counts — that shape recurses through the definition even though
            // the constructed symbols differ.
            if (!SymbolEqualityComparer.Default.Equals(target.OriginalDefinition, enclosing.OriginalDefinition))
            {
                return;
            }

            // A virtual/interface call that happens to land on the same symbol is not provably a self-call:
            // the runtime target depends on the receiver's type. Only flag it when the receiver is `this`
            // (or absent, i.e. a static call), which is the shape that definitely recurses.
            if (target.IsVirtual || target.IsAbstract || target.IsOverride)
            {
                var receiver = operation.Instance;

                if (receiver is not null && receiver.Kind != OperationKind.InstanceReference)
                {
                    return;
                }
            }

            context.ReportDiagnostic(Diagnostic.Create(
                Rule,
                operation.Syntax.GetLocation(),
                enclosing.Name));
        }
    }
}
