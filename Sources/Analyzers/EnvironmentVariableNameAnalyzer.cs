// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT024 — a string <b>literal</b> passed as the variable name to
    /// <c>Environment.GetEnvironmentVariable</c> / <c>SetEnvironmentVariable</c>. Every environment-variable
    /// name the engine reads belongs in <c>DevOnBike.Overfit.Runtime.OverfitEnvironment</c>, referenced as a
    /// constant from there.
    ///
    /// <para><b>Why.</b> Scattered literals drift: the read site, the docs, the <c>overfit doctor</c> output
    /// and the release notes each end up with their own spelling of the same switch, and a typo silently
    /// disables the feature instead of failing. One declaration site makes the full set of switches
    /// enumerable and reviewable — which is the point, since these are the knobs users are told to set.
    /// Tuning flags added during a perf sprint are exactly the ones that leak: six were introduced across four
    /// kernels in a single session before this rule existed.</para>
    ///
    /// <para>Only the literal is flagged; passing <c>OverfitEnvironment.Something</c> — or any other constant
    /// reference — is fine. <c>OverfitEnvironment</c> itself is the sanctioned place for the literals.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class EnvironmentVariableNameAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT024";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Environment-variable name literal — declare it in OverfitEnvironment",
            messageFormat: "Environment-variable name \"{0}\" is a literal — declare it as a constant in OverfitEnvironment and reference that: one audit point for every switch, no spelling drift between read site and docs",
            category: "Maintainability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "All OVERFIT_* (and any other) environment-variable names are centralised in DevOnBike.Overfit.Runtime.OverfitEnvironment so the complete set of tuning switches is enumerable in one file and cannot drift between the code that reads it and the documentation that advertises it.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzeInvocation, OperationKind.Invocation);
        }

        private static void AnalyzeInvocation(OperationAnalysisContext context)
        {
            var invocation = (IInvocationOperation)context.Operation;
            var method = invocation.TargetMethod;

            if (method.Name is not ("GetEnvironmentVariable" or "SetEnvironmentVariable"))
            {
                return;
            }

            if (method.ContainingType is not { Name: "Environment" } containing
                || containing.ContainingNamespace is not { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true })
            {
                return;
            }

            // OverfitEnvironment is where the literals are supposed to live.
            if (context.ContainingSymbol.ContainingType is { Name: "OverfitEnvironment" })
            {
                return;
            }

            if (invocation.Arguments.Length == 0)
            {
                return;
            }

            var name = invocation.Arguments[0].Value;

            // Unwrap an implicit conversion so a literal behind one is still seen.
            if (name is IConversionOperation conversion)
            {
                name = conversion.Operand;
            }

            // A constant reference (OverfitEnvironment.X) is an IFieldReferenceOperation with a constant value;
            // only a bare literal in source is the violation.
            if (name is not ILiteralOperation { ConstantValue: { HasValue: true, Value: string variableName } })
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(Rule, name.Syntax.GetLocation(), variableName));
        }
    }
}
