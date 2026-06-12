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
    /// OVERFIT015 — direct <c>Xxx.IsSupported</c> on a hardware-intrinsics class
    /// (<c>System.Runtime.Intrinsics.*</c>) outside <c>CpuFeatures</c>. ISA gating goes through
    /// the single facade <c>DevOnBike.Overfit.Intrinsics.CpuFeatures</c> (<c>HasAvx2</c>,
    /// <c>HasFma</c>, <c>HasAvx2Fma</c>, …): one audit point for which ISAs the library uses,
    /// composed flags live in one place, and the <c>static readonly bool</c> fields are folded to
    /// constants by the tier-1 JIT exactly like the raw <c>IsSupported</c> intrinsic — zero cost.
    /// The only sanctioned reader of the raw properties is <c>CpuFeatures</c> itself.
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class CpuFeaturesGateAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT015";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Direct intrinsics IsSupported — use CpuFeatures",
            messageFormat: "Direct '{0}.IsSupported' — gate ISA paths through CpuFeatures (CpuFeatures.Has{0}): one audit point, composed flags, same JIT constant-folding",
            category: "Performance",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "ISA capability checks are centralised in DevOnBike.Overfit.Intrinsics.CpuFeatures; static readonly bool fields fold to JIT constants just like the raw intrinsic property. Only CpuFeatures itself reads IsSupported directly.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterOperationAction(AnalyzePropertyReference, OperationKind.PropertyReference);
        }

        private static void AnalyzePropertyReference(OperationAnalysisContext context)
        {
            var operation = (IPropertyReferenceOperation)context.Operation;
            var property = operation.Property;

            if (property.Name != "IsSupported" || !property.IsStatic)
            {
                return;
            }

            if (!IsHardwareIntrinsicsType(property.ContainingType))
            {
                return;
            }

            // CpuFeatures itself is the sanctioned reader of the raw properties.
            if (context.ContainingSymbol.ContainingType is { Name: "CpuFeatures" })
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(
                Rule, operation.Syntax.GetLocation(), property.ContainingType.Name));
        }

        private static bool IsHardwareIntrinsicsType(INamedTypeSymbol? type)
        {
            // Match System.Runtime.Intrinsics[.X86/.Arm/...] — walk the namespace chain upwards.
            for (var ns = type?.ContainingNamespace; ns is { IsGlobalNamespace: false }; ns = ns.ContainingNamespace)
            {
                if (ns.Name == "Intrinsics" &&
                    ns.ContainingNamespace is { Name: "Runtime", ContainingNamespace: { Name: nameof(System), ContainingNamespace.IsGlobalNamespace: true } })
                {
                    return true;
                }
            }

            return false;
        }
    }
}
