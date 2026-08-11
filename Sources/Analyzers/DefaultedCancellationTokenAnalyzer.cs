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
    /// OVERFIT041 — a <see cref="System.Threading.CancellationToken"/> parameter with a default value.
    ///
    /// <para><b>A defaulted token is cancellation that quietly is not there.</b> The caller writes nothing,
    /// the compiler supplies <c>CancellationToken.None</c>, and the operation becomes uncancellable — with
    /// no diagnostic, no warning and nothing at the call site to read. The method still looks cancellable
    /// from its signature, which is the worst of both: the API advertises a capability the caller has
    /// silently declined.</para>
    ///
    /// <para><b>This is the same failure shape the rest of this repository is organised against.</b> A guard
    /// that reports nothing and a guard that is switched off produce identical output; a floor set too high
    /// looks exactly like a healthy cluster; an unobserved task looks like a running one. A defaulted token
    /// belongs to that family — the defect is invisible at the place where it is introduced, which is the
    /// call site rather than the declaration.</para>
    ///
    /// <para><b>It is the complement of OVERFIT030, not a duplicate.</b> That rule requires an awaitable
    /// public API to HAVE a token, so a caller can abandon it. This one requires the caller to actually pass
    /// one. Having the parameter and defaulting it satisfies OVERFIT030 while delivering none of what it was
    /// for, which is exactly the gap worth closing.</para>
    ///
    /// <para><b>Interfaces are flagged too, and that is where the fix usually belongs.</b> A default declared
    /// on an interface member is what every implementation inherits at the call site; removing it there is
    /// one edit that fixes every caller. A default on an override or an explicit implementation is worse
    /// than useless — it is IGNORED when the method is called through the interface, so it reads as a
    /// promise the language does not keep.</para>
    ///
    /// <para><b>The escape is a pragma with a reason</b>, on the same "name the constraint" contract as
    /// OVERFIT022, OVERFIT023, OVERFIT039 and OVERFIT040. The legitimate case is an API whose signature is
    /// fixed by someone else — an interface from a package, a framework-mandated shape — where the default
    /// is the only way to match.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class DefaultedCancellationTokenAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT041";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "CancellationToken parameter with a default value",
            messageFormat:
                "'{0}' defaults its CancellationToken — a caller that passes nothing gets "
                + "CancellationToken.None and the operation is silently uncancellable while still looking "
                + "cancellable; drop the default so every caller states which token it is handing over",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "A defaulted CancellationToken hides the decision not to support cancellation at the call "
                + "site, where nothing is written and nothing can be reviewed. Removing the default makes "
                + "every caller name the token it is passing, including when that is deliberately "
                + "CancellationToken.None.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeParameter, SyntaxKind.Parameter);
        }

        private static void AnalyzeParameter(SyntaxNodeAnalysisContext context)
        {
            var parameter = (ParameterSyntax)context.Node;

            if (parameter.Default is null)
            {
                return;
            }

            var symbol = context.SemanticModel.GetDeclaredSymbol(parameter, context.CancellationToken);

            if (symbol is null
                || symbol.Type.ToDisplayString() != "System.Threading.CancellationToken")
            {
                return;
            }

            // Named after the method rather than the parameter: the reader has to change a signature, and
            // the signature is what they will search for.
            var owner = symbol.ContainingSymbol?.Name ?? parameter.Identifier.Text;

            context.ReportDiagnostic(Diagnostic.Create(Rule, parameter.GetLocation(), owner));
        }
    }
}
