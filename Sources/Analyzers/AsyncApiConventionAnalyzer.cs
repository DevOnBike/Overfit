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
    /// OVERFIT029 — an awaitable method not named <c>…Async</c>, and OVERFIT030 — an awaitable public API with
    /// no <see cref="System.Threading.CancellationToken"/> parameter.
    ///
    /// <para><b>The suffix is a caller-safety device, not decoration.</b> A method returning a task looks
    /// exactly like one that does the work, at the call site, right up until someone forgets to await it and
    /// the operation silently never happens — or throws into an unobserved task. CS4014 catches the bare
    /// discard; it does not help when the result is assigned or passed along. The naming convention is what
    /// makes the mistake visible while reading.</para>
    ///
    /// <para><b>The token is a liveness device.</b> An async operation with no way to cancel it is an
    /// operation the host cannot abandon: a request that times out keeps its model, its KV cache and its
    /// thread until it finishes on its own. In a server that is how a slow dependency becomes an outage.
    /// CA1068 checks where a token sits in the parameter list; nothing in the shipped set checks that one is
    /// offered at all, which is the part that matters.</para>
    ///
    /// <para><b>Scoping keeps it honest.</b> The token rule fires only on <c>public</c>/<c>internal</c>
    /// surface — a private helper inherits its caller's token and does not need its own. Neither rule fires on
    /// an override or an interface implementation, where the signature is not the author's to choose, nor on
    /// <c>Main</c>, nor on local functions.</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class AsyncApiConventionAnalyzer : DiagnosticAnalyzer
    {
        public const string SuffixDiagnosticId = "OVERFIT029";
        public const string CancellationDiagnosticId = "OVERFIT030";

        private static readonly DiagnosticDescriptor SuffixRule = new(
            SuffixDiagnosticId,
            title: "Awaitable method is not named Async",
            messageFormat: "'{0}' returns an awaitable but is not named '{0}Async' — at the call site it reads like completed work, and a missing await then fails silently",
            category: "Naming",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "A task-returning method whose name does not say so is indistinguishable from a synchronous one while reading. Suffix it with Async. Overrides and interface implementations, whose names are fixed elsewhere, are not reported.");

        private static readonly DiagnosticDescriptor CancellationRule = new(
            CancellationDiagnosticId,
            title: "Awaitable API takes no CancellationToken",
            messageFormat: "'{0}' is awaitable public API with no CancellationToken — callers cannot abandon it, so a slow dependency holds its resources until it finishes",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description: "An async operation that cannot be cancelled cannot be abandoned: a timed-out request keeps its buffers, model and thread until the work completes on its own. Accept a CancellationToken (last parameter, per CA1068) and honour it. Private helpers, overrides and interface implementations are not reported.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [SuffixRule, CancellationRule];

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSymbolAction(Analyze, SymbolKind.Method);
        }

        private static void Analyze(SymbolAnalysisContext context)
        {
            var method = (IMethodSymbol)context.Symbol;

            if (method.IsImplicitlyDeclared
                || method.MethodKind is not (MethodKind.Ordinary or MethodKind.LocalFunction)
                || method.MethodKind == MethodKind.LocalFunction)
            {
                return;
            }

            // The signature is fixed somewhere else; renaming it here is not an option the author has.
            if (method.IsOverride || method.ExplicitInterfaceImplementations.Length > 0 || ImplementsInterface(method))
            {
                return;
            }

            // "<Main>$" is the compiler's name for a top-level-statements entry point — as fixed as "Main".
            if (method.Name is "Main" or "<Main>$" or "DisposeAsync")
            {
                return;
            }

            if (!IsAwaitable(context.Compilation, method.ReturnType))
            {
                return;
            }

            var location = method.Locations.Length > 0 ? method.Locations[0] : Location.None;

            if (!method.Name.EndsWith("Async", System.StringComparison.Ordinal))
            {
                context.ReportDiagnostic(Diagnostic.Create(SuffixRule, location, method.Name));
            }

            if (method.DeclaredAccessibility is not (Accessibility.Public or Accessibility.Internal))
            {
                return;
            }

            if (!HasCancellationToken(method))
            {
                context.ReportDiagnostic(Diagnostic.Create(CancellationRule, location, method.Name));
            }
        }

        /// <summary>Task, Task&lt;T&gt;, ValueTask, ValueTask&lt;T&gt; and IAsyncEnumerable&lt;T&gt;.</summary>
        private static bool IsAwaitable(Compilation compilation, ITypeSymbol type)
        {
            var named = type as INamedTypeSymbol;
            if (named is null)
            {
                return false;
            }

            var definition = named.OriginalDefinition;

            foreach (var candidate in new[]
                     {
                         "System.Threading.Tasks.Task",
                         "System.Threading.Tasks.Task`1",
                         "System.Threading.Tasks.ValueTask",
                         "System.Threading.Tasks.ValueTask`1",
                         "System.Collections.Generic.IAsyncEnumerable`1",
                     })
            {
                var symbol = compilation.GetTypeByMetadataName(candidate);
                if (symbol is not null && SymbolEqualityComparer.Default.Equals(definition, symbol))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// A token parameter, or something that already carries one. An ASP.NET middleware or handler taking
        /// an <c>HttpContext</c> has <c>HttpContext.RequestAborted</c> — asking it for a second token would be
        /// asking it to ignore the real one, and the signature is convention-bound anyway.
        /// </summary>
        private static bool HasCancellationToken(IMethodSymbol method)
        {
            foreach (var parameter in method.Parameters)
            {
                var name = parameter.Type.ToDisplayString();

                if (name is "System.Threading.CancellationToken" or "Microsoft.AspNetCore.Http.HttpContext")
                {
                    return true;
                }
            }

            return false;
        }

        private static bool ImplementsInterface(IMethodSymbol method)
        {
            var type = method.ContainingType;
            if (type is null)
            {
                return false;
            }

            foreach (var contract in type.AllInterfaces)
            {
                foreach (var member in contract.GetMembers(method.Name))
                {
                    if (member is IMethodSymbol declared
                        && SymbolEqualityComparer.Default.Equals(type.FindImplementationForInterfaceMember(declared), method))
                    {
                        return true;
                    }
                }
            }

            return false;
        }
    }
}
