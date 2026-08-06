// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.FindSymbols;

namespace DevOnBike.Overfit.Navigator
{
    /// <summary>
    /// The four questions this tool answers, each of which grep answers wrongly.
    /// </summary>
    /// <remarks>
    /// Every one of these is a question about <b>symbols</b>, not about text. Grep on a method name finds the
    /// comment that mentions it, the unrelated method with the same name on a different type, and the string
    /// literal — and misses the call made through an interface, the one made through a delegate, and the
    /// override in a derived class. That difference is the entire justification for paying seven seconds to
    /// build a semantic model.
    /// </remarks>
    internal sealed class NavigatorQueries
    {
        private readonly Solution _solution;

        public NavigatorQueries(Solution solution)
        {
            _solution = solution;
        }

        /// <summary>A single place a symbol is used, named the way a reader can act on it.</summary>
        public sealed record Reference(string Location, string InSymbol, string Project, bool IsTest);

        /// <summary>A symbol that appears to be dead, with the reason that verdict might be wrong.</summary>
        public sealed record UnusedCandidate(string Symbol, string Location, string Verdict);

        /// <summary>One edge of a call graph.</summary>
        public sealed record CallEdge(int Depth, string Caller, string Location, bool IsDirect);

        /// <summary>
        /// Every place <paramref name="symbol"/> is actually referenced, across the whole solution.
        /// </summary>
        public async Task<IReadOnlyList<Reference>> FindReferencesAsync(
            ISymbol symbol,
            CancellationToken cancellationToken)
        {
            var found = await SymbolFinder.FindReferencesAsync(symbol, _solution, cancellationToken)
                .ConfigureAwait(false);

            var results = new List<Reference>();

            foreach (var referenced in found)
            {
                foreach (var location in referenced.Locations)
                {
                    var document = location.Document;
                    var containing = await ContainingSymbolNameAsync(location, cancellationToken).ConfigureAwait(false);

                    results.Add(new Reference(
                        SymbolResolver.FormatLocation(location.Location),
                        containing,
                        document.Project.Name,
                        IsTestProject(document.Project)));
                }
            }

            results.Sort(static (a, b) => string.CompareOrdinal(a.Location, b.Location));
            return results;
        }

        /// <summary>
        /// Implementations of an interface or members, and derived classes of a class.
        /// </summary>
        /// <remarks>
        /// Both directions are needed because the question "who implements this" means different things for an
        /// interface and for an abstract base, and answering only one of them silently under-reports on the other.
        /// </remarks>
        public async Task<IReadOnlyList<string>> FindImplementationsAsync(
            ISymbol symbol,
            CancellationToken cancellationToken)
        {
            var results = new List<string>();

            var implementations = await SymbolFinder
                .FindImplementationsAsync(symbol, _solution, cancellationToken: cancellationToken)
                .ConfigureAwait(false);

            foreach (var implementation in implementations)
            {
                results.Add($"{implementation.ToDisplayString()}   {SymbolResolver.DescribeLocation(implementation)}");
            }

            if (symbol is INamedTypeSymbol { TypeKind: TypeKind.Class } namedType)
            {
                var derived = await SymbolFinder
                    .FindDerivedClassesAsync(namedType, _solution, transitive: true, cancellationToken: cancellationToken)
                    .ConfigureAwait(false);

                foreach (var type in derived)
                {
                    results.Add($"{type.ToDisplayString()}   {SymbolResolver.DescribeLocation(type)}");
                }
            }

            results.Sort(StringComparer.Ordinal);
            return results;
        }

        /// <summary>
        /// Walks callers of <paramref name="symbol"/> up to <paramref name="maxDepth"/> levels.
        /// </summary>
        /// <remarks>
        /// Answers "what reaches this" — the question behind every "is this on the hot path" and every "what
        /// breaks if I change this signature". Cycles are possible in real code and are cut, not followed.
        /// </remarks>
        public async Task<IReadOnlyList<CallEdge>> FindCallersAsync(
            ISymbol symbol,
            int maxDepth,
            CancellationToken cancellationToken)
        {
            var edges = new List<CallEdge>();
            var visited = new HashSet<string>(StringComparer.Ordinal) { symbol.ToDisplayString() };
            var frontier = new List<ISymbol> { symbol };

            for (var depth = 1; depth <= maxDepth && frontier.Count > 0; depth++)
            {
                var next = new List<ISymbol>();

                foreach (var current in frontier)
                {
                    var callers = await SymbolFinder.FindCallersAsync(current, _solution, cancellationToken)
                        .ConfigureAwait(false);

                    foreach (var caller in callers)
                    {
                        var name = caller.CallingSymbol.ToDisplayString();
                        var location = string.Empty;

                        foreach (var callSite in caller.Locations)
                        {
                            location = SymbolResolver.FormatLocation(callSite);
                            break;
                        }

                        edges.Add(new CallEdge(depth, name, location, caller.IsDirect));

                        if (visited.Add(name))
                        {
                            next.Add(caller.CallingSymbol);
                        }
                    }
                }

                frontier = next;
            }

            return edges;
        }

        /// <summary>
        /// Symbols in one project that nothing references, split by how much the verdict can be trusted.
        /// </summary>
        /// <param name="project">Project to scan. One project at a time — the cost is one reference search per symbol.</param>
        /// <param name="includePublic">
        /// Include <c>public</c> symbols. Off by default and that default matters: <c>DevOnBike.Overfit</c> is a
        /// published library, so "nothing in this repository calls it" is not evidence that a public member is
        /// dead — its callers are outside the repository by design.
        /// </param>
        /// <param name="cancellationToken">Cancellation.</param>
        /// <remarks>
        /// <para>
        /// The following are skipped because a reference count of zero does <b>not</b> mean unused for them, and
        /// reporting them would train the reader to ignore the whole list:
        /// </para>
        /// <list type="bullet">
        ///   <item>overrides and interface implementations — reached through the base, never by name;</item>
        ///   <item>anything carrying an attribute — xUnit facts, JSON-serialized members and DI-registered types
        ///         are all invoked by a framework that no call site mentions;</item>
        ///   <item>entry points;</item>
        ///   <item>implicitly declared symbols (record members, default constructors).</item>
        /// </list>
        /// <para>
        /// What is <i>not</i> skipped is the case worth having this query for: a symbol referenced only from
        /// <c>Tests</c>. That is production code kept alive solely by its own test, and it is reported under its
        /// own verdict rather than hidden, because it is a deletion candidate that no compiler warning finds.
        /// </para>
        /// </remarks>
        public async Task<IReadOnlyList<UnusedCandidate>> FindUnusedAsync(
            Project project,
            bool includePublic,
            CancellationToken cancellationToken)
        {
            var compilation = await project.GetCompilationAsync(cancellationToken).ConfigureAwait(false);

            if (compilation is null)
            {
                return Array.Empty<UnusedCandidate>();
            }

            var candidates = new List<ISymbol>();
            CollectSymbols(compilation.Assembly.GlobalNamespace, includePublic, candidates);

            var results = new List<UnusedCandidate>();

            foreach (var symbol in candidates)
            {
                var references = await FindReferencesAsync(symbol, cancellationToken).ConfigureAwait(false);
                var verdict = Judge(references);

                if (verdict is null)
                {
                    continue;
                }

                results.Add(new UnusedCandidate(
                    symbol.ToDisplayString(),
                    SymbolResolver.DescribeLocation(symbol),
                    verdict));
            }

            return results;
        }

        private static string? Judge(IReadOnlyList<Reference> references)
        {
            var productionUses = 0;
            var testUses = 0;

            foreach (var reference in references)
            {
                if (reference.IsTest)
                {
                    testUses++;
                    continue;
                }

                productionUses++;
            }

            if (productionUses == 0 && testUses == 0)
            {
                return "no references anywhere";
            }

            if (productionUses == 0)
            {
                return $"referenced only by tests ({testUses})";
            }

            return null;
        }

        /// <summary>
        /// Walks the namespace tree with an explicit worklist rather than recursion (OVERFIT022).
        /// </summary>
        /// <remarks>
        /// Namespace and type nesting are shallow in practice, but the depth here is a property of whatever
        /// source is loaded, not of this code — and an input-dependent recursion depth is exactly what the rule
        /// exists to stop, because a .NET stack overflow cannot be caught and takes the process with it.
        /// </remarks>
        private static void CollectSymbols(INamespaceSymbol root, bool includePublic, List<ISymbol> into)
        {
            var namespaces = new Stack<INamespaceSymbol>();
            namespaces.Push(root);

            while (namespaces.Count > 0)
            {
                foreach (var member in namespaces.Pop().GetMembers())
                {
                    if (member is INamespaceSymbol nested)
                    {
                        namespaces.Push(nested);
                        continue;
                    }

                    if (member is INamedTypeSymbol type)
                    {
                        CollectTypes(type, includePublic, into);
                    }
                }
            }
        }

        /// <summary>Walks a type and its nested types with an explicit worklist (OVERFIT022, as above).</summary>
        private static void CollectTypes(INamedTypeSymbol root, bool includePublic, List<ISymbol> into)
        {
            var types = new Stack<INamedTypeSymbol>();
            types.Push(root);

            while (types.Count > 0)
            {
                var type = types.Pop();

                if (IsCandidate(type, includePublic))
                {
                    into.Add(type);
                }

                foreach (var member in type.GetMembers())
                {
                    if (member is INamedTypeSymbol nested)
                    {
                        types.Push(nested);
                        continue;
                    }

                    if (IsCandidate(member, includePublic))
                    {
                        into.Add(member);
                    }
                }
            }
        }

        private static bool IsCandidate(ISymbol symbol, bool includePublic)
        {
            if (symbol.IsImplicitlyDeclared || symbol.Locations.Length == 0)
            {
                return false;
            }

            if (!includePublic && IsExternallyVisible(symbol))
            {
                return false;
            }

            if (symbol.IsOverride || symbol.GetAttributes().Length > 0)
            {
                return false;
            }

            if (symbol is IMethodSymbol method)
            {
                if (method.MethodKind != MethodKind.Ordinary || method.ExplicitInterfaceImplementations.Length > 0)
                {
                    return false;
                }

                if (IsInterfaceImplementation(method))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// True when a symbol can actually be reached from outside the assembly.
        /// </summary>
        /// <remarks>
        /// The symbol's own accessibility is not sufficient and using it alone under-reports: a <c>public</c>
        /// method on an <c>internal</c> class cannot be called from outside the assembly, so the "its callers
        /// live outside this repository" exemption does not apply to it — yet declared accessibility says
        /// <c>Public</c> and excludes it from the default scan. Visibility is the accessibility of the symbol
        /// <i>and of every type containing it</i>.
        /// </remarks>
        private static bool IsExternallyVisible(ISymbol symbol)
        {
            var current = symbol;

            while (current is not null and not INamespaceSymbol)
            {
                if (current.DeclaredAccessibility is not (Accessibility.Public
                    or Accessibility.Protected
                    or Accessibility.ProtectedOrInternal))
                {
                    return false;
                }

                current = current.ContainingSymbol;
            }

            return true;
        }

        private static bool IsInterfaceImplementation(IMethodSymbol method)
        {
            var containing = method.ContainingType;

            foreach (var iface in containing.AllInterfaces)
            {
                foreach (var member in iface.GetMembers())
                {
                    if (SymbolEqualityComparer.Default.Equals(containing.FindImplementationForInterfaceMember(member), method))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        private static async Task<string> ContainingSymbolNameAsync(
            ReferenceLocation location,
            CancellationToken cancellationToken)
        {
            var model = await location.Document.GetSemanticModelAsync(cancellationToken).ConfigureAwait(false);
            var root = await location.Document.GetSyntaxRootAsync(cancellationToken).ConfigureAwait(false);

            if (model is null || root is null)
            {
                return string.Empty;
            }

            var node = root.FindNode(location.Location.SourceSpan);
            var enclosing = model.GetEnclosingSymbol(node.SpanStart, cancellationToken);

            return enclosing?.ToDisplayString() ?? string.Empty;
        }

        /// <summary>
        /// True for projects whose references do not count as production use.
        /// </summary>
        /// <remarks>
        /// Matched on the project name rather than the path, because the path form differs between the solution
        /// root and a CI checkout and a name is what the workspace reports consistently.
        /// </remarks>
        private static bool IsTestProject(Project project)
        {
            return project.Name is "Tests" or "AotSmokeTest";
        }
    }
}
