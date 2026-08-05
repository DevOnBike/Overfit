// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.FindSymbols;

namespace DevOnBike.Overfit.Navigator
{
    /// <summary>
    /// Turns a name typed by a human into the symbols Roslyn knows about.
    /// </summary>
    /// <remarks>
    /// This is the part of a semantic tool most likely to be quietly wrong, so it is deliberately literal:
    /// it never picks a "best" match. A name that resolves to several symbols comes back as several symbols
    /// and the caller must disambiguate. Guessing here would produce a confident answer about the wrong
    /// method, which is worse than no answer — the caller cannot tell the two apart from the output.
    /// </remarks>
    internal static class SymbolResolver
    {
        /// <summary>
        /// Finds source-declared symbols matching <paramref name="name"/>.
        /// </summary>
        /// <param name="solution">Solution to search.</param>
        /// <param name="name">
        /// A simple name (<c>InferenceEngine</c>), a member path (<c>InferenceEngine.Run</c>), or a
        /// namespace-qualified type (<c>DevOnBike.Overfit.InferenceEngine</c>). Matching is on the last
        /// segment; any earlier segments are applied as a containment filter afterwards.
        /// </param>
        /// <param name="cancellationToken">Cancellation.</param>
        /// <returns>Every match, ordered so types come before members of the same name.</returns>
        public static async Task<IReadOnlyList<ISymbol>> ResolveAsync(
            Solution solution,
            string name,
            CancellationToken cancellationToken)
        {
            var lastDot = name.LastIndexOf('.');
            var simpleName = lastDot >= 0 ? name[(lastDot + 1)..] : name;
            var qualifier = lastDot >= 0 ? name[..lastDot] : null;

            // Metadata symbols are excluded: this tool answers questions about THIS repository, and a search
            // that also matched framework types would bury every real result under BCL noise.
            var found = await SymbolFinder
                .FindSourceDeclarationsAsync(solution, simpleName, ignoreCase: false, cancellationToken)
                .ConfigureAwait(false);

            var matches = new List<ISymbol>();

            foreach (var symbol in found)
            {
                if (qualifier is not null && !MatchesQualifier(symbol, qualifier))
                {
                    continue;
                }

                matches.Add(symbol);
            }

            matches.Sort(static (a, b) => Rank(a).CompareTo(Rank(b)));
            return matches;
        }

        /// <summary>
        /// True when <paramref name="qualifier"/> is a suffix of the symbol's containing type/namespace chain.
        /// </summary>
        private static bool MatchesQualifier(ISymbol symbol, string qualifier)
        {
            var container = symbol.ContainingSymbol;

            while (container is not null && container is not IModuleSymbol)
            {
                var display = container.ToDisplayString();

                if (display == qualifier || display.EndsWith("." + qualifier, StringComparison.Ordinal))
                {
                    return true;
                }

                container = container.ContainingSymbol;
            }

            return false;
        }

        private static int Rank(ISymbol symbol)
        {
            return symbol.Kind switch
            {
                SymbolKind.NamedType => 0,
                SymbolKind.Method => 1,
                SymbolKind.Property => 2,
                SymbolKind.Field => 3,
                _ => 4,
            };
        }

        /// <summary>Renders a symbol the way a report should name it: qualified enough to be unambiguous.</summary>
        public static string Describe(ISymbol symbol)
        {
            return $"{symbol.Kind} {symbol.ToDisplayString()}";
        }

        /// <summary>Renders the first source location of a symbol as <c>path:line</c>, or empty if it has none.</summary>
        public static string DescribeLocation(ISymbol symbol)
        {
            foreach (var location in symbol.Locations)
            {
                if (location.IsInSource)
                {
                    return FormatLocation(location);
                }
            }

            return string.Empty;
        }

        /// <summary>Renders a source location as <c>path:line</c> with a 1-based line number.</summary>
        public static string FormatLocation(Location location)
        {
            var span = location.GetLineSpan();
            return $"{span.Path}:{span.StartLinePosition.Line + 1}";
        }
    }
}
