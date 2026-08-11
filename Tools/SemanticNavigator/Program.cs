// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.CompilerServices;
using Microsoft.CodeAnalysis;

namespace DevOnBike.Overfit.Navigator
{
    /// <summary>Entry point for the semantic navigator.</summary>
    /// <remarks>
    /// Two shapes, deliberately: verbs for a human at a terminal, and <c>serve</c> for an MCP host. They run the
    /// same query objects, so a result seen at the terminal is the result the host receives — which is what makes
    /// the terminal usable for verifying the server rather than merely for demonstrating it.
    /// </remarks>
    internal static class Program
    {
        private const string Usage = """
            usage: overfit-navigator <verb> [args]

              measure                         time a full solution load on this box
              refs     <symbol>               every real reference, solution-wide
              impls    <symbol>               implementations of an interface / classes derived from a class
              callers  <symbol> [--depth N]   what reaches this, transitively (default depth 3)
              unused   <project> [--public]   symbols nothing references, and ones only tests reference
              serve                           run as an MCP server over stdio

            options:
              --solution <path>               defaults to the Overfit.sln above the executable
            """;

        private static async Task<int> Main(string[] args)
        {
            // MUST come before any type from Microsoft.CodeAnalysis.MSBuild is resolved. The JIT loads the types
            // a method references when it compiles that method, not when execution reaches the line — so the
            // registration cannot live in a method that also mentions MSBuildWorkspace.
            Microsoft.Build.Locator.MSBuildLocator.RegisterDefaults();

            if (args.Length == 0)
            {
                Console.WriteLine(Usage);
                return 1;
            }

            try
            {
                return await RunAsync(args).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"error: {ex.Message}");
                return 1;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static async Task<int> RunAsync(string[] args)
        {
            var verb = args[0];
            var options = new Options(args);

            if (verb == "serve")
            {
                return await NavigatorMcpServer.RunAsync(options.Solution, CancellationToken.None)
                    .ConfigureAwait(false);
            }

            var started = Stopwatch.GetTimestamp();
            using var loader = await WorkspaceLoader.OpenAsync(options.Solution, CancellationToken.None)
                .ConfigureAwait(false);

            if (verb == "measure")
            {
                return await MeasureAsync(loader).ConfigureAwait(false);
            }

            ReportLoadFailures(loader);
            Console.Error.WriteLine($"[loaded in {Stopwatch.GetElapsedTime(started).TotalSeconds:F1} s]");

            var queries = new NavigatorQueries(loader.Solution);
            var queryStarted = Stopwatch.GetTimestamp();
            var exitCode = await DispatchAsync(verb, options, loader, queries).ConfigureAwait(false);
            Console.Error.WriteLine($"[query {Stopwatch.GetElapsedTime(queryStarted).TotalSeconds:F2} s]");

            return exitCode;
        }

        private static async Task<int> DispatchAsync(
            string verb,
            Options options,
            WorkspaceLoader loader,
            NavigatorQueries queries)
        {
            if (verb == "unused")
            {
                return await RunUnusedAsync(options, loader, queries).ConfigureAwait(false);
            }

            if (options.Target is null)
            {
                Console.Error.WriteLine($"verb '{verb}' needs a symbol name");
                return 2;
            }

            var symbol = await ResolveOneAsync(loader.Solution, options.Target).ConfigureAwait(false);

            if (symbol is null)
            {
                return 3;
            }

            Console.WriteLine($"# {SymbolResolver.Describe(symbol)}");
            Console.WriteLine($"# declared at {SymbolResolver.DescribeLocation(symbol)}");
            Console.WriteLine();

            if (verb == "refs")
            {
                var references = await queries.FindReferencesAsync(symbol, CancellationToken.None)
                    .ConfigureAwait(false);

                foreach (var reference in references)
                {
                    var marker = reference.IsTest ? "[test] " : "       ";
                    Console.WriteLine($"{marker}{reference.Location}   in {reference.InSymbol}");
                }

                Console.WriteLine($"\n{references.Count} reference(s)");
                return 0;
            }

            if (verb == "impls")
            {
                var implementations = await queries.FindImplementationsAsync(symbol, CancellationToken.None)
                    .ConfigureAwait(false);

                foreach (var implementation in implementations)
                {
                    Console.WriteLine($"   {implementation}");
                }

                Console.WriteLine($"\n{implementations.Count} implementation(s)");
                return 0;
            }

            if (verb == "callers")
            {
                var edges = await queries.FindCallersAsync(symbol, options.Depth, CancellationToken.None)
                    .ConfigureAwait(false);

                foreach (var edge in edges)
                {
                    var indirect = edge.IsDirect ? string.Empty : "  (indirect)";
                    Console.WriteLine($"{new string(' ', edge.Depth * 2)}<- {edge.Caller}   {edge.Location}{indirect}");
                }

                Console.WriteLine($"\n{edges.Count} call site(s), depth {options.Depth}");
                return 0;
            }

            Console.Error.WriteLine($"unknown verb '{verb}'");
            Console.WriteLine(Usage);
            return 2;
        }

        private static async Task<int> RunUnusedAsync(Options options, WorkspaceLoader loader, NavigatorQueries queries)
        {
            if (options.Target is null)
            {
                Console.Error.WriteLine("verb 'unused' needs a project name");
                return 2;
            }

            Project? project = null;

            foreach (var candidate in loader.Solution.Projects)
            {
                if (string.Equals(candidate.Name, options.Target, StringComparison.OrdinalIgnoreCase))
                {
                    project = candidate;
                    break;
                }
            }

            if (project is null)
            {
                Console.Error.WriteLine($"no project named '{options.Target}'");
                return 3;
            }

            var candidates = await queries.FindUnusedAsync(project, options.IncludePublic, CancellationToken.None)
                .ConfigureAwait(false);

            foreach (var candidate in candidates)
            {
                Console.WriteLine($"{candidate.Verdict,-32} {candidate.Symbol}");
                Console.WriteLine($"{string.Empty,-32} {candidate.Location}");
            }

            Console.WriteLine($"\n{candidates.Count} candidate(s) in {project.Name}");
            Console.WriteLine("NOTE: overrides, interface implementations and attributed symbols are excluded —");
            Console.WriteLine("      a zero reference count does not mean unused for those.");
            return 0;
        }

        /// <summary>
        /// Resolves a name to exactly one symbol, or explains why it could not.
        /// </summary>
        /// <remarks>
        /// Ambiguity is reported rather than resolved by picking the first match. An answer about the wrong
        /// overload is indistinguishable from an answer about the right one once it reaches the reader.
        /// </remarks>
        private static async Task<ISymbol?> ResolveOneAsync(Solution solution, string name)
        {
            var symbols = await SymbolResolver.ResolveAsync(solution, name, CancellationToken.None)
                .ConfigureAwait(false);

            if (symbols.Count == 0)
            {
                Console.Error.WriteLine($"no source symbol named '{name}'");
                return null;
            }

            if (symbols.Count == 1)
            {
                return symbols[0];
            }

            var first = symbols[0];
            var allSameName = true;

            foreach (var symbol in symbols)
            {
                if (symbol.ToDisplayString() != first.ToDisplayString())
                {
                    allSameName = false;
                    break;
                }
            }

            if (allSameName)
            {
                return first;
            }

            Console.Error.WriteLine($"'{name}' is ambiguous ({symbols.Count} matches) — qualify it:");

            foreach (var symbol in symbols)
            {
                Console.Error.WriteLine($"   {SymbolResolver.Describe(symbol)}   {SymbolResolver.DescribeLocation(symbol)}");
            }

            return null;
        }

        private static void ReportLoadFailures(WorkspaceLoader loader)
        {
            if (loader.Failures.Count == 0)
            {
                return;
            }

            // Loud on stderr, always. A project that failed to load contributes no symbols, so every query over
            // it answers "nothing found" — which is indistinguishable from a correct empty result.
            Console.Error.WriteLine($"WARNING: {loader.Failures.Count} project(s) failed to load; results are incomplete.");

            foreach (var failure in loader.Failures)
            {
                Console.Error.WriteLine($"   {failure}");
            }
        }

        /// <summary>Reports what a semantic query costs on this box, split into the two phases that scale differently.</summary>
        private static async Task<int> MeasureAsync(WorkspaceLoader loader)
        {
            var projects = new List<Project>(loader.Solution.Projects);
            Console.WriteLine($"open     : {loader.OpenDuration.TotalSeconds,7:F2} s   ({projects.Count} projects)");

            var compilations = await loader.WarmCompilationsAsync(CancellationToken.None).ConfigureAwait(false);
            var total = TimeSpan.Zero;
            var documents = 0;

            foreach (var (_, duration, docs) in compilations)
            {
                total += duration;
                documents += docs;
            }

            Console.WriteLine($"compile  : {total.TotalSeconds,7:F2} s   ({documents} documents)");
            Console.WriteLine($"TOTAL    : {(loader.OpenDuration + total).TotalSeconds,7:F2} s");
            Console.WriteLine();
            Console.WriteLine("slowest projects (a project's own cost is charged to whoever compiled it first,");
            Console.WriteLine("so a referenced project's time appears under its first dependent, not under itself):");

            var ordered = new List<(string Project, TimeSpan Duration, int Documents)>(compilations);
            ordered.Sort(static (a, b) => b.Duration.CompareTo(a.Duration));

            for (var i = 0; i < Math.Min(8, ordered.Count); i++)
            {
                Console.WriteLine($"   {ordered[i].Duration.TotalSeconds,6:F2} s  {ordered[i].Documents,5} docs  {ordered[i].Project}");
            }

            ReportLoadFailures(loader);
            return 0;
        }

        /// <summary>Command-line options, parsed positionally with a couple of named switches.</summary>
        private sealed class Options
        {
            public Options(string[] args)
            {
                string? solution = null;
                string? target = null;
                var depth = 3;
                var includePublic = false;

                for (var i = 1; i < args.Length; i++)
                {
                    var arg = args[i];

                    if (arg == "--solution" && i + 1 < args.Length)
                    {
                        solution = args[++i];
                        continue;
                    }

                    if (arg == "--depth" && i + 1 < args.Length)
                    {
                        depth = int.Parse(args[++i]);
                        continue;
                    }

                    if (arg is "--public" or "--include-public")
                    {
                        includePublic = true;
                        continue;
                    }

                    target ??= arg;
                }

                Solution = solution ?? DefaultSolutionPath();
                Target = target;
                Depth = depth;
                IncludePublic = includePublic;
            }

            public string Solution
            {
                get;
            }

            public string? Target
            {
                get;
            }

            public int Depth
            {
                get;
            }

            public bool IncludePublic
            {
                get;
            }

            private static string DefaultSolutionPath()
            {
                var dir = AppContext.BaseDirectory;

                while (dir is not null)
                {
                    var candidate = Path.Combine(dir, "Overfit.sln");

                    if (File.Exists(candidate))
                    {
                        return candidate;
                    }

                    dir = Path.GetDirectoryName(dir.TrimEnd(Path.DirectorySeparatorChar));
                }

                throw new FileNotFoundException("Overfit.sln not found above the executable; pass --solution.");
            }
        }
    }
}
