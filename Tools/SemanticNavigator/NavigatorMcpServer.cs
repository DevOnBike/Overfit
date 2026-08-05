// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.CodeAnalysis;

namespace DevOnBike.Overfit.Navigator
{
    /// <summary>
    /// Exposes the navigator's queries to an MCP host over stdio, holding one warm workspace for its lifetime.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a <b>separate server from <c>Sources/Mcp</c> and must stay separate.</b> That one ships inside the
    /// Native-AOT <c>overfit</c> CLI, which is why its wire contracts are source-generated DTOs with no
    /// reflection. This one hosts MSBuild and Roslyn, is reflection-heavy by construction, and exists only on a
    /// developer machine — merging them would drag an AOT-hostile dependency graph into the product to save a
    /// hundred lines of JSON plumbing.
    /// </para>
    /// <para>
    /// <b>stdout carries the protocol and nothing else.</b> A stray <c>Console.WriteLine</c> anywhere on this path
    /// corrupts the JSON-RPC stream and the host reports a parse error that names no cause. All diagnostics,
    /// including the load time and any project that failed to open, go to stderr.
    /// </para>
    /// </remarks>
    internal static class NavigatorMcpServer
    {
        private const string ProtocolVersion = "2024-11-05";

        /// <summary>Loads the solution once, then serves requests until stdin closes.</summary>
        public static async Task<int> RunAsync(string solutionPath, CancellationToken cancellationToken)
        {
            var started = Stopwatch.GetTimestamp();
            using var loader = await WorkspaceLoader.OpenAsync(solutionPath, cancellationToken).ConfigureAwait(false);

            // Warming here rather than on first query is the whole reason this is a server: the host's first
            // question would otherwise pay for every project in the solution and look like a hang.
            await loader.WarmCompilationsAsync(cancellationToken).ConfigureAwait(false);

            Console.Error.WriteLine(
                $"[navigator] {solutionPath} loaded in {Stopwatch.GetElapsedTime(started).TotalSeconds:F1} s");

            foreach (var failure in loader.Failures)
            {
                Console.Error.WriteLine($"[navigator] LOAD FAILURE (queries over this project return nothing): {failure}");
            }

            var queries = new NavigatorQueries(loader.Solution);
            var output = new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)) { AutoFlush = true };

            while (!cancellationToken.IsCancellationRequested)
            {
                var line = await Console.In.ReadLineAsync(cancellationToken).ConfigureAwait(false);

                if (line is null)
                {
                    break;
                }

                if (line.Length == 0)
                {
                    continue;
                }

                var response = await HandleAsync(line, loader, queries, cancellationToken).ConfigureAwait(false);

                if (response is not null)
                {
                    await output.WriteLineAsync(response.ToJsonString()).ConfigureAwait(false);
                }
            }

            return 0;
        }

        private static async Task<JsonNode?> HandleAsync(
            string line,
            WorkspaceLoader loader,
            NavigatorQueries queries,
            CancellationToken cancellationToken)
        {
            JsonNode? request;

            try
            {
                request = JsonNode.Parse(line);
            }
            catch (JsonException ex)
            {
                Console.Error.WriteLine($"[navigator] unparseable request: {ex.Message}");
                return null;
            }

            if (request is null)
            {
                return null;
            }

            var method = request["method"]?.GetValue<string>();
            var id = request["id"];

            // A notification has no id and must never be answered — replying to one is a protocol violation
            // that some hosts surface as a hang rather than as an error.
            if (id is null)
            {
                return null;
            }

            if (method == "initialize")
            {
                return Success(id, new JsonObject
                {
                    ["protocolVersion"] = ProtocolVersion,
                    ["capabilities"] = new JsonObject { ["tools"] = new JsonObject() },
                    ["serverInfo"] = new JsonObject
                    {
                        ["name"] = "overfit-navigator",
                        ["version"] = "1.0.0",
                    },
                });
            }

            if (method == "ping")
            {
                return Success(id, new JsonObject());
            }

            if (method == "tools/list")
            {
                return Success(id, new JsonObject { ["tools"] = ToolCatalog.Describe() });
            }

            if (method == "tools/call")
            {
                var parameters = request["params"];
                var name = parameters?["name"]?.GetValue<string>() ?? string.Empty;
                var arguments = parameters?["arguments"] as JsonObject ?? new JsonObject();

                try
                {
                    var text = await ToolCatalog
                        .InvokeAsync(name, arguments, loader, queries, cancellationToken)
                        .ConfigureAwait(false);

                    return Success(id, new JsonObject
                    {
                        ["content"] = new JsonArray(new JsonObject
                        {
                            ["type"] = "text",
                            ["text"] = text,
                        }),
                    });
                }
                catch (Exception ex)
                {
                    // Reported as a tool result with isError, not as a JSON-RPC error: the host shows the former
                    // to the model, which can then correct the argument, and swallows the latter.
                    return Success(id, new JsonObject
                    {
                        ["isError"] = true,
                        ["content"] = new JsonArray(new JsonObject
                        {
                            ["type"] = "text",
                            ["text"] = $"{name} failed: {ex.Message}",
                        }),
                    });
                }
            }

            return Error(id, -32601, $"unknown method '{method}'");
        }

        private static JsonObject Success(JsonNode id, JsonNode result)
        {
            return new JsonObject
            {
                ["jsonrpc"] = "2.0",
                ["id"] = id.DeepClone(),
                ["result"] = result,
            };
        }

        private static JsonObject Error(JsonNode id, int code, string message)
        {
            return new JsonObject
            {
                ["jsonrpc"] = "2.0",
                ["id"] = id.DeepClone(),
                ["error"] = new JsonObject
                {
                    ["code"] = code,
                    ["message"] = message,
                },
            };
        }

        /// <summary>The tools this server offers, and how each one runs.</summary>
        private static class ToolCatalog
        {
            public static JsonArray Describe()
            {
                return new JsonArray(
                    Tool(
                        "find_references",
                        "Every place a symbol is actually used, solution-wide, resolved semantically. Finds calls "
                        + "made through an interface or a base class and ignores same-named members of unrelated "
                        + "types, comments and string literals — all of which a text search gets wrong.",
                        Schema(("symbol", "string", "Symbol name, optionally qualified: Run, InferenceEngine.Run, DevOnBike.Overfit.InferenceEngine"))),
                    Tool(
                        "find_implementations",
                        "Types implementing an interface, or classes derived from a class (transitively).",
                        Schema(("symbol", "string", "Interface, interface member, or base class name"))),
                    Tool(
                        "find_callers",
                        "What reaches a symbol, walked transitively up the call graph. Answers 'is this on the hot "
                        + "path' and 'what breaks if I change this signature'.",
                        Schema(
                            ("symbol", "string", "Method or property name"),
                            ("depth", "integer", "How many levels of callers to walk (default 3)"))),
                    Tool(
                        "find_unused",
                        "Symbols in one project that nothing references, plus ones referenced only by tests. "
                        + "Excludes overrides, interface implementations and attributed symbols, for which a zero "
                        + "reference count does not mean unused. Public symbols are excluded by default because "
                        + "this library's public callers live outside the repository.",
                        Schema(
                            ("project", "string", "Project name, e.g. Main, Anomalies, Cli"),
                            ("includePublic", "boolean", "Also consider public symbols (default false)"))));
            }

            public static async Task<string> InvokeAsync(
                string name,
                JsonObject arguments,
                WorkspaceLoader loader,
                NavigatorQueries queries,
                CancellationToken cancellationToken)
            {
                if (name == "find_unused")
                {
                    return await RunUnusedAsync(arguments, loader, queries, cancellationToken).ConfigureAwait(false);
                }

                var symbolName = arguments["symbol"]?.GetValue<string>()
                    ?? throw new ArgumentException("missing required argument 'symbol'");

                var symbols = await SymbolResolver.ResolveAsync(loader.Solution, symbolName, cancellationToken)
                    .ConfigureAwait(false);

                if (symbols.Count == 0)
                {
                    return $"No source symbol named '{symbolName}'. Note that only symbols declared in this "
                        + "solution are searched — framework types are not.";
                }

                var symbol = symbols[0];
                var builder = new StringBuilder();

                if (symbols.Count > 1 && symbols[0].ToDisplayString() != symbols[1].ToDisplayString())
                {
                    builder.AppendLine($"'{symbolName}' matched {symbols.Count} symbols; answering for the first. Qualify to pick another:");

                    foreach (var candidate in symbols)
                    {
                        builder.AppendLine($"  {SymbolResolver.Describe(candidate)}   {SymbolResolver.DescribeLocation(candidate)}");
                    }

                    builder.AppendLine();
                }

                builder.AppendLine($"{SymbolResolver.Describe(symbol)}");
                builder.AppendLine($"declared at {SymbolResolver.DescribeLocation(symbol)}");
                builder.AppendLine();

                if (name == "find_references")
                {
                    var references = await queries.FindReferencesAsync(symbol, cancellationToken).ConfigureAwait(false);

                    foreach (var reference in references)
                    {
                        var marker = reference.IsTest ? "[test] " : "       ";
                        builder.AppendLine($"{marker}{reference.Location}   in {reference.InSymbol}");
                    }

                    builder.AppendLine();
                    builder.AppendLine($"{references.Count} reference(s)");
                    return builder.ToString();
                }

                if (name == "find_implementations")
                {
                    var implementations = await queries.FindImplementationsAsync(symbol, cancellationToken)
                        .ConfigureAwait(false);

                    foreach (var implementation in implementations)
                    {
                        builder.AppendLine($"  {implementation}");
                    }

                    builder.AppendLine();
                    builder.AppendLine($"{implementations.Count} implementation(s)");
                    return builder.ToString();
                }

                if (name == "find_callers")
                {
                    var depth = arguments["depth"]?.GetValue<int>() ?? 3;
                    var edges = await queries.FindCallersAsync(symbol, depth, cancellationToken).ConfigureAwait(false);

                    foreach (var edge in edges)
                    {
                        var indirect = edge.IsDirect ? string.Empty : "  (indirect)";
                        builder.AppendLine($"{new string(' ', edge.Depth * 2)}<- {edge.Caller}   {edge.Location}{indirect}");
                    }

                    builder.AppendLine();
                    builder.AppendLine($"{edges.Count} call site(s) within {depth} level(s)");
                    return builder.ToString();
                }

                throw new ArgumentException($"unknown tool '{name}'");
            }

            private static async Task<string> RunUnusedAsync(
                JsonObject arguments,
                WorkspaceLoader loader,
                NavigatorQueries queries,
                CancellationToken cancellationToken)
            {
                var projectName = arguments["project"]?.GetValue<string>()
                    ?? throw new ArgumentException("missing required argument 'project'");

                var includePublic = arguments["includePublic"]?.GetValue<bool>() ?? false;
                Project? project = null;

                foreach (var candidate in loader.Solution.Projects)
                {
                    if (string.Equals(candidate.Name, projectName, StringComparison.OrdinalIgnoreCase))
                    {
                        project = candidate;
                        break;
                    }
                }

                if (project is null)
                {
                    throw new ArgumentException($"no project named '{projectName}'");
                }

                var candidates = await queries.FindUnusedAsync(project, includePublic, cancellationToken)
                    .ConfigureAwait(false);

                var builder = new StringBuilder();

                foreach (var candidate in candidates)
                {
                    builder.AppendLine($"{candidate.Verdict,-32} {candidate.Symbol}");
                    builder.AppendLine($"{string.Empty,-32} {candidate.Location}");
                }

                builder.AppendLine();
                builder.AppendLine($"{candidates.Count} candidate(s) in {project.Name}");
                builder.AppendLine("Overrides, interface implementations and attributed symbols were excluded: a zero");
                builder.AppendLine("reference count does not mean unused for those.");
                return builder.ToString();
            }

            private static JsonObject Tool(string name, string description, JsonObject schema)
            {
                return new JsonObject
                {
                    ["name"] = name,
                    ["description"] = description,
                    ["inputSchema"] = schema,
                };
            }

            private static JsonObject Schema(params (string Name, string Type, string Description)[] properties)
            {
                var props = new JsonObject();
                var required = new JsonArray();

                foreach (var (name, type, description) in properties)
                {
                    props[name] = new JsonObject
                    {
                        ["type"] = type,
                        ["description"] = description,
                    };

                    // Only the first property is required; the rest are optional switches with defaults.
                    if (required.Count == 0)
                    {
                        required.Add(name);
                    }
                }

                return new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = props,
                    ["required"] = required,
                };
            }
        }
    }
}
