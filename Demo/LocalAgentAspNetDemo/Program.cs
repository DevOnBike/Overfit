// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

// Overfit Local Agent Starter.
//
// What this is: a Minimal-API ASP.NET host that loads a local GGUF model via
// OverfitClient.LoadGguf and runs a private agent in-process. No Python, no
// model server, no native binary, no data leaving the process. This file is a
// reference template a .NET developer can drop into an existing service — see
// README.md for the 5-minute quickstart.
//
// Phase 1 (chat):    /health, /chat, /reset.
// Phase 2 (RAG):     /documents/index, /rag/query over an in-process VectorStore.
// Phase 3 (agent):   /agent (forced C# tool call), /chat/json (guaranteed JSON).
// Phase 4 (metrics): /metrics (Prometheus), Dockerfile + compose.yaml.
// Phase 5 (OpenAI):  /v1/chat/completions (+ SSE stream), /v1/embeddings, /v1/models — point any
//                    OpenAI client/SDK at this base URL; the in-process .NET runtime serves it.

using System.Diagnostics;
using System.Text.Json.Nodes;
using DevOnBike.Overfit.Demo.LocalAgent.Agent;
using DevOnBike.Overfit.Demo.LocalAgent.Chat;
using DevOnBike.Overfit.Demo.LocalAgent.Infrastructure;
using DevOnBike.Overfit.Demo.LocalAgent.Observability;
using DevOnBike.Overfit.Demo.LocalAgent.OpenAi;
using DevOnBike.Overfit.Demo.LocalAgent.Rag;
using DevOnBike.Overfit.Demo.LocalAgent.Swagger;
using DevOnBike.Overfit.Demo.LocalAgent.Tools;
using DevOnBike.Overfit.LanguageModels;
using OpenTelemetry.Metrics;

namespace DevOnBike.Overfit.Demo.LocalAgent
{
    /// <summary>
    /// Entry point for the Local Agent ASP.NET host. Classic <see cref="Main"/> structure
    /// (not top-level statements) so the bootstrap is explicit and easy to follow.
    /// </summary>
    public static class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Resolve the model before building services so a misconfigured run fails fast with a clear
            // message. Accepts either a *.gguf file or a HuggingFace directory (model.safetensors). The demo
            // defaults to Qwen2.5-3B Q4_K_M because it routes tools RELIABLY; a 0.5B is ~2x faster and does
            // chat/RAG/JSON, but its tool selection is below par (see README "Model choice").
            var modelPath = ModelPathResolver.Resolve(builder.Configuration);
            var systemMessage =
                builder.Configuration.GetValue<string>("SystemMessage")
                ?? "You are a concise, helpful assistant running locally inside a .NET process. "
                   + "Answer only from context the user provides; if you are unsure, say so.";

            // Load the model eagerly at startup so load time is measured deterministically and the first
            // request isn't slow. The chat session is stateful across requests — this is a single-tenant demo,
            // not safe for multi-user concurrent conversations. For multi-tenant, swap to a per-tenant
            // client-pool / session-per-request.
            var isGguf = modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
            var modelDisplay = isGguf
                ? Path.GetFileName(modelPath)
                : $"{Path.GetFileName(modelPath.TrimEnd('\\', '/'))} (safetensors)";

            // Log which model is being loaded — the environment name distinguishes the default English demo
            // from the Bielik Polish preset. A bootstrap logger because this runs before the host is built.
            using var startupLoggerFactory = LoggerFactory.Create(b => b.AddSimpleConsole(o => o.SingleLine = true));
            var startupLog = startupLoggerFactory.CreateLogger("Overfit.LocalAgent");
            startupLog.LogInformation("Environment '{Env}': loading {Kind} model {Model} from {Path} ...",
                builder.Environment.EnvironmentName, isGguf ? "GGUF" : "safetensors", modelDisplay, modelPath);

            var loadStopwatch = Stopwatch.StartNew();
            var sharedClient = isGguf
                ? OverfitClient.LoadGguf(modelPath, mmap: true)
                : OverfitClient.LoadPretrained(modelPath);   // HuggingFace safetensors directory
            sharedClient.AddSystem(systemMessage);
            loadStopwatch.Stop();
            startupLog.LogInformation("Model loaded: {Model} in {Sec:F2}s (mmap={Mmap}).",
                modelDisplay, loadStopwatch.Elapsed.TotalSeconds, isGguf);

            builder.Services.AddSingleton(sharedClient);

            // Observability (Phase 4). Instrumented with the built-in System.Diagnostics.Metrics (Meter) API and
            // exported to a Prometheus scrape endpoint at /metrics by OpenTelemetry (configured below). Static
            // build info captured once; per-generation stats recorded by the endpoints.
            var modelFile = isGguf ? modelPath : Path.Combine(modelPath, "model.safetensors");
            builder.Services.AddSingleton(new MetricsCollector
            {
                ModelFile = modelDisplay,
                ModelFingerprint = MetricsCollector.FingerprintModel(modelFile),
                MmapEnabled = isGguf,
                ModelLoadSeconds = Math.Round(loadStopwatch.Elapsed.TotalSeconds, 3),
            });

            builder.Services.AddOpenTelemetry().WithMetrics(metrics =>
            {
                metrics.AddMeter(MetricsCollector.MeterName);
                // Latency-shaped buckets for the retrieval histogram (defaults start at 0 then jump to 5s, useless
                // for ~10-50 ms searches); the rest keep their defaults.
                metrics.AddView(
                    "overfit.rag.search",
                    new ExplicitBucketHistogramConfiguration
                    {
                        Boundaries = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
                    });
                metrics.AddPrometheusExporter();
            });

            // Audit trail (production gate): append-only metadata log of every request — never prompt/response
            // content. Mirrors to the structured logger; writes JSONL to 'AuditLogPath' when configured.
            builder.Services.AddSingleton<AuditLog>();

            // RAG over local documents (Phase 2). Loads its embedder lazily on first /documents/index,
            // so a run without an embedding model still serves /chat — RAG just returns a clear error.
            builder.Services.AddSingleton<RagService>();

            // Tool calling + guaranteed JSON (Phase 3). The registry holds the C# tools and their in-memory
            // back-office data; AgentService drives the constrained-decoding flows.
            builder.Services.AddSingleton<ToolRegistry>();
            builder.Services.AddSingleton<AgentService>();

            // Swagger UI — explore and exercise every endpoint from a browser at /swagger. The operation filter
            // pre-fills each request body with a ready-to-run example (Try it out → Execute, no typing needed).
            builder.Services.AddEndpointsApiExplorer();
            // Pre-filled request examples — in Polish when the preset asks for it (ExamplesLanguage: "pl",
            // set by appsettings.Bielik.json), English otherwise.
            var polishExamples = string.Equals(
                builder.Configuration.GetValue<string>("ExamplesLanguage"), "pl", StringComparison.OrdinalIgnoreCase);
            builder.Services.AddSwaggerGen(o => o.OperationFilter<RequestExamplesFilter>(polishExamples));

            var app = builder.Build();

            app.UseSwagger();
            app.UseSwaggerUI(o =>
            {
                o.DocumentTitle = "Overfit Local Agent";
                o.RoutePrefix = "swagger";
            });

            // Audit OUTERMOST so it records 401s too; then the API-key gate. Both no-op on probe/docs paths and
            // when unconfigured, so the demo still runs out of the box.
            app.UseMiddleware<AuditMiddleware>();
            app.UseMiddleware<ApiKeyAuthMiddleware>();

            // Dispose the engine + session cleanly on shutdown so file-mapped weights and
            // KV-cache buffers release before the process exits. Without this, a hosted
            // run that re-binds the port can leak the underlying engine for a moment.
            app.Lifetime.ApplicationStopping.Register(() =>
            {
                app.Services.GetService<AuditLog>()?.Dispose();
                app.Services.GetService<RagService>()?.Dispose();
                app.Services.GetService<OverfitClient>()?.Dispose();
            });

            app.MapGet("/", () => Results.Redirect("/swagger"));

            app.MapGet("/health", (RagService rag, MetricsCollector metrics) =>
            {
                return Results.Ok(new
                {
                    status = "ok",
                    model = metrics.ModelFile,
                    modelPath,
                    modelFingerprint = metrics.ModelFingerprint,
                    modelLoadSeconds = metrics.ModelLoadSeconds,
                    mmap = metrics.MmapEnabled,
                    runtime = ".NET " + Environment.Version,
                    privateToProcess = true,
                    rag = new
                    {
                        indexed = rag.IsIndexed,
                        chunks = rag.ChunkCount
                    },
                });
            });

            // Kubernetes-style split probes (the rich /health above stays for humans):
            //   /healthz = liveness  — the process is up (never touches the model, so a long generation can't fail it).
            //   /readyz  = readiness — the model is loaded and the host can serve (loaded eagerly at startup).
            app.MapGet("/healthz", () => Results.Ok(new { status = "live" }));
            app.MapGet("/readyz", (OverfitClient client) =>
                client is not null
                    ? Results.Ok(new { status = "ready" })
                    : Results.Json(new { status = "loading" }, statusCode: StatusCodes.Status503ServiceUnavailable));

            app.MapPost("/chat", (ChatRequest req, OverfitClient client, MetricsCollector metrics) =>
            {
                if (string.IsNullOrWhiteSpace(req.Message))
                {
                    return Results.BadRequest(new
                    {
                        error = "'message' is required and must be non-empty."
                    });
                }

                var reply = client.Send(req.Message);
                var stats = client.Chat.LastStats;
                metrics.RecordGeneration("chat", stats);

                return Results.Ok(new ChatReply(
                    Reply: reply,
                    Stats: new ChatStats(
                        PromptTokens: stats.PromptTokens,
                        GeneratedTokens: stats.GeneratedTokens,
                        TokensPerSecond: Math.Round(stats.TokensPerSecond, 2),
                        AllocatedBytes: stats.AllocatedBytes,
                        UsedKeyValueCache: stats.UsedKeyValueCache)));
            });

            app.MapPost("/reset", (OverfitClient client) =>
            {
                client.Reset();
                client.AddSystem(systemMessage);
                return Results.Ok(new
                {
                    status = "conversation cleared"
                });
            });

            // ── RAG endpoints (Phase 2) ───────────────────────────────────────────────

            app.MapPost("/documents/index", (RagService rag) =>
            {
                try
                {
                    var summary = rag.IndexDocuments();
                    return Results.Ok(summary);
                }
                catch (InvalidOperationException ex)
                {
                    // Missing embedding model / empty data dir — actionable client error, not a 500.
                    return Results.Problem(detail: ex.Message, statusCode: StatusCodes.Status400BadRequest);
                }
            });

            app.MapPost("/rag/query", (RagQueryRequest req, HttpContext httpContext, OverfitClient client, RagService rag, MetricsCollector metrics) =>
            {
                if (string.IsNullOrWhiteSpace(req.Question))
                {
                    return Results.BadRequest(new
                    {
                        error = "'question' is required and must be non-empty."
                    });
                }

                var topK = req.TopK is > 0 ? Math.Min(req.TopK.Value, 10) : 4;

                try
                {
                    var answer = rag.Query(client, req.Question, topK);
                    // Audit which sources were retrieved (ids only — never the snippet content).
                    httpContext.Items["audit.sources"] = answer.Sources.Select(s => s.Id).ToArray();
                    metrics.RecordRagSearch(answer.SearchSeconds);
                    metrics.RecordGeneration("rag", client.Chat.LastStats);
                    return Results.Ok(answer);
                }
                catch (InvalidOperationException ex)
                {
                    return Results.Problem(detail: ex.Message, statusCode: StatusCodes.Status400BadRequest);
                }
            });

            // RAG Stability Harness — "RAG is testable". Deterministic retrieval-side evaluation (no LLM call):
            // expected-source recall, paraphrase stability, false-premise traps, + corpus lint. Gate it in CI.
            app.MapPost("/rag/eval", (RagEvalRequest req, RagService rag) =>
            {
                try
                {
                    return Results.Ok(rag.Evaluate(req));
                }
                catch (InvalidOperationException ex)
                {
                    return Results.Problem(detail: ex.Message, statusCode: StatusCodes.Status400BadRequest);
                }
            });

            // ── Agent endpoints (Phase 3): tool calling + guaranteed JSON ──────────────

            app.MapPost("/agent", (ToolCallRequest req, HttpContext httpContext, OverfitClient client, AgentService agent, MetricsCollector metrics) =>
            {
                if (string.IsNullOrWhiteSpace(req.Message))
                {
                    return Results.BadRequest(new
                    {
                        error = "'message' is required and must be non-empty."
                    });
                }

                var result = agent.RunToolCall(client, req.Message);
                httpContext.Items["audit.tool"] = result.ToolName;   // audit which C# tool the model invoked
                metrics.RecordGeneration("agent", client.Chat.LastStats);
                metrics.RecordToolCall(result.ToolName);

                // arguments + result are raw JSON strings (the constraint guarantees the arguments are valid JSON;
                // the handler emits JSON). Embed them as real JSON nodes so the response isn't double-escaped.
                return Results.Ok(new
                {
                    toolName = result.ToolName,
                    arguments = JsonNode.Parse(result.Arguments),
                    result = JsonNode.Parse(result.Result),
                });
            });

            app.MapPost("/chat/json", (ChatRequest req, OverfitClient client, AgentService agent, MetricsCollector metrics) =>
            {
                if (string.IsNullOrWhiteSpace(req.Message))
                {
                    return Results.BadRequest(new
                    {
                        error = "'message' is required and must be non-empty."
                    });
                }

                string json;
                try
                {
                    // With req.Schema set, the reply is constrained to CONFORM to that JSON-Schema (typed /
                    // required / enum fields); without it, just guaranteed well-formed JSON.
                    json = agent.RunJson(client, req.Message, req.Schema).Json;
                }
                catch (System.Text.Json.JsonException ex)
                {
                    return Results.Problem(detail: $"Invalid 'schema' (not valid JSON-Schema): {ex.Message}",
                        statusCode: StatusCodes.Status400BadRequest);
                }
                metrics.RecordGeneration("chat_json", client.Chat.LastStats);

                // The reply is guaranteed well-formed (and, with a schema, schema-conforming) JSON — return verbatim.
                return Results.Content(json, "application/json");
            });

            // A business-process flavour of guaranteed JSON: a free-text refund/withdrawal scenario in, a
            // structured { eligible, reason, requiredAction, confidence } decision out — well-formed by
            // construction. Shows "the model drives a real decision", not just chat.
            app.MapPost("/decision/refund", (ChatRequest req, OverfitClient client, AgentService agent, MetricsCollector metrics) =>
            {
                if (string.IsNullOrWhiteSpace(req.Message))
                {
                    return Results.BadRequest(new
                    {
                        error = "'message' is required and must be non-empty."
                    });
                }

                var json = agent.RunRefundDecision(client, req.Message).Json;

                metrics.RecordGeneration("decision_refund", client.Chat.LastStats);

                return Results.Content(json, "application/json");
            });

            // ── OpenAI-compatible API (Phase 5): /v1/chat/completions, /v1/embeddings, /v1/models ──────
            // Point any OpenAI client/SDK/tool at this host's base URL (model name is echoed, not selected —
            // the one loaded model is served). Stateless per request, serialized through a single-flight gate.
            app.MapOpenAiApi(modelDisplay, systemMessage);

            // ── Metrics (Phase 4): Prometheus scrape endpoint ─────────────────────────
            // OpenTelemetry serves the Prometheus exposition at /metrics from the Meter instruments recorded by
            // MetricsCollector — the idiomatic ASP.NET Core path, no hand-rolled text.
            app.MapPrometheusScrapingEndpoint();

            app.Run();
        }
    }
}
