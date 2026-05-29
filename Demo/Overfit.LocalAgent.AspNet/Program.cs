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

using System.Diagnostics;
using System.Text.Json.Nodes;
using DevOnBike.Overfit.Demo.LocalAgent.Agent;
using DevOnBike.Overfit.Demo.LocalAgent.Observability;
using DevOnBike.Overfit.Demo.LocalAgent.Rag;
using DevOnBike.Overfit.Demo.LocalAgent.Tools;
using DevOnBike.Overfit.LanguageModels;

var builder = WebApplication.CreateBuilder(args);

// Resolve the GGUF model path before building services so a misconfigured run
// fails fast with a clear message, not deep inside DI resolution.
var modelPath = ModelPathResolver.Resolve(builder.Configuration);
var systemMessage =
    builder.Configuration.GetValue<string>("SystemMessage")
    ?? "You are a concise, helpful assistant running locally inside a .NET process. "
       + "Answer only from context the user provides; if you are unsure, say so.";

// Load the model eagerly at startup so load time is measured deterministically and the first
// request isn't slow. The chat session is stateful across requests — this is a single-tenant demo,
// not safe for multi-user concurrent conversations. For multi-tenant, swap to a per-tenant
// client-pool / session-per-request.
var loadStopwatch = Stopwatch.StartNew();
var sharedClient = OverfitClient.LoadGguf(modelPath, mmap: true);
sharedClient.AddSystem(systemMessage);
loadStopwatch.Stop();

builder.Services.AddSingleton(sharedClient);

// Observability (Phase 4). Static build info captured once; per-generation stats recorded by the
// endpoints. Exposed at /metrics in Prometheus text format (scraped by the compose Prometheus service).
builder.Services.AddSingleton(new MetricsCollector
{
    ModelFile = Path.GetFileName(modelPath),
    ModelFingerprint = MetricsCollector.FingerprintModel(modelPath),
    MmapEnabled = true,
    ModelLoadSeconds = Math.Round(loadStopwatch.Elapsed.TotalSeconds, 3),
});

// RAG over local documents (Phase 2). Loads its embedder lazily on first /documents/index,
// so a run without an embedding model still serves /chat — RAG just returns a clear error.
builder.Services.AddSingleton<RagService>();

// Tool calling + guaranteed JSON (Phase 3). The registry holds the C# tools and their in-memory
// back-office data; AgentService drives the constrained-decoding flows.
builder.Services.AddSingleton<ToolRegistry>();
builder.Services.AddSingleton<AgentService>();

// Swagger UI — explore and exercise every endpoint from a browser at /swagger.
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI(o =>
{
    o.DocumentTitle = "Overfit Local Agent";
    o.RoutePrefix = "swagger";
});

// Dispose the engine + session cleanly on shutdown so file-mapped weights and
// KV-cache buffers release before the process exits. Without this, a hosted
// run that re-binds the port can leak the underlying engine for a moment.
app.Lifetime.ApplicationStopping.Register(() =>
{
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
        rag = new { indexed = rag.IsIndexed, chunks = rag.ChunkCount },
    });
});

app.MapPost("/chat", (ChatRequest req, OverfitClient client, MetricsCollector metrics) =>
{
    if (string.IsNullOrWhiteSpace(req.Message))
    {
        return Results.BadRequest(new { error = "'message' is required and must be non-empty." });
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
    return Results.Ok(new { status = "conversation cleared" });
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

app.MapPost("/rag/query", (RagQueryRequest req, OverfitClient client, RagService rag, MetricsCollector metrics) =>
{
    if (string.IsNullOrWhiteSpace(req.Question))
    {
        return Results.BadRequest(new { error = "'question' is required and must be non-empty." });
    }

    var topK = req.TopK is > 0 ? Math.Min(req.TopK.Value, 10) : 4;

    try
    {
        var answer = rag.Query(client, req.Question, topK);
        metrics.RecordRagSearch(answer.SearchSeconds);
        metrics.RecordGeneration("rag", client.Chat.LastStats);
        return Results.Ok(answer);
    }
    catch (InvalidOperationException ex)
    {
        return Results.Problem(detail: ex.Message, statusCode: StatusCodes.Status400BadRequest);
    }
});

// ── Agent endpoints (Phase 3): tool calling + guaranteed JSON ──────────────

app.MapPost("/agent", (ToolCallRequest req, OverfitClient client, AgentService agent, MetricsCollector metrics) =>
{
    if (string.IsNullOrWhiteSpace(req.Message))
    {
        return Results.BadRequest(new { error = "'message' is required and must be non-empty." });
    }

    var result = agent.RunToolCall(client, req.Message);
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
        return Results.BadRequest(new { error = "'message' is required and must be non-empty." });
    }

    var json = agent.RunJson(client, req.Message).Json;
    metrics.RecordGeneration("chat_json", client.Chat.LastStats);

    // The reply is guaranteed well-formed JSON by construction — return it verbatim as application/json.
    return Results.Content(json, "application/json");
});

// ── Metrics (Phase 4): Prometheus text exposition ─────────────────────────

app.MapGet("/metrics", (MetricsCollector metrics) =>
    Results.Text(metrics.ToPrometheus(), "text/plain; version=0.0.4"));

app.Run();
return;

internal static class ModelPathResolver
{
    public static string Resolve(IConfiguration config)
    {
        // 1) appsettings.json `ModelPath` — absolute path to a *.gguf.
        var fromSettings = config.GetValue<string>("ModelPath");
        if (!string.IsNullOrWhiteSpace(fromSettings) && File.Exists(fromSettings))
        {
            return fromSettings;
        }

        // 2) Env var `OVERFIT_MODEL_DIR` pointing at a directory holding one *.gguf.
        var fromEnv = Environment.GetEnvironmentVariable("OVERFIT_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(fromEnv) && Directory.Exists(fromEnv))
        {
            var ggufs = Directory.GetFiles(fromEnv, "*.gguf");
            if (ggufs.Length > 0)
            {
                return ggufs[0];
            }
        }

        throw new InvalidOperationException(
            "Could not locate a GGUF model. Either: " +
            "(a) set 'ModelPath' in appsettings.json to an absolute *.gguf file " +
            "(e.g. C:\\qwen3b\\qwen.q4km.gguf), or " +
            "(b) set the OVERFIT_MODEL_DIR environment variable to a directory " +
            "containing exactly one *.gguf. See Demo/Overfit.LocalAgent.AspNet/README.md.");
    }
}

internal record ChatRequest(string Message);

internal record ChatReply(string Reply, ChatStats Stats);

internal record ChatStats(
    int PromptTokens,
    int GeneratedTokens,
    double TokensPerSecond,
    long AllocatedBytes,
    bool UsedKeyValueCache);
