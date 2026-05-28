// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

// Overfit Local Agent Starter — Phase 1 walking skeleton.
//
// What this is: a Minimal-API ASP.NET host that loads a local GGUF model via
// OverfitClient.LoadGguf and exposes /health + /chat + /reset. No Python, no
// model server, no native binary, no data leaving the process. This file is a
// reference template a .NET developer can drop into an existing service to add
// a private agent — see README.md for the 5-minute quickstart.
//
// Phase 1 scope: prove the wire-up. Phases 2-4 (RAG over local docs, C# tool
// calling, guaranteed JSON output, /metrics with tokens/sec + allocations/token)
// layer on top of this skeleton.

using DevOnBike.Overfit.LanguageModels;

var builder = WebApplication.CreateBuilder(args);

// Resolve the GGUF model path before building services so a misconfigured run
// fails fast with a clear message, not deep inside DI resolution.
var modelPath = ModelPathResolver.Resolve(builder.Configuration);
var systemMessage =
    builder.Configuration.GetValue<string>("SystemMessage")
    ?? "You are a concise, helpful assistant running locally inside a .NET process. "
       + "Answer only from context the user provides; if you are unsure, say so.";

// Singleton OverfitClient. The chat session is stateful across requests — this
// is a single-tenant demo and not safe for multi-user concurrent conversations.
// For multi-tenant, swap to a per-tenant client-pool / session-per-request.
builder.Services.AddSingleton(_ =>
{
    var client = OverfitClient.LoadGguf(modelPath);
    client.AddSystem(systemMessage);
    return client;
});

var app = builder.Build();

// Dispose the engine + session cleanly on shutdown so file-mapped weights and
// KV-cache buffers release before the process exits. Without this, a hosted
// run that re-binds the port can leak the underlying engine for a moment.
app.Lifetime.ApplicationStopping.Register(() =>
{
    app.Services.GetService<OverfitClient>()?.Dispose();
});

app.MapGet("/", () => Results.Redirect("/health"));

app.MapGet("/health", (OverfitClient client) =>
{
    return Results.Ok(new
    {
        status = "ok",
        model = Path.GetFileName(modelPath),
        modelPath,
        runtime = ".NET " + Environment.Version,
        privateToProcess = true,
        phase = "walking-skeleton",
    });
});

app.MapPost("/chat", (ChatRequest req, OverfitClient client) =>
{
    if (string.IsNullOrWhiteSpace(req.Message))
    {
        return Results.BadRequest(new { error = "'message' is required and must be non-empty." });
    }

    var reply = client.Send(req.Message);
    var stats = client.Chat.LastStats;

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
