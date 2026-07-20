using DevOnBike.Overfit.Extensions.AI;
using DevOnBike.Overfit.LanguageModels;
using Microsoft.Extensions.AI;

var builder = WebApplication.CreateBuilder(args);

// The whole point of this template: a local GGUF model, loaded IN THIS PROCESS, exposed as a standard
// Microsoft.Extensions.AI IChatClient. No Python, no Ollama, no Docker, no cloud key. This is the same
// IChatClient the official .NET AI template consumes — so everything downstream (function calling,
// caching, telemetry, Semantic Kernel) works unchanged; the model just runs locally on the CPU.
var modelPath = builder.Configuration["ModelPath"] ?? "model.gguf";
if (!File.Exists(modelPath))
{
    Console.Error.WriteLine(
        $"Model file not found: {Path.GetFullPath(modelPath)}\n\n" +
        "Download a small GGUF (~400 MB) and point \"ModelPath\" (appsettings.json) at it, e.g.:\n" +
        "  curl -L -o model.gguf https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf\n\n" +
        "…then run again.");
    return;
}

// Load once at startup. LoadGguf memory-maps the weights, so the managed heap stays tiny. The adapter
// serializes concurrent calls internally, so registering one shared client as a singleton IChatClient is
// safe (requests queue). For real multi-user throughput, load N clients or use a session pool.
var overfit = OverfitClient.LoadGguf(modelPath, mmap: true);

builder.Services.AddSingleton(overfit);
builder.Services.AddChatClient(overfit.AsChatClient());   // ← Overfit IS your IChatClient now.

var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();

// Streaming chat over the standard IChatClient — identical in shape to the .NET AI template's endpoint,
// except the tokens are generated locally. Streams plain-text token deltas to the browser.
app.MapPost("/chat", async (ChatInput input, IChatClient chat, HttpContext ctx, CancellationToken ct) =>
{
    ctx.Response.ContentType = "text/plain; charset=utf-8";
    await foreach (var update in chat.GetStreamingResponseAsync(input.Message, cancellationToken: ct))
    {
        await ctx.Response.WriteAsync(update.Text ?? string.Empty, ct);
        await ctx.Response.Body.FlushAsync(ct);
    }
});

app.Run();

internal sealed record ChatInput(string Message);
