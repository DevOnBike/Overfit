// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Server.AspNet.Endpoints;
using DevOnBike.Overfit.Server.AspNet.Services;
using DevOnBike.Overfit.Server.OpenAi;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Tests.Server
{
    /// <summary>
    /// Integration tests for the AOT-ready ASP.NET host, driven through Microsoft's in-memory
    /// <see cref="TestServer"/> (<c>Microsoft.AspNetCore.TestHost</c>) — a real request pipeline, no sockets,
    /// no model. The engine is faked behind <see cref="IOpenAiInferenceService"/> (the reason the logic sits
    /// behind an interface), so these exercise the whole HTTP surface — routing, the source-gen JSON binding,
    /// the SSE framing, status codes, the docs endpoints — and run anywhere, including GitHub CI where no GGUF
    /// exists. The real-model path is covered separately by <c>[SmallModelFact]</c>/<c>[LongFact]</c> tests.
    /// </summary>
    public sealed class OverfitAspNetServerIntegrationTests
    {
        private static async Task<(WebApplication App, HttpClient Client)> StartAsync(IOpenAiInferenceService service)
        {
            var builder = WebApplication.CreateSlimBuilder();
            builder.WebHost.UseTestServer();
            builder.Logging.ClearProviders();
            builder.Services.AddOverfitOpenAi(service, new ServerMetrics());

            var app = builder.Build();
            app.MapOverfitOpenAiApi();
            await app.StartAsync();
            return (app, app.GetTestClient());
        }

        [Fact]
        public async Task Health_ReturnsOk()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var resp = await client.GetAsync("/health");
                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
                Assert.Equal("ok", await resp.Content.ReadAsStringAsync());
            }
        }

        [Fact]
        public async Task Models_ReportsTheServedModelId()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var doc = await client.GetFromJsonAsync<JsonElement>("/v1/models");
                Assert.Equal("fake-model", doc.GetProperty("data")[0].GetProperty("id").GetString());
            }
        }

        [Fact]
        public async Task ChatCompletions_NonStreaming_ReturnsAssistantContent()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var resp = await client.PostAsync("/v1/chat/completions", JsonBody(
                    """{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}"""));

                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
                var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
                var msg = doc.RootElement.GetProperty("choices")[0].GetProperty("message");
                Assert.Equal("assistant", msg.GetProperty("role").GetString());
                Assert.Equal("pong", msg.GetProperty("content").GetString());
            }
        }

        [Fact]
        public async Task ChatCompletions_Streaming_EmitsSseFramesAndDone()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var resp = await client.PostAsync("/v1/chat/completions", JsonBody(
                    """{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"""));

                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
                Assert.Equal("text/event-stream", resp.Content.Headers.ContentType?.MediaType);

                var body = await resp.Content.ReadAsStringAsync();
                var frames = body.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
                Assert.Contains("\"content\":\"po\"", body);
                Assert.Contains("\"content\":\"ng\"", body);
                Assert.EndsWith("data: [DONE]", frames[^1].Trim());
            }
        }

        [Fact]
        public async Task ChatCompletions_MalformedJson_Returns400()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var resp = await client.PostAsync("/v1/chat/completions",
                    new StringContent("{ this is not json", Encoding.UTF8, "application/json"));

                Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
                var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
                Assert.Contains("invalid JSON", doc.RootElement.GetProperty("error").GetProperty("message").GetString());
            }
        }

        [Fact]
        public async Task Embeddings_ReturnsVectorFromTheService()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var resp = await client.PostAsync("/v1/embeddings", JsonBody(
                    """{"model":"m","input":"hello"}"""));

                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
                var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
                var vector = doc.RootElement.GetProperty("data")[0].GetProperty("embedding");
                Assert.Equal(3, vector.GetArrayLength());
            }
        }

        [Fact]
        public async Task Embeddings_WhenServiceHasNoModel_Returns501()
        {
            var (app, client) = await StartAsync(new FakeInferenceService { EmbeddingsAvailable = false });
            await using (app)
            {
                var resp = await client.PostAsync("/v1/embeddings", JsonBody("""{"input":"hello"}"""));
                Assert.Equal(HttpStatusCode.NotImplemented, resp.StatusCode);
            }
        }

        [Fact]
        public async Task Docs_ServeOpenApiYamlAndApiReference()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var yaml = await client.GetAsync("/openapi.yaml");
                Assert.Equal(HttpStatusCode.OK, yaml.StatusCode);
                Assert.Contains("openapi", await yaml.Content.ReadAsStringAsync());

                var docs = await client.GetAsync("/docs");
                Assert.Equal(HttpStatusCode.OK, docs.StatusCode);
                Assert.Equal("text/html", docs.Content.Headers.ContentType?.MediaType);
            }
        }

        [Fact]
        public async Task Metrics_AreOrderedByName_WithEachFamilyIntact()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var body = await (await client.GetAsync("/metrics")).Content.ReadAsStringAsync();
                var lines = body.Split('\n');

                var names = new List<string>();
                var helped = new HashSet<string>(StringComparer.Ordinal);

                foreach (var line in lines)
                {
                    if (line.StartsWith("# HELP ", StringComparison.Ordinal))
                    {
                        helped.Add(line.Split(' ')[2]);
                    }

                    if (line.StartsWith("# TYPE ", StringComparison.Ordinal))
                    {
                        names.Add(line.Split(' ')[2]);
                    }
                }

                Assert.NotEmpty(names);

                // Sorted by name. Prometheus does not require this; two readers do — a human diffing
                // /metrics between two replicas, which is the premise of the peer comparison this server is
                // instrumented for, and anyone scanning the endpoint for a name they expect to find.
                for (var i = 1; i < names.Count; i++)
                {
                    Assert.True(
                        string.CompareOrdinal(names[i - 1], names[i]) < 0,
                        $"metric families are out of order: '{names[i - 1]}' precedes '{names[i]}'");
                }

                // Sorting has to happen per family, never per line: every TYPE must still be preceded by its
                // own HELP. A flat sort of the rendered text would satisfy the check above and destroy this
                // one, which is exactly the mistake worth guarding against.
                foreach (var name in names)
                {
                    Assert.Contains(name, helped);
                }
            }
        }

        [Fact]
        public async Task Metrics_ExposePrometheusProcessMetrics()
        {
            var (app, client) = await StartAsync(new FakeInferenceService());
            await using (app)
            {
                var resp = await client.GetAsync("/metrics");
                Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
                Assert.StartsWith("text/plain", resp.Content.Headers.ContentType?.MediaType ?? "");

                var body = await resp.Content.ReadAsStringAsync();
                // Prometheus exposition format: HELP/TYPE lines + the memory + CPU series the user asked for.
                Assert.Contains("# TYPE process_resident_memory_bytes gauge", body);
                Assert.Contains("# TYPE process_cpu_seconds_total counter", body);
                Assert.Contains("dotnet_gc_collections_total{generation=\"0\"}", body);

                // Server metrics: requests, tokens (rate() -> tokens/s), and live session-pool gauges.
                Assert.Contains("# TYPE overfit_chat_requests_total counter", body);
                Assert.Contains("# TYPE overfit_generated_tokens_total counter", body);
                Assert.Contains("overfit_pool_active_sessions ", body);
                Assert.Contains("overfit_pool_size ", body);

                // Latency histograms (TTFT + response time) in Prometheus histogram format.
                Assert.Contains("# TYPE overfit_chat_ttft_seconds histogram", body);
                Assert.Contains("overfit_chat_ttft_seconds_bucket{le=\"+Inf\"}", body);
                Assert.Contains("# TYPE overfit_chat_response_time_seconds histogram", body);
                Assert.Contains("overfit_chat_response_time_seconds_count", body);

                // The resident-memory value must parse as a positive number (the working set is never zero).
                var line = Array.Find(body.Split('\n'), l => l.StartsWith("process_resident_memory_bytes ", StringComparison.Ordinal));
                Assert.NotNull(line);
                Assert.True(long.Parse(line!.Split(' ')[1]) > 0);
            }
        }

        private static StringContent JsonBody(string json) => new(json, Encoding.UTF8, "application/json");

        /// <summary>
        /// A deterministic stand-in for the real engine: it writes canned responses through the sink so the
        /// HTTP surface can be exercised with no model. Streaming splits "pong" across two SSE chunks so the
        /// framing is genuinely tested.
        /// </summary>
        private sealed class FakeInferenceService : IOpenAiInferenceService
        {
            public bool EmbeddingsAvailable { get; init; } = true;

            public PoolStatus PoolStatus => new(Size: 4, Active: 0, Available: 4, RejectedTotal: 0, PeakActive: 1);

            public ModelsResponse ListModels()
                => new()
                {
                    Data = [new ModelInfo { Id = "fake-model", Created = 0 }]
                };

            public void CompleteChat(ChatCompletionRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken)
            {
                if (request?.Stream == true)
                {
                    sink.BeginEventStream();
                    sink.WriteEvent(Chunk("po"));
                    sink.WriteEvent(Chunk("ng"));
                    sink.WriteEvent("[DONE]");
                    return;
                }

                var response = new ChatCompletionResponse
                {
                    Id = "chatcmpl-fake",
                    Model = "fake-model",
                    Choices =
                    [
                        new ChatChoice
                        {
                            Index = 0,
                            Message = new OpenAiMessage { Role = "assistant", Content = "pong" },
                            FinishReason = "stop",
                        },
                    ],
                    Usage = new OpenAiUsage { PromptTokens = 1, CompletionTokens = 1, TotalTokens = 2 },
                };
                sink.WriteBody(200, "application/json",
                    JsonSerializer.Serialize(response, OpenAiJsonContext.Default.ChatCompletionResponse));
            }

            public void Embed(EmbeddingsRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken)
            {
                if (!EmbeddingsAvailable)
                {
                    sink.WriteBody(501, "application/json",
                        JsonSerializer.Serialize(
                            new OpenAiErrorResponse { Error = new OpenAiError { Message = "no embedding model" } },
                            OpenAiJsonContext.Default.OpenAiErrorResponse));
                    return;
                }

                var response = new EmbeddingsResponse
                {
                    Model = "fake-model",
                    Data = [new EmbeddingData { Index = 0, Embedding = [0.1f, 0.2f, 0.3f] }],
                    Usage = new OpenAiUsage { PromptTokens = 1, TotalTokens = 1 },
                };
                sink.WriteBody(200, "application/json",
                    JsonSerializer.Serialize(response, OpenAiJsonContext.Default.EmbeddingsResponse));
            }

            public void Synthesize(SpeechRequest? request, IOpenAiResponseSink sink, CancellationToken cancellationToken)
                => sink.WriteBinary(200, "audio/wav", [0x52, 0x49, 0x46, 0x46]);

            private static string Chunk(string content)
            {
                var chunk = new ChatCompletionChunk
                {
                    Id = "chatcmpl-fake",
                    Model = "fake-model",
                    Choices = [new ChatChoice { Index = 0, Delta = new OpenAiMessage { Content = content } }],
                };
                return JsonSerializer.Serialize(chunk, OpenAiJsonContext.Default.ChatCompletionChunk);
            }
        }
    }
}
