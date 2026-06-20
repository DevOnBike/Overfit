// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Threading.Tasks;
using DevOnBike.Overfit.Extensions.AI;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using Microsoft.Extensions.AI;
using Xunit.Abstractions;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;

namespace DevOnBike.Overfit.Tests.Adapters
{
    /// <summary>
    /// End-to-end validation of the <c>Microsoft.Extensions.AI</c> adapters against real models: the Overfit
    /// runtime behind the standard <see cref="IChatClient"/> / <see cref="IEmbeddingGenerator{TInput,TEmbedding}"/>
    /// produces coherent chat (greedy "Paris") and 384-dim MiniLM embeddings. [LongFact] (needs C:\qwen3b + C:\minilm).
    /// </summary>
    public sealed class MeaiAdapterEndToEndTests
    {
        private const string Gguf = @"C:\qwen3b\qwen.q4km.gguf";
        private const string MiniLm = @"C:\minilm";

        private readonly ITestOutputHelper _out;
        public MeaiAdapterEndToEndTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public async Task IChatClient_OnRealQwen_AnswersCoherently_AndStreams()
        {
            if (!File.Exists(Gguf))
            {
                _out.WriteLine($"missing {Gguf}");
                return;
            }

            using var overfit = OverfitClient.LoadGguf(Gguf, mmap: true);
            using var chat = overfit.AsChatClient("qwen2.5-3b");

            var response = await chat.GetResponseAsync(
            [
                new ChatMessage(ChatRole.System, "You are concise."),
                new ChatMessage(ChatRole.User, "What is the capital of France? One word."),
            ], new ChatOptions { Temperature = 0f, MaxOutputTokens = 16 });

            _out.WriteLine($"non-stream: '{response.Text}'  finish={response.FinishReason}  out_tokens={response.Usage?.OutputTokenCount}");
            Assert.Contains("Paris", response.Text, StringComparison.OrdinalIgnoreCase);
            Assert.Equal("qwen2.5-3b", response.ModelId);

            // Streaming returns the same answer assembled from deltas.
            var sb = new StringBuilder();
            await foreach (var update in chat.GetStreamingResponseAsync(
                [new ChatMessage(ChatRole.User, "Capital of France? One word.")],
                new ChatOptions { Temperature = 0f, MaxOutputTokens = 16 }))
            {
                sb.Append(update.Text);
            }
            _out.WriteLine($"stream: '{sb}'");
            Assert.Contains("Paris", sb.ToString(), StringComparison.OrdinalIgnoreCase);
        }

        [LongFact]
        public async Task IEmbeddingGenerator_OnRealMiniLm_Produces384DimVectors()
        {
            if (!Directory.Exists(MiniLm))
            {
                _out.WriteLine($"missing {MiniLm}");
                return;
            }

            using var embedder = SentenceEmbedder.ForMiniLm(MiniLm);
            using var gen = embedder.AsEmbeddingGenerator("all-minilm-l6-v2", dimensions: 384);

            var result = await gen.GenerateAsync(["hello world", "a second sentence"]);

            Assert.Equal(2, result.Count);
            Assert.Equal(384, result[0].Vector.Length);
            Assert.Equal(384, result[1].Vector.Length);
            _out.WriteLine($"embeddings: {result.Count} x dim {result[0].Vector.Length}");
        }
    }
}
