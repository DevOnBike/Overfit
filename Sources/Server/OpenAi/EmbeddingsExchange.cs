// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Embeddings;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// The one implementation of <c>POST /v1/embeddings</c>, shared by every host: validates the input,
    /// embeds each string in-process (nothing leaves the box) and writes the response through an
    /// <see cref="IOpenAiResponseSink"/>. The caller owns concurrency — a <see cref="SentenceEmbedder"/> has a
    /// single scratch arena, so the host serializes calls to it.
    /// </summary>
    public static class EmbeddingsExchange
    {
        public static void Handle(
            EmbeddingsRequest? req, SentenceEmbedder embedder, string modelName, IOpenAiResponseSink sink)
        {
            ArgumentNullException.ThrowIfNull(embedder);
            ArgumentNullException.ThrowIfNull(sink);

            var inputs = req == null ? [] : OpenAiChatMapping.ParseInputs(req.Input);
            if (inputs.Count == 0)
            {
                WriteError(sink, 400, "'input' is required (a string or an array of strings).");
                return;
            }

            var data = new List<EmbeddingData>(inputs.Count);
            var approxTokens = 0;
            for (var i = 0; i < inputs.Count; i++)
            {
                data.Add(new EmbeddingData { Index = i, Embedding = embedder.Embed(inputs[i]) });
                approxTokens += Math.Max(1, inputs[i].Length / 4);   // rough proxy; we don't bill tokens
            }

            var response = new EmbeddingsResponse
            {
                Model = modelName,
                Data = data,
                Usage = new OpenAiUsage { PromptTokens = approxTokens, TotalTokens = approxTokens },
            };

            sink.WriteBody(200, "application/json",
                JsonSerializer.Serialize(response, OpenAiJsonContext.Default.EmbeddingsResponse));
        }

        private static void WriteError(IOpenAiResponseSink sink, int status, string message)
        {
            var body = new OpenAiErrorResponse { Error = new OpenAiError { Message = message } };
            sink.WriteBody(status, "application/json",
                JsonSerializer.Serialize(body, OpenAiJsonContext.Default.OpenAiErrorResponse));
        }
    }
}
