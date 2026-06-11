// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Mcp
{
    /// <summary>
    /// Factories for the built-in Overfit MCP tools — each wraps an existing runtime facade
    /// (<see cref="OverfitClient"/>, <see cref="McpRagIndex"/>, <see cref="WhisperTranscriber"/>)
    /// in an <see cref="McpTool"/> with a hand-written JSON Schema. All tools are local and
    /// zero-egress: the model, the documents and the audio never leave the machine.
    /// </summary>
    public static class OverfitMcpTools
    {
        /// <summary>
        /// <c>ask</c> — one-shot prompt → the loaded local chat model, via the stateless
        /// <see cref="OverfitClient.Complete"/> path (tool calls must not accumulate history).
        /// </summary>
        public static McpTool CreateAsk(OverfitClient client)
        {
            ArgumentNullException.ThrowIfNull(client);

            return new McpTool(
                "ask",
                "Ask the locally-loaded LLM (runs fully on this machine, no data leaves it). Returns the model's text answer.",
                """
                {"type":"object","properties":{"prompt":{"type":"string","description":"The question or instruction for the local model."}},"required":["prompt"]}
                """.Trim(),
                arguments =>
                {
                    if (!TryGetRequiredString(arguments, "prompt", out var prompt, out var error))
                    {
                        return error;
                    }

                    return McpToolResult.Success(client.Complete(prompt).Trim());
                });
        }

        /// <summary>
        /// <c>rag_query</c> — grounded Q&amp;A with citations over a locally-indexed document folder
        /// (see <see cref="McpRagIndex.Build"/>).
        /// </summary>
        public static McpTool CreateRagQuery(McpRagIndex index)
        {
            ArgumentNullException.ThrowIfNull(index);

            return new McpTool(
                "rag_query",
                "Answer a question from the private local document index (RAG with source citations). Use this for anything the indexed documents might cover — the documents never leave this machine.",
                """
                {"type":"object","properties":{"question":{"type":"string","description":"The question to answer from the indexed documents."},"top_k":{"type":"integer","description":"How many document chunks to retrieve (default 4).","minimum":1,"maximum":16}},"required":["question"]}
                """.Trim(),
                arguments =>
                {
                    if (!TryGetRequiredString(arguments, "question", out var question, out var error))
                    {
                        return error;
                    }

                    var topK = 4;

                    if (arguments is { ValueKind: JsonValueKind.Object } args &&
                        args.TryGetProperty("top_k", out var topKElement) &&
                        topKElement.ValueKind == JsonValueKind.Number)
                    {
                        topK = Math.Clamp(topKElement.GetInt32(), 1, 16);
                    }

                    return McpToolResult.Success(index.Query(question, topK));
                });
        }

        /// <summary>
        /// <c>transcribe</c> — WAV/MP3 file → text via Whisper, pure C# on the CPU. The transcriber
        /// is created lazily on first use (loading the ggml model costs seconds — don't pay it when
        /// the host never calls the tool).
        /// </summary>
        public static McpTool CreateTranscribe(Func<WhisperTranscriber> transcriberFactory)
        {
            ArgumentNullException.ThrowIfNull(transcriberFactory);

            WhisperTranscriber? transcriber = null;

            return new McpTool(
                "transcribe",
                "Transcribe a local audio file (WAV or MP3) to text using Whisper, fully on this machine. Returns the transcript.",
                """
                {"type":"object","properties":{"path":{"type":"string","description":"Absolute path to a .wav or .mp3 file on this machine."},"language":{"type":"string","description":"ISO language code of the speech, e.g. 'en' or 'pl' (default 'en')."}},"required":["path"]}
                """.Trim(),
                arguments =>
                {
                    if (!TryGetRequiredString(arguments, "path", out var path, out var error))
                    {
                        return error;
                    }

                    if (!File.Exists(path))
                    {
                        return McpToolResult.Error($"Audio file not found: {path}");
                    }

                    var language = "en";

                    if (arguments is { ValueKind: JsonValueKind.Object } args &&
                        args.TryGetProperty("language", out var languageElement) &&
                        languageElement.ValueKind == JsonValueKind.String)
                    {
                        language = languageElement.GetString()!;
                    }

                    transcriber ??= transcriberFactory();

                    return McpToolResult.Success(transcriber.TranscribeFile(path, language).Trim());
                });
        }

        private static bool TryGetRequiredString(JsonElement? arguments, string property, out string value, out McpToolResult error)
        {
            if (arguments is { ValueKind: JsonValueKind.Object } args &&
                args.TryGetProperty(property, out var element) &&
                element.ValueKind == JsonValueKind.String &&
                element.GetString() is { Length: > 0 } text)
            {
                value = text;
                error = null!;
                return true;
            }

            value = null!;
            error = McpToolResult.Error($"Missing required string argument '{property}'.");
            return false;
        }
    }
}
