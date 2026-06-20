// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.LanguageModels.Tools;

namespace DevOnBike.Overfit.Demo.Agent
{
    /// <summary>
    /// One end-to-end walkthrough of the in-process agentic .NET stack on a single GGUF model,
    /// with no Python, no native binary, no server: memory-mapped load → RAG over embeddings →
    /// guaranteed-valid tool calling dispatched to C# delegates → guaranteed-valid JSON output.
    ///
    /// Point the demo at a Qwen2.5 GGUF directory (containing the .gguf + tokenizer.json):
    ///   set OVERFIT_MODEL_DIR=C:\qwen3b   (defaults to C:\qwen3b)
    ///   dotnet run -c Release --project Demo/AgentDemo
    /// </summary>
    internal static class Program
    {
        private static int Main()
        {
            var dir = Environment.GetEnvironmentVariable("OVERFIT_MODEL_DIR") ?? @"C:\qwen3b";
            var ggufPath = ResolveGguf(dir);
            if (ggufPath is null)
            {
                Console.Error.WriteLine(
                    $"No .gguf found in '{dir}'. Set OVERFIT_MODEL_DIR to a Qwen2.5 GGUF directory " +
                    "(containing the .gguf and tokenizer.json).");
                return 1;
            }

            Banner("Overfit — in-process agentic .NET");
            Console.WriteLine($"Model:  {ggufPath}");
            Console.WriteLine("Stack:  pure C# / .NET 10 — no Python, no native binary, no server.\n");

            // ── Load (memory-mapped GGUF) ──────────────────────────────────────
            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ggufPath);   // mmap default
            sw.Stop();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            var liveHeapMb = GC.GetTotalMemory(forceFullCollection: true) / (1024.0 * 1024);

            Console.WriteLine($"Loaded in {sw.ElapsedMilliseconds} ms.");
            Console.WriteLine(
                $"Live managed heap with the model loaded: {liveHeapMb:F0} MB " +
                "(weights are file-mapped — shared/clean pages, not committed private RAM).\n");

            var inner = QwenTokenizer.Load(dir);
            var tok = new QwenChatTokenizer(inner);

            ChatTemplate template;
            using (var reader = new GgufReader(ggufPath))
            {
                template = ChatTemplate.Detect(reader.GetMeta("tokenizer.chat_template", string.Empty));
            }

            using var session = engine.CreateSession(1024);

            RunRag(session, inner);
            RunToolCalling(session, tok, template);
            RunJsonMode(session, tok, template);

            Banner("Done");
            Console.WriteLine("RAG + tool calling + structured output — all in one .NET process, on the CPU.");
            return 0;
        }

        // ── 1. Retrieval over embeddings (in-process RAG) ──────────────────────
        private static void RunRag(CachedLlamaSession session, QwenTokenizer inner)
        {
            Banner("1. In-process RAG (embeddings + cosine search)");

            string[] docs =
            [
                "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
                "Photosynthesis converts light energy into chemical energy in plants.",
                "The TCP three-way handshake establishes a reliable network connection.",
                "Mount Everest is the highest mountain above sea level, in the Himalayas.",
            ];
            const string query = "Which landmark is in France?";

            // Index the documents in an in-process vector store, then retrieve by cosine.
            var store = new VectorStore(session.EmbeddingDimension);
            for (var i = 0; i < docs.Length; i++)
            {
                store.Add($"doc{i}", session.Embed(inner.Encode(docs[i])), docs[i]);
            }

            Console.WriteLine($"Query: \"{query}\"\n");
            var hits = store.Search(session.Embed(inner.Encode(query)), topK: docs.Length);
            foreach (var hit in hits)
            {
                Console.WriteLine($"  cos={hit.Score:F3}  {hit.Payload}");
            }
            Console.WriteLine($"\nTop match → {hits[0].Payload}\n");
        }

        // ── 2. Tool calling (constrained → parse → dispatch to C#) ─────────────
        private static void RunToolCalling(CachedLlamaSession session, QwenChatTokenizer tok, ChatTemplate template)
        {
            Banner("2. Tool calling (guaranteed-valid call → dispatch to C#)");

            var tools = new[]
            {
                new ToolDefinition("get_weather", "Get the current weather for a city."),
                new ToolDefinition("get_time", "Get the current time in a timezone."),
            };

            // The C# functions the model is allowed to call.
            var dispatch = new Dictionary<string, Func<JsonElement, string>>
            {
                ["get_weather"] = args => $"18°C, light rain  (args: {args.GetRawText()})",
                ["get_time"] = args => $"14:05 local  (args: {args.GetRawText()})",
            };

            var chat = new ChatSession(session, tok, template, ["<|im_end|>", "\n<|im_start|>"]);
            chat.AddSystem(
                "You can call tools. Tools: get_weather(city), get_time(timezone). " +
                "Respond with a single tool call.");

            var options = new GenerationOptions(
                maxNewTokens: 128, maxContextLength: 1024, sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: true, endOfTextTokenId: QwenTokenizer.ImEnd);

            const string question = "What is the weather in Paris?";
            var reply = chat.Send(question, in options, onText: null, constraint: new ToolCallConstraint(tools, tok));

            Console.WriteLine($"User:        {question}");
            Console.WriteLine($"Model emits: {reply.Trim()}");

            if (ToolCall.TryParse(reply, out var call) && dispatch.TryGetValue(call.Name, out var fn))
            {
                using var argsDoc = JsonDocument.Parse(call.Arguments);
                Console.WriteLine($"Dispatch:    {call.Name}(...) → {fn(argsDoc.RootElement)}\n");
            }
        }

        // ── 3. Structured output (guaranteed well-formed JSON) ─────────────────
        private static void RunJsonMode(CachedLlamaSession session, QwenChatTokenizer tok, ChatTemplate template)
        {
            Banner("3. Structured output (guaranteed well-formed JSON)");

            var chat = new ChatSession(session, tok, template, ["<|im_end|>", "\n<|im_start|>"]);
            chat.AddSystem("You output only JSON.");

            var options = new GenerationOptions(
                maxNewTokens: 128, maxContextLength: 1024, sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: true, endOfTextTokenId: QwenTokenizer.ImEnd);

            const string question = "Give a person object with fields name and age for Alice, who is 30.";
            var reply = chat.Send(question, in options, onText: null, constraint: new JsonGrammarConstraint(tok));

            Console.WriteLine($"User:        {question}");
            Console.WriteLine($"Model emits: {reply.Trim()}");
            using var doc = JsonDocument.Parse(reply);   // always parses — enforced at decode
            Console.WriteLine($"Parsed OK → root is {doc.RootElement.ValueKind}.\n");
        }

        private static string? ResolveGguf(string dir)
        {
            if (!Directory.Exists(dir))
            {
                return null;
            }
            var preferred = Path.Combine(dir, "qwen.q4km.gguf");
            if (File.Exists(preferred))
            {
                return preferred;
            }
            var any = Directory.GetFiles(dir, "*.gguf");
            return any.Length > 0 ? any[0] : null;
        }

        private static void Banner(string title)
        {
            Console.WriteLine(new string('═', 66));
            Console.WriteLine($"  {title}");
            Console.WriteLine(new string('═', 66));
        }
    }
}
