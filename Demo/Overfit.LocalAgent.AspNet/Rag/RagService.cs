// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Text;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.LanguageModels.Retrieval;

namespace DevOnBike.Overfit.Demo.LocalAgent.Rag
{
    /// <summary>
    /// In-process RAG over local markdown documents. Loads a BERT-family sentence embedder
    /// (MiniLM by default) lazily, chunks the documents in the data directory, embeds each chunk
    /// into a <see cref="VectorStore"/>, and answers questions by retrieving the top-K chunks and
    /// feeding them to the chat model as grounding context.
    ///
    /// Everything stays inside the .NET process: the embedder, the vector store, and the chat model.
    /// No external vector database, no embedding API, no data egress.
    ///
    /// Single-tenant demo: indexing and querying are serialised behind one lock. The retrieved
    /// context is sent through the shared <see cref="OverfitClient"/> chat session, so a RAG turn
    /// appends to the same conversation history as <c>/chat</c> and is cleared by <c>/reset</c>.
    /// </summary>
    public sealed class RagService : IDisposable
    {
        private readonly IConfiguration _config;
        private readonly ILogger<RagService> _logger;
        private readonly object _gate = new();

        private SentenceEmbedder? _embedder;
        private VectorStore? _store;
        private bool _disposed;

        public RagService(IConfiguration config, ILogger<RagService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public bool IsIndexed => _store is { Count: > 0 };

        public int ChunkCount => _store?.Count ?? 0;

        /// <summary>
        /// Rebuilds the index from every <c>*.md</c> file in the data directory. Idempotent — each
        /// call discards the previous index and builds a fresh one.
        /// </summary>
        public IndexSummary IndexDocuments()
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);

                var embedder = GetEmbedder();
                var dataDir = ResolveDataDir();
                var files = Directory.GetFiles(dataDir, "*.md");
                if (files.Length == 0)
                {
                    throw new InvalidOperationException(
                        $"No *.md documents found in data directory '{dataDir}'. " +
                        "Drop your markdown documents there (or set 'DataPath' in appsettings).");
                }

                var store = new VectorStore(embedder.Dimension, initialCapacity: 64);
                var perFile = new List<FileIndexInfo>(files.Length);

                foreach (var file in files)
                {
                    var name = Path.GetFileName(file);
                    var chunks = Chunk(File.ReadAllText(file));
                    for (var i = 0; i < chunks.Count; i++)
                    {
                        // AddTo embeds the chunk (no prefix — MiniLM convention) and stores the
                        // chunk text as the payload so we can return it as a citation.
                        embedder.AddTo(store, $"{name}#{i}", chunks[i]);
                    }
                    perFile.Add(new FileIndexInfo(name, chunks.Count));
                }

                _store = store;
                _logger.LogInformation("Indexed {Chunks} chunks from {Files} documents.", store.Count, files.Length);
                return new IndexSummary(store.Count, perFile);
            }
        }

        /// <summary>
        /// Answers <paramref name="question"/> by retrieving the top-<paramref name="topK"/> chunks and
        /// sending them as grounding context through the shared chat client. Returns the reply plus the
        /// cited source chunks.
        /// </summary>
        public RagAnswer Query(OverfitClient client, string question, int topK)
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);

                if (_store is null || _store.Count == 0)
                {
                    throw new InvalidOperationException(
                        "No documents are indexed yet. POST /documents/index first.");
                }

                var embedder = GetEmbedder();

                // Time only the retrieval step (embed + cosine scan) — the in-process RAG latency,
                // distinct from the model decode time reported in the generation stats.
                var searchStart = Stopwatch.GetTimestamp();
                var queryVector = embedder.EmbedQuery(question);
                var hits = _store.Search(queryVector, topK);
                var searchSeconds = Stopwatch.GetElapsedTime(searchStart).TotalSeconds;

                var prompt = BuildPrompt(question, hits);
                var reply = client.Send(prompt);
                var stats = client.Chat.LastStats;

                var sources = new RagSource[hits.Length];
                for (var i = 0; i < hits.Length; i++)
                {
                    sources[i] = new RagSource(
                        Index: i + 1,
                        Id: hits[i].Id,
                        Similarity: MathF.Round(hits[i].Score, 4),
                        Snippet: Snippet(hits[i].Payload, 200));
                }

                return new RagAnswer(
                    Reply: reply,
                    Sources: sources,
                    PromptTokens: stats.PromptTokens,
                    GeneratedTokens: stats.GeneratedTokens,
                    TokensPerSecond: Math.Round(stats.TokensPerSecond, 2),
                    SearchSeconds: searchSeconds);
            }
        }

        private static string BuildPrompt(string question, IReadOnlyList<VectorMatch> hits)
        {
            var sb = new StringBuilder();
            sb.AppendLine(
                "Answer the question using ONLY the context below. " +
                "If the answer is not in the context, say you don't have that information. " +
                "Cite the bracketed source numbers you used.");
            sb.AppendLine();
            sb.AppendLine("Context:");
            for (var i = 0; i < hits.Count; i++)
            {
                sb.Append('[').Append(i + 1).Append("] ").AppendLine(hits[i].Payload?.Trim());
            }
            sb.AppendLine();
            sb.Append("Question: ").AppendLine(question);
            return sb.ToString();
        }

        private SentenceEmbedder GetEmbedder()
        {
            if (_embedder is not null)
            {
                return _embedder;
            }

            var dir = ResolveEmbeddingDir();
            _logger.LogInformation("Loading sentence embedder (MiniLM) from {Dir}.", dir);
            _embedder = SentenceEmbedder.ForMiniLm(dir);
            return _embedder;
        }

        private string ResolveEmbeddingDir()
        {
            // 1) appsettings 'EmbeddingModelPath' — directory with config.json + vocab.txt + model.safetensors.
            var fromSettings = _config.GetValue<string>("EmbeddingModelPath");
            if (!string.IsNullOrWhiteSpace(fromSettings) && Directory.Exists(fromSettings))
            {
                return fromSettings;
            }

            // 2) Env var OVERFIT_EMBEDDING_DIR.
            var fromEnv = Environment.GetEnvironmentVariable("OVERFIT_EMBEDDING_DIR");
            if (!string.IsNullOrWhiteSpace(fromEnv) && Directory.Exists(fromEnv))
            {
                return fromEnv;
            }

            throw new InvalidOperationException(
                "Could not locate a sentence-embedding model directory. Either set 'EmbeddingModelPath' " +
                "in appsettings to a MiniLM directory (config.json + vocab.txt + model.safetensors), or set " +
                "the OVERFIT_EMBEDDING_DIR environment variable. RAG (/documents/index, /rag/query) needs this; " +
                "plain /chat does not. See Demo/Overfit.LocalAgent.AspNet/README.md.");
        }

        private string ResolveDataDir()
        {
            var fromSettings = _config.GetValue<string>("DataPath");
            if (!string.IsNullOrWhiteSpace(fromSettings) && Directory.Exists(fromSettings))
            {
                return fromSettings;
            }

            // Default: the Data folder copied next to the built assembly.
            var defaultDir = Path.Combine(AppContext.BaseDirectory, "Data");
            if (Directory.Exists(defaultDir))
            {
                return defaultDir;
            }

            throw new InvalidOperationException(
                $"Data directory not found at '{defaultDir}'. Set 'DataPath' in appsettings to a folder of *.md files.");
        }

        /// <summary>
        /// Splits markdown into chunks at paragraph (blank-line) boundaries, greedily packing
        /// paragraphs up to <paramref name="maxChars"/>. Paragraphs are never split mid-way, so a
        /// chunk stays a coherent passage. Good enough for a demo corpus; production chunking would
        /// be heading-aware with overlap.
        /// </summary>
        private static List<string> Chunk(string text, int maxChars = 600)
        {
            var paragraphs = text
                .Replace("\r\n", "\n")
                .Split("\n\n", StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            var chunks = new List<string>();
            var current = new StringBuilder();

            foreach (var paragraph in paragraphs)
            {
                if (current.Length > 0 && current.Length + paragraph.Length > maxChars)
                {
                    chunks.Add(current.ToString().Trim());
                    current.Clear();
                }

                if (current.Length > 0)
                {
                    current.Append("\n\n");
                }
                current.Append(paragraph);
            }

            if (current.Length > 0)
            {
                chunks.Add(current.ToString().Trim());
            }

            return chunks;
        }

        private static string Snippet(string? text, int maxChars)
        {
            if (string.IsNullOrEmpty(text))
            {
                return string.Empty;
            }

            var oneLine = text.Replace("\r\n", " ").Replace('\n', ' ').Trim();
            return oneLine.Length <= maxChars ? oneLine : oneLine[..maxChars].TrimEnd() + "…";
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _embedder?.Dispose();
        }
    }

    public record IndexSummary(int TotalChunks, IReadOnlyList<FileIndexInfo> Files);

    public record FileIndexInfo(string File, int Chunks);

    public record RagQueryRequest(string Question, int? TopK);

    public record RagAnswer(
        string Reply,
        IReadOnlyList<RagSource> Sources,
        int PromptTokens,
        int GeneratedTokens,
        double TokensPerSecond,
        double SearchSeconds);

    public record RagSource(int Index, string Id, float Similarity, string Snippet);
}
