// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation;

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
    /// context is sent through the shared <see cref="OverfitClient"/> as a stateless one-shot
    /// (<c>Complete</c>), so a RAG answer neither inherits the <c>/chat</c> conversation nor pollutes
    /// it — each query prefills only its own grounding context.
    /// </summary>
    public sealed class RagService : IDisposable
    {
        private readonly IConfiguration _config;
        private readonly ILogger<RagService> _logger;
        private readonly OverfitClient _client;
        private readonly bool _useModelEmbeddings;
        private readonly object _gate = new();

        private SentenceEmbedder? _embedder;
        private VectorStore? _store;
        private float[]? _embeddingMean;   // corpus mean for anisotropy correction (model-embedding mode); null = off
        private bool _disposed;

        public RagService(IConfiguration config, ILogger<RagService> logger, OverfitClient client)
        {
            _config = config;
            _logger = logger;
            _client = client;

            // When true, embed with the loaded GGUF's OWN hidden states (multilingual, no separate embedder)
            // instead of the English-only MiniLM sentence-embedder. Set it for non-English corpora — e.g. the
            // Bielik (Polish) preset — where MiniLM mis-retrieves. The chat model already understands the language.
            _useModelEmbeddings = config.GetValue<bool>("UseModelEmbeddings");
        }

        /// <summary>Embedding dimensionality for the active mode (model hidden size, or the MiniLM dimension).</summary>
        private int EmbeddingDim => _useModelEmbeddings ? _client.EmbeddingDimension : GetEmbedder().Dimension;

        /// <summary>Raw document-side embedding (no retrieval prefix), before mean-centering.</summary>
        private float[] EmbedDocumentRaw(string text)
            => _useModelEmbeddings ? _client.Embed(text) : GetEmbedder().Embed(text);

        /// <summary>Raw query-side embedding — applies the embedder's query prefix in MiniLM mode; model mode is symmetric.</summary>
        private float[] EmbedQueryRaw(string text)
            => _useModelEmbeddings ? _client.Embed(text) : GetEmbedder().EmbedQuery(text);

        /// <summary>Query embedding as the store sees it — mean-centered when centering is active (model mode).</summary>
        private float[] EmbedQuery(string text) => Center(EmbedQueryRaw(text));

        /// <summary>
        /// Subtracts the corpus mean from <paramref name="vector"/> in place (no-op until a mean is computed at
        /// index time). LLM-decoder embeddings are strongly anisotropic — they share a dominant direction, so
        /// every cosine sits in a narrow high band and a "central" chunk wins unrelated queries. Removing the mean
        /// spreads the cosines back out, which restores meaningful absolute similarity (e.g. the false-premise
        /// threshold) without hurting ranking. The store / search re-normalise, so we only subtract here.
        /// </summary>
        private float[] Center(float[] vector)
        {
            var mean = _embeddingMean;
            if (mean is null)
            {
                return vector;
            }
            for (var i = 0; i < vector.Length && i < mean.Length; i++)
            {
                vector[i] -= mean[i];
            }
            return vector;
        }

        private static float[] ComputeMean(List<float[]> vectors, int dim)
        {
            var mean = new float[dim];
            foreach (var v in vectors)
            {
                for (var j = 0; j < dim; j++)
                {
                    mean[j] += v[j];
                }
            }
            var inv = 1f / vectors.Count;
            for (var j = 0; j < dim; j++)
            {
                mean[j] *= inv;
            }
            return mean;
        }

        public bool IsIndexed => _store is { Count: > 0 };

        public int ChunkCount => _store?.Count ?? 0;

        /// <summary>
        /// Indexes every <c>*.md</c> file in the data directory. <b>Index once, restart, query without
        /// re-embedding</b>: a persisted <see cref="PersistentVectorStore"/> (a pure-managed binary file next to
        /// the documents — no SQLite, no native dependency) is reloaded when the source set and per-file content
        /// hashes still match, skipping all embedding. Any change to the corpus triggers a full rebuild (the
        /// model-embedding anisotropy mean is corpus-wide, so it must be recomputed when the corpus changes), and
        /// the fresh index is persisted for next time.
        /// </summary>
        public IndexSummary IndexDocuments()
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);

                var dataDir = ResolveDataDir();
                var files = Directory.GetFiles(dataDir, "*.md");
                if (files.Length == 0)
                {
                    throw new InvalidOperationException(
                        $"No *.md documents found in data directory '{dataDir}'. " +
                        "Drop your markdown documents there (or set 'DataPath' in appsettings).");
                }

                var dim = EmbeddingDim;
                var cachePath = Path.Combine(dataDir, ".overfit-rag-index.bin");

                // Content hash per file (name → hash) — the "reload unchanged, re-embed changed" decision.
                var hashByName = new Dictionary<string, string>(StringComparer.Ordinal);
                foreach (var file in files)
                {
                    hashByName[Path.GetFileName(file)] = ComputeFileHash(file);
                }

                // Fast path: a persisted index whose sources + hashes exactly match → reload, no embedding.
                var summaryFromCache = TryReloadFromCache(cachePath, dim, hashByName, files.Length);
                if (summaryFromCache is not null)
                {
                    return summaryFromCache;
                }

                // Rebuild: embed everything (no cache, or the corpus changed).
                var perFile = new List<FileIndexInfo>(files.Length);
                var fileChunks = new List<FileChunks>(files.Length);
                var allRaw = new List<float[]>();
                foreach (var file in files)
                {
                    var name = Path.GetFileName(file);
                    var chunks = Chunk(File.ReadAllText(file));
                    var fc = new FileChunks(name, new List<VectorChunk>(chunks.Count));
                    for (var i = 0; i < chunks.Count; i++)
                    {
                        var rawVector = EmbedDocumentRaw(chunks[i]);
                        allRaw.Add(rawVector);
                        fc.Raw.Add(new VectorChunk($"{name}#{i}", rawVector, chunks[i]));
                    }
                    fileChunks.Add(fc);
                    perFile.Add(new FileIndexInfo(name, chunks.Count));
                }

                // Mean-center model embeddings (anisotropy fix). MiniLM is already isotropic-ish — leave it alone.
                _embeddingMean = _useModelEmbeddings && allRaw.Count > 0 ? ComputeMean(allRaw, dim) : null;

                var collection = Path.GetFileName(dataDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
                var persistent = new PersistentVectorStore(dim, string.IsNullOrEmpty(collection) ? "rag" : collection);
                foreach (var fc in fileChunks)
                {
                    // Center (no-op when no mean) the raw vectors, then index the file under its content hash.
                    var centered = new List<VectorChunk>(fc.Raw.Count);
                    foreach (var c in fc.Raw)
                    {
                        centered.Add(new VectorChunk(c.Id, Center(c.Vector), c.Payload));
                    }
                    persistent.IndexSource(fc.Name, hashByName[fc.Name], centered);
                }

                _store = persistent.Store;
                TrySaveCache(persistent, cachePath, _embeddingMean);
                _logger.LogInformation("Indexed {Chunks} chunks from {Files} documents (persisted to cache).", persistent.Count, files.Length);
                return new IndexSummary(persistent.Count, perFile);
            }
        }

        private IndexSummary? TryReloadFromCache(string cachePath, int dim, Dictionary<string, string> hashByName, int fileCount)
        {
            if (!File.Exists(cachePath))
            {
                return null;
            }

            PersistentVectorStore loaded;
            try
            {
                loaded = PersistentVectorStore.Load(cachePath);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Ignoring unreadable RAG index cache at {Path}; rebuilding.", cachePath);
                return null;
            }

            // Exact match: same dimension, same source set, every file's hash unchanged.
            if (loaded.Dimension != dim || loaded.SourceCount != fileCount)
            {
                return null;
            }
            foreach (var (name, hash) in hashByName)
            {
                if (loaded.NeedsReindex(name, hash))
                {
                    return null;
                }
            }

            // Model-embedding mode needs the corpus mean restored, or query centering would be wrong.
            var mean = _useModelEmbeddings ? LoadMean(cachePath, dim) : null;
            if (_useModelEmbeddings && mean is null)
            {
                _logger.LogWarning("RAG index cache is missing its mean vector; rebuilding.");
                return null;
            }

            _store = loaded.Store;
            _embeddingMean = mean;
            var perFile = new List<FileIndexInfo>(fileCount);
            foreach (var name in hashByName.Keys)
            {
                perFile.Add(new FileIndexInfo(name, loaded.GetSourceChunkCount(name)));
            }
            _logger.LogInformation("Loaded {Chunks} chunks from cache ({Files} documents, no re-embedding).", loaded.Count, fileCount);
            return new IndexSummary(loaded.Count, perFile);
        }

        private static string ComputeFileHash(string path)
        {
            using var stream = File.OpenRead(path);
            return Convert.ToHexString(SHA256.HashData(stream));
        }

        private void TrySaveCache(PersistentVectorStore store, string cachePath, float[]? mean)
        {
            var meanPath = cachePath + ".mean";
            try
            {
                store.Save(cachePath);
                if (mean is null)
                {
                    if (File.Exists(meanPath))
                    {
                        File.Delete(meanPath);
                    }
                }
                else
                {
                    using var stream = new FileStream(meanPath, FileMode.Create, FileAccess.Write);
                    using var writer = new BinaryWriter(stream);
                    writer.Write(mean.Length);
                    foreach (var v in mean)
                    {
                        writer.Write(v);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not persist the RAG index cache to {Path}; it will rebuild next run.", cachePath);
            }
        }

        private float[]? LoadMean(string cachePath, int dim)
        {
            var path = cachePath + ".mean";
            if (!File.Exists(path))
            {
                return null;
            }
            try
            {
                using var stream = File.OpenRead(path);
                using var reader = new BinaryReader(stream);
                var n = reader.ReadInt32();
                if (n != dim)
                {
                    return null;
                }
                var mean = new float[n];
                for (var i = 0; i < n; i++)
                {
                    mean[i] = reader.ReadSingle();
                }
                return mean;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Ignoring unreadable RAG mean sidecar at {Path}.", path);
                return null;
            }
        }

        private readonly record struct FileChunks(string Name, List<VectorChunk> Raw);

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

                // Time only the retrieval step (embed + cosine scan) — the in-process RAG latency,
                // distinct from the model decode time reported in the generation stats.
                var searchStart = Stopwatch.GetTimestamp();
                var queryVector = EmbedQuery(question);
                var hits = _store.Search(queryVector, topK);
                var searchSeconds = Stopwatch.GetElapsedTime(searchStart).TotalSeconds;

                var prompt = BuildPrompt(question, hits);
                // Stateless: the retrieved context is self-contained grounding — answering must not
                // inherit prior chat turns (it would bias the answer and re-prefill the whole history).
                var reply = client.Complete(prompt);
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

        /// <summary>
        /// Embeds each input string with the loaded sentence embedder (lazy-loaded on first use, like the
        /// RAG index). Backs the OpenAI-compatible <c>/v1/embeddings</c> endpoint. Throws
        /// <see cref="InvalidOperationException"/> with an actionable message if no embedding model is found.
        /// </summary>
        public IReadOnlyList<float[]> EmbedAll(IReadOnlyList<string> texts)
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);

                var result = new float[texts.Count][];
                for (var i = 0; i < texts.Count; i++)
                {
                    result[i] = EmbedQuery(texts[i] ?? string.Empty);
                }
                return result;
            }
        }

        /// <summary>
        /// Runs the RAG Stability Harness over the indexed corpus: expected-source recall, paraphrase stability,
        /// false-premise traps, plus a corpus lint (near-duplicates / short docs / orphans). Pure retrieval-side
        /// evaluation — deterministic, no LLM call. Makes "RAG is testable" a live endpoint.
        /// </summary>
        public RagEvalResult Evaluate(RagEvalRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);

            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);

                if (_store is null || _store.Count == 0)
                {
                    throw new InvalidOperationException("No documents are indexed yet. POST /documents/index first.");
                }

                var evaluator = new RagEvaluator(_store, EmbedQuery);

                RetrievalReport? retrieval = null;
                if (request.Retrieval.Count > 0)
                {
                    var cases = new List<RetrievalCase>(request.Retrieval.Count);
                    foreach (var rc in request.Retrieval)
                    {
                        cases.Add(new RetrievalCase(rc.Query, ExpandExpectedSource(rc.ExpectedSource)));
                    }
                    retrieval = evaluator.EvaluateRetrieval(cases, request.TopK);
                }

                ParaphraseStabilityReport? paraphrase = null;
                if (request.Paraphrase.Count > 0)
                {
                    var groups = new List<ParaphraseGroup>(request.Paraphrase.Count);
                    foreach (var g in request.Paraphrase)
                    {
                        groups.Add(new ParaphraseGroup(g.Name, g.Variants.ToArray()));
                    }
                    paraphrase = evaluator.EvaluateParaphraseStability(groups, request.TopK, request.MinJaccard);
                }

                FalsePremiseReport? falsePremise = null;
                if (request.FalsePremise.Count > 0)
                {
                    var traps = new List<FalsePremiseCase>(request.FalsePremise.Count);
                    foreach (var q in request.FalsePremise)
                    {
                        traps.Add(new FalsePremiseCase(q));
                    }
                    falsePremise = evaluator.EvaluateFalsePremise(traps, request.GroundedThreshold);
                }

                var linter = new CorpusLinter(_store);
                var nearDuplicates = linter.FindNearDuplicates(request.DuplicateThreshold);
                var shortDocuments = linter.FindShortDocuments(request.MinDocChars);

                // Orphans relative to every query the request mentions (retrieval + paraphrase variants).
                var queryVectors = new List<float[]>();
                foreach (var rc in request.Retrieval)
                {
                    queryVectors.Add(EmbedQuery(rc.Query));
                }
                foreach (var g in request.Paraphrase)
                {
                    foreach (var v in g.Variants)
                    {
                        queryVectors.Add(EmbedQuery(v));
                    }
                }
                IReadOnlyList<string> orphans = queryVectors.Count > 0 ? linter.FindOrphans(queryVectors, request.TopK) : [];

                return new RagEvalResult(retrieval, paraphrase, falsePremise, nearDuplicates, shortDocuments, orphans);
            }
        }

        /// <summary>Expands an expected-source substring (e.g. a file name) to the matching chunk ids in the store.</summary>
        private string[] ExpandExpectedSource(string expectedSource)
        {
            var ids = new List<string>();
            for (var i = 0; i < _store!.Count; i++)
            {
                var id = _store.GetId(i);
                if (id.Contains(expectedSource, StringComparison.OrdinalIgnoreCase))
                {
                    ids.Add(id);
                }
            }
            return ids.ToArray();
        }

        private static string BuildPrompt(string question, IReadOnlyList<VectorMatch> hits)
        {
            var sb = new StringBuilder();
            sb.AppendLine("You are a support assistant. Answer the question using ONLY the numbered context below. Rules:");
            sb.AppendLine("- Reply in the SAME LANGUAGE as the question.");
            sb.AppendLine("- Use only facts stated in the context. If the answer is not there, say you don't have that information.");
            sb.AppendLine("- For a yes/no question about a number, date, period, or limit: FIRST state the relevant limit from the context and whether the value in the question is within or beyond that limit; THEN give a clear yes/no verdict (in the question's language).");
            sb.AppendLine("- Keep the answer to at most two sentences, and cite the [n] source(s) you used.");
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
                "plain /chat does not. See Demo/LocalAgentAspNetDemo/README.md.");
        }

        private string ResolveDataDir()
        {
            var fromSettings = _config.GetValue<string>("DataPath");
            if (!string.IsNullOrWhiteSpace(fromSettings))
            {
                // Absolute path, or a folder name relative to the output dir (e.g. the Bielik preset's
                // "Data-pl", copied next to the assembly).
                if (Directory.Exists(fromSettings))
                {
                    return fromSettings;
                }
                var relative = Path.Combine(AppContext.BaseDirectory, fromSettings);
                if (Directory.Exists(relative))
                {
                    return relative;
                }
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
        /// Splits markdown into chunks at paragraph (blank-line) boundaries, greedily packing paragraphs up to
        /// <paramref name="maxChars"/>. Paragraphs are never split mid-way, so a chunk stays a coherent passage.
        /// (A heading-aware + overlap variant was tried and MEASURED to regress recall 1.0 -> 0.0 on the LLM-
        /// embedding + mean-centering path — the smaller chunk set shifted the centering basis and the anisotropy
        /// hack degenerated. Heading-aware chunking is standard practice and worth revisiting with a real
        /// multilingual sentence-embedder, where it isn't fighting a global-mean correction.)
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

}
