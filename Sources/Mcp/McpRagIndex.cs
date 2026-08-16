// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Retrieval;

namespace DevOnBike.Overfit.Mcp
{
    /// <summary>
    /// A self-contained RAG index over a local document folder for the <c>rag_query</c> MCP tool:
    /// chunks <c>.txt</c>/<c>.md</c> files on paragraph boundaries, embeds every chunk with the chat
    /// model's OWN embeddings (<see cref="OverfitClient.Embed"/> — multilingual, no second model
    /// needed), then answers questions grounded in the top-K chunks with per-chunk source citations.
    /// Everything stays on the machine.
    ///
    /// <para>Retrieval is <b>hybrid</b> (<see cref="HybridRetriever"/>): semantic search over the embeddings
    /// fused with BM25 over the chunk text. Measured on this repository's own <c>docs/</c> folder (481 chunks,
    /// MiniLM, recall@5) that lifted recall from <b>0.61 to 0.94</b> and MRR from <b>0.500 to 0.775</b> —
    /// see <c>HybridVsDenseOnDocsCorpusTests</c>. Local document sets are full of literal tokens (file names,
    /// env vars, error codes, API names) that embeddings blur together and BM25 matches exactly.</para>
    /// </summary>
    public sealed class McpRagIndex
    {
        private const int TargetChunkChars = 1200;

        private readonly OverfitClient _client;
        private readonly HybridRetriever _retriever;

        public int ChunkCount => _retriever.Count;

        private McpRagIndex(OverfitClient client, HybridRetriever retriever)
        {
            _client = client;
            _retriever = retriever;
        }

        /// <summary>
        /// Indexes every <c>*.txt</c> / <c>*.md</c> under <paramref name="directory"/> (recursive).
        /// Embedding happens here, once — queries only embed the question.
        /// </summary>
        // OVERFIT040 on Build: `File.ReadAllText` has a `ReadAllTextAsync` sibling and this is the one place
        // that reading synchronously is right. BOUND BY WHEN IT RUNS: Build is a ONE-SHOT STARTUP STEP —
        // Cli/Commands.cs calls it once, on the process main thread, before `McpServer.Run` begins serving,
        // and there is no request in flight and no pool thread behind it to give back. The loop is dominated
        // by `client.Embed` per chunk, which is synchronous CPU work on that same thread regardless, so
        // awaiting the file read would change where the method suspends and not how long it holds anything.
#pragma warning disable OVERFIT040
        public static McpRagIndex Build(OverfitClient client, string directory, TextWriter? log = null)
#pragma warning restore OVERFIT040
        {
            ArgumentNullException.ThrowIfNull(client);

            if (!Directory.Exists(directory))
            {
                throw new DirectoryNotFoundException($"RAG document directory not found: {directory}");
            }

            var retriever = new HybridRetriever(client.EmbeddingDimension);
            var files = new List<string>();
            files.AddRange(Directory.GetFiles(directory, "*.txt", SearchOption.AllDirectories));
            files.AddRange(Directory.GetFiles(directory, "*.md", SearchOption.AllDirectories));
            files.Sort(StringComparer.OrdinalIgnoreCase);

            foreach (var file in files)
            {
                var name = Path.GetFileName(file);
                var chunks = ChunkParagraphs(File.ReadAllText(file));

                for (var i = 0; i < chunks.Count; i++)
                {
                    var vector = client.Embed(chunks[i]);
                    retriever.Add($"{name}#{i + 1}", vector, chunks[i]);
                }

                log?.WriteLine($"[overfit-mcp] indexed {name}: {chunks.Count} chunk(s)");
            }

            if (retriever.Count == 0)
            {
                throw new OverfitRuntimeException($"No indexable .txt/.md content found under: {directory}");
            }

            return new McpRagIndex(client, retriever);
        }

        /// <summary>
        /// Embeds the question, retrieves the top-<paramref name="topK"/> chunks, and generates a
        /// grounded answer via the stateless <see cref="OverfitClient.Complete"/> path (no
        /// conversation accumulation between tool calls). Returns the answer followed by a
        /// "Sources:" list naming the cited chunks.
        /// </summary>
        public string Query(string question, int topK = 4)
        {
            ArgumentException.ThrowIfNullOrEmpty(question);

            var queryVector = _client.Embed(question);
            var matches = _retriever.Search(queryVector, question, Math.Min(topK, _retriever.Count));

            var prompt = new StringBuilder(4096);
            prompt.AppendLine("Answer the question using ONLY the context below. Cite the context entries you used as [1], [2], … . If the context does not contain the answer, say so plainly.");
            prompt.AppendLine();
            prompt.AppendLine("Context:");

            for (var i = 0; i < matches.Length; i++)
            {
                prompt.Append('[').Append(i + 1).Append("] (").Append(matches[i].Id).AppendLine(")");
                prompt.AppendLine(matches[i].Payload);
                prompt.AppendLine();
            }

            prompt.Append("Question: ").Append(question);

            var answer = _client.Complete(prompt.ToString());

            var result = new StringBuilder(answer.Length + 256);
            result.AppendLine(answer.Trim());
            result.AppendLine();
            result.AppendLine("Sources:");

            for (var i = 0; i < matches.Length; i++)
            {
                result.Append('[').Append(i + 1).Append("] ").Append(matches[i].Id)
                      .Append(" (score ").Append(matches[i].Score.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)).AppendLine(")");
            }

            return result.ToString();
        }

        /// <summary>Greedy paragraph packing: split on blank lines, pack consecutive paragraphs up to
        /// ~<see cref="TargetChunkChars"/> chars per chunk (a paragraph longer than the target becomes
        /// its own chunk — never split mid-paragraph).</summary>
        internal static List<string> ChunkParagraphs(string text)
        {
            var chunks = new List<string>();
            var current = new StringBuilder(TargetChunkChars + 256);
            var paragraphs = text.Replace("\r\n", "\n").Split("\n\n", StringSplitOptions.RemoveEmptyEntries);

            foreach (var raw in paragraphs)
            {
                var paragraph = raw.Trim();

                if (paragraph.Length == 0)
                {
                    continue;
                }

                if (current.Length > 0 && current.Length + paragraph.Length > TargetChunkChars)
                {
                    chunks.Add(current.ToString());
                    current.Clear();
                }

                if (current.Length > 0)
                {
                    current.Append('\n').Append('\n');
                }

                current.Append(paragraph);
            }

            if (current.Length > 0)
            {
                chunks.Add(current.ToString());
            }

            return chunks;
        }
    }
}
