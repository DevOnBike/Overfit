// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Retrieval;

namespace DevOnBike.Overfit.Mcp
{
    /// <summary>
    /// A self-contained RAG index over a local document folder for the <c>rag_query</c> MCP tool:
    /// chunks <c>.txt</c>/<c>.md</c> files on paragraph boundaries, embeds every chunk with the chat
    /// model's OWN embeddings (<see cref="OverfitClient.Embed"/> — multilingual, no second model
    /// needed) into an in-process <see cref="VectorStore"/>, then answers questions grounded in the
    /// top-K chunks with per-chunk source citations. Everything stays on the machine.
    /// </summary>
    public sealed class McpRagIndex
    {
        private const int TargetChunkChars = 1200;

        private readonly OverfitClient _client;
        private readonly VectorStore _store;

        public int ChunkCount => _store.Count;

        private McpRagIndex(OverfitClient client, VectorStore store)
        {
            _client = client;
            _store = store;
        }

        /// <summary>
        /// Indexes every <c>*.txt</c> / <c>*.md</c> under <paramref name="directory"/> (recursive).
        /// Embedding happens here, once — queries only embed the question.
        /// </summary>
        public static McpRagIndex Build(OverfitClient client, string directory, TextWriter? log = null)
        {
            ArgumentNullException.ThrowIfNull(client);

            if (!Directory.Exists(directory))
            {
                throw new DirectoryNotFoundException($"RAG document directory not found: {directory}");
            }

            var store = new VectorStore(client.EmbeddingDimension);
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
                    store.Add($"{name}#{i + 1}", vector, chunks[i]);
                }

                log?.WriteLine($"[overfit-mcp] indexed {name}: {chunks.Count} chunk(s)");
            }

            if (store.Count == 0)
            {
                throw new InvalidDataException($"No indexable .txt/.md content found under: {directory}");
            }

            return new McpRagIndex(client, store);
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
            var matches = _store.Search(queryVector, Math.Min(topK, _store.Count));

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
