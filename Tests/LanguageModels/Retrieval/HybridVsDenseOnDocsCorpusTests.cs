// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// The decisive hybrid-vs-dense measurement, on a <b>real</b> corpus: this repository's own
    /// <c>docs/</c> folder (~30 markdown files, ~500 KB), chunked exactly the way <c>McpRagIndex</c> chunks a
    /// user's document folder in production.
    ///
    /// <para><b>Why this test exists.</b> <see cref="HybridVsDenseRecallTests"/> measured hybrid as a net
    /// regression on an 18-chunk synthetic corpus, but that result was not trustworthy: BM25's IDF cannot
    /// discriminate on a corpus that small, so function words like "how" earned a high IDF and dragged the
    /// wrong document to rank 1. Several hundred chunks is the smallest honest test of the technique — the
    /// scaffolding has to be bigger than the effect being measured.</para>
    ///
    /// <para><b>Ground truth is file-level and deliberately strict:</b> a case is a hit only if a chunk of the
    /// document that actually covers the topic appears in the top-K. The corpus also contains three
    /// cross-cutting summary documents (<c>claim-to-test</c>, <c>release-infographic</c>, <c>use-cases-2026</c>)
    /// that mention nearly every feature in passing and therefore act as genuine distractors. That makes the
    /// absolute numbers pessimistic — but it biases <i>both arms identically</i>, which is all an A/B
    /// requires.</para>
    ///
    /// <para><b>MEASURED RESULT (2026-07-21, MiniLM, 45 files / 481 chunks, recall@5). This is what put
    /// <c>McpRagIndex</c> onto hybrid retrieval:</b></para>
    /// <code>
    /// group         dense R@K  hybrid R@K  dense MRR  hybrid MRR
    /// semantic           0.67        0.83      0.417       0.542
    /// identifier         0.33        0.83      0.333       0.667
    /// mixed              0.83        1.00      0.750       1.000
    /// OVERALL            0.61        0.89      0.500       0.736
    /// </code>
    ///
    /// <para>Hybrid won every group — including the semantic one, which the small-corpus run had shown it
    /// damaging. Two cases still fail and both are informative rather than mysterious:</para>
    /// <list type="number">
    ///   <item><c>OVERFIT_DECODE_WORKERS</c> is missed by BOTH arms. <see cref="Bm25Index.Tokenize"/> split it
    ///     into <c>overfit</c> + <c>decode</c> + <c>workers</c> — three of the commonest words in this corpus.
    ///     <b>The joined-identifier fix was then built and measured, and it is a mechanism win with zero
    ///     end-to-end effect.</b> The lexical arm moved exactly as designed — <c>docker.md#2</c>, the document
    ///     that defines the variable, went from lexical rank 2 (score 8.21) to rank 1 (13.15) — yet every
    ///     number in the table above stayed bit-identical, because fusion, not tokenisation, is now the binding
    ///     constraint: a hit present in one arm at rank 1 scores 1/61 ≈ 0.016, while a document present in both
    ///     arms at ranks 5 and 3 scores 1/65 + 1/63 ≈ 0.031. Agreement outweighing a single confident arm is
    ///     RRF working as designed, and it is wrong for a unique identifier. <b>The next hypothesis to measure
    ///     is therefore fusion weighting (or an exact-identifier short-circuit), NOT more tokenisation
    ///     work.</b></item>
    ///   <item>"teach a model my own private data without a graphics card" regressed from dense rank 1 to
    ///     missed, because the lexical arm confidently returned unrelated chunks on common words. This is the
    ///     same function-word weakness the small-corpus test isolated; it is now rare rather than
    ///     systematic.</item>
    /// </list>
    /// </summary>
    public sealed class HybridVsDenseOnDocsCorpusTests
    {
        private readonly ITestOutputHelper _out;

        public HybridVsDenseOnDocsCorpusTests(ITestOutputHelper output) => _out = output;

        private sealed record Case(string Group, string Query, params string[] ExpectedFiles);

        // Literal-token lookups. Every token below was verified to occur in EXACTLY ONE file of the corpus,
        // so the expected answer is not a matter of opinion. This is the group BM25 should win.
        private static readonly Case[] Identifier =
        [
            new("identifier", "vpmaddubsw", "llamacpp-cpu-analysis.md"),
            new("identifier", "block_q4_Kx8", "llamacpp-cpu-analysis.md"),
            new("identifier", "OVERFIT_DECODE_WORKERS", "docker.md"),
            new("identifier", "PESEL", "redaction-gateway-spec.md"),
            new("identifier", "AdamEpsilon", "qlora-finetuning.md"),
            new("identifier", "Luhn", "redaction-gateway-spec.md"),
        ];

        // Paraphrases that deliberately avoid the target document's own vocabulary, so only the semantic arm
        // can reach them. This is the group hybrid can damage.
        private static readonly Case[] Semantic =
        [
            new("semantic", "how can I teach a model my own private data without a graphics card?", "qlora-finetuning.md"),
            new("semantic", "can I replace Ollama in an app I already wrote?", "microsoft-extensions-ai.md"),
            new("semantic", "how do I make a synthetic voice that sounds like one particular person?", "voice-cloning.md"),
            new("semantic", "which language models is this able to open?", "supported-models.md"),
            new("semantic", "how do I stop my document search from silently getting worse?", "rag-testing.md"),
            new("semantic", "how do I run this inside a container?", "docker.md"),
        ];

        // Everyday questions with partial lexical overlap — neither arm is obviously right.
        private static readonly Case[] Mixed =
        [
            new("mixed", "how do I plug local AI into Claude Code?", "mcp.md"),
            new("mixed", "why is decode slower than llama.cpp?", "overfit_perf_decode_analysis.md", "llamacpp-cpu-analysis.md"),
            new("mixed", "is the Polish model any good?", "bielik.md"),
            new("mixed", "how do I decode an MP3 in pure C#?", "mp3-decoding.md"),
            new("mixed", "what does the serving benchmark measure?", "serving-benchmark.md"),
            new("mixed", "how do I evaluate prompts locally for free?", "skill-eval.md"),
        ];

        [LocalOnlyFact]
        public void Hybrid_VsDense_OnRealDocsCorpus()
        {
            if (!File.Exists(TestModelPaths.MiniLm.SafetensorsPath))
            {
                _out.WriteLine($"missing MiniLM fixture at {TestModelPaths.MiniLm.Dir}");
                return;
            }

            var docsDirectory = FindDocsDirectory();
            if (docsDirectory is null)
            {
                _out.WriteLine("could not locate the repository docs/ folder from the test output directory");
                return;
            }

            using var embedder = SentenceEmbedder.ForMiniLm(TestModelPaths.MiniLm.Dir);

            var hybrid = new HybridRetriever(embedder.Dimension, 512);
            var chunkIdsByFile = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);

            var files = Directory.GetFiles(docsDirectory, "*.md", SearchOption.AllDirectories);
            Array.Sort(files, StringComparer.OrdinalIgnoreCase);

            foreach (var file in files)
            {
                var name = Path.GetFileName(file);
                var chunks = ChunkParagraphs(File.ReadAllText(file));
                var ids = new List<string>();

                for (var i = 0; i < chunks.Count; i++)
                {
                    var id = $"{name}#{i + 1}";
                    hybrid.Add(id, embedder.Embed(chunks[i]), chunks[i]);
                    ids.Add(id);
                }

                chunkIdsByFile[name] = ids;
            }

            _out.WriteLine($"=== Hybrid vs dense on docs/ — {files.Length} files, {hybrid.Count} chunks, MiniLM ===");

            var denseEvaluator = new RagEvaluator(hybrid.Vectors, embedder.EmbedQuery);
            var hybridEvaluator = RagEvaluator.ForHybrid(hybrid, embedder);

            const int TopK = 5;
            var groups = new (string Name, Case[] Cases)[]
            {
                ("semantic", Semantic),
                ("identifier", Identifier),
                ("mixed", Mixed),
            };

            _out.WriteLine($"  {"group",-12} {"dense R@K",10} {"hybrid R@K",11} {"dense MRR",10} {"hybrid MRR",11}");

            var allCases = new List<Case>();
            var allRetrievalCases = new List<RetrievalCase>();

            foreach (var group in groups)
            {
                var retrievalCases = ToRetrievalCases(group.Cases, chunkIdsByFile);
                var dense = denseEvaluator.EvaluateRetrieval(retrievalCases, TopK);
                var hyb = hybridEvaluator.EvaluateRetrieval(retrievalCases, TopK);

                allCases.AddRange(group.Cases);
                allRetrievalCases.AddRange(retrievalCases);

                _out.WriteLine(
                    $"  {group.Name,-12} {dense.RecallAtK,10:F2} {hyb.RecallAtK,11:F2} "
                    + $"{dense.MeanReciprocalRank,10:F3} {hyb.MeanReciprocalRank,11:F3}");
            }

            var denseAll = denseEvaluator.EvaluateRetrieval(allRetrievalCases, TopK);
            var hybridAll = hybridEvaluator.EvaluateRetrieval(allRetrievalCases, TopK);

            _out.WriteLine(
                $"  {"OVERALL",-12} {denseAll.RecallAtK,10:F2} {hybridAll.RecallAtK,11:F2} "
                + $"{denseAll.MeanReciprocalRank,10:F3} {hybridAll.MeanReciprocalRank,11:F3}");

            _out.WriteLine(string.Empty);
            _out.WriteLine("  per-case rank (0 = missed); lexical arm shown where hybrid did not improve:");

            for (var i = 0; i < allCases.Count; i++)
            {
                var denseRank = denseAll.Cases[i].Rank;
                var hybridRank = hybridAll.Cases[i].Rank;

                _out.WriteLine(
                    $"    [{allCases[i].Group,-10}] dense {denseRank}  hybrid {hybridRank}   \"{allCases[i].Query}\"");

                if (hybridRank == 0 || (denseRank > 0 && hybridRank > denseRank))
                {
                    var lexical = hybrid.Lexical.Search(allCases[i].Query, 3);
                    var shown = new List<string>();
                    for (var j = 0; j < lexical.Length; j++)
                    {
                        shown.Add($"{lexical[j].Id}({lexical[j].Score:F2})");
                    }
                    _out.WriteLine($"          lexical top-3: {string.Join(", ", shown)}");
                }
            }

            // The one claim the lexical arm is bought for. Everything else is reported, not asserted — the
            // corpus is real but the case list is hand-written, so pinning the other numbers would pin my
            // choice of questions rather than the retriever.
            var denseIdentifier = denseEvaluator.EvaluateRetrieval(ToRetrievalCases(Identifier, chunkIdsByFile), TopK);
            var hybridIdentifier = hybridEvaluator.EvaluateRetrieval(ToRetrievalCases(Identifier, chunkIdsByFile), TopK);

            Assert.True(
                hybridIdentifier.MeanReciprocalRank >= denseIdentifier.MeanReciprocalRank,
                $"hybrid identifier MRR {hybridIdentifier.MeanReciprocalRank:F3} fell below dense "
                + $"{denseIdentifier.MeanReciprocalRank:F3}");
        }

        private static List<RetrievalCase> ToRetrievalCases(
            Case[] cases, Dictionary<string, List<string>> chunkIdsByFile)
        {
            var result = new List<RetrievalCase>();

            foreach (var c in cases)
            {
                var expected = new List<string>();
                foreach (var file in c.ExpectedFiles)
                {
                    if (chunkIdsByFile.TryGetValue(file, out var ids))
                    {
                        expected.AddRange(ids);
                    }
                }

                Assert.True(expected.Count > 0, $"no chunks indexed for the expected file(s) of \"{c.Query}\"");
                result.Add(new RetrievalCase(c.Query, [.. expected]));
            }

            return result;
        }

        /// <summary>Mirrors <c>McpRagIndex.ChunkParagraphs</c> (internal to the Mcp assembly) so the corpus is
        /// split exactly as it would be in production.</summary>
        private static List<string> ChunkParagraphs(string text)
        {
            const int TargetChunkChars = 1200;

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

        // Walks up from the test output directory to the repository root (the folder holding Overfit.sln).
        private static string? FindDocsDirectory()
        {
            var directory = new DirectoryInfo(AppContext.BaseDirectory);

            for (var depth = 0; depth < 12 && directory is not null; depth++)
            {
                if (File.Exists(Path.Combine(directory.FullName, "Overfit.sln")))
                {
                    var docs = Path.Combine(directory.FullName, "docs");
                    return Directory.Exists(docs) ? docs : null;
                }

                directory = directory.Parent;
            }

            return null;
        }
    }
}
