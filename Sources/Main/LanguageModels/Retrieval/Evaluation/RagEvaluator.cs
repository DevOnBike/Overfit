// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Embeddings;

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// Makes a RAG retriever <b>testable</b>. Given an indexed <see cref="VectorStore"/> and a query-embedding
    /// function, it runs three checks you can assert on in CI:
    /// <list type="bullet">
    ///   <item><b>Expected-source recall</b> — does each question pull the right document into the top-K?
    ///     (<see cref="EvaluateRetrieval"/>)</item>
    ///   <item><b>Paraphrase stability</b> — do rephrasings of one question retrieve the same documents?
    ///     (<see cref="EvaluateParaphraseStability"/>)</item>
    ///   <item><b>False-premise traps</b> — do un-grounded questions correctly find no confident source, so the
    ///     LLM isn't handed a spurious passage to hallucinate from? (<see cref="EvaluateFalsePremise"/>)</item>
    /// </list>
    /// In-process, pure .NET, deterministic — the retrieval-quality counterpart to a unit test.
    /// </summary>
    public sealed class RagEvaluator
    {
        private readonly VectorStore _store;
        private readonly Func<string, float[]> _embedQuery;

        /// <summary>Creates an evaluator over an indexed store and a query-embedding delegate (e.g.
        /// <c>embedder.EmbedQuery</c>, or a deterministic fake in a unit test).</summary>
        public RagEvaluator(VectorStore store, Func<string, float[]> embedQuery)
        {
            ArgumentNullException.ThrowIfNull(store);
            ArgumentNullException.ThrowIfNull(embedQuery);
            _store = store;
            _embedQuery = embedQuery;
        }

        /// <summary>Convenience factory over a <see cref="SentenceEmbedder"/> — uses <c>EmbedQuery</c> so the
        /// model's retrieval-side query prefix (BGE / E5) is applied consistently with how the corpus was indexed.</summary>
        public static RagEvaluator ForEmbedder(VectorStore store, SentenceEmbedder embedder)
        {
            ArgumentNullException.ThrowIfNull(embedder);
            return new RagEvaluator(store, embedder.EmbedQuery);
        }

        /// <summary>Runs every <see cref="RetrievalCase"/> and reports recall@K + MRR + per-case ranks.</summary>
        public RetrievalReport EvaluateRetrieval(IEnumerable<RetrievalCase> cases, int topK = 5)
        {
            ArgumentNullException.ThrowIfNull(cases);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);

            var results = new List<RetrievalReport.CaseResult>();
            foreach (var c in cases)
            {
                var retrieved = RetrieveIds(c.Query, topK);

                var rank = 0;
                for (var i = 0; i < retrieved.Length && rank == 0; i++)
                {
                    for (var j = 0; j < c.ExpectedSourceIds.Count; j++)
                    {
                        if (string.Equals(retrieved[i], c.ExpectedSourceIds[j], StringComparison.Ordinal))
                        {
                            rank = i + 1;   // 1-based
                            break;
                        }
                    }
                }

                results.Add(new RetrievalReport.CaseResult(c.Query, c.ExpectedSourceIds, retrieved, rank));
            }

            return new RetrievalReport(topK, results);
        }

        /// <summary>Runs every <see cref="ParaphraseGroup"/> and reports mean pairwise retrieval overlap (Jaccard).</summary>
        public ParaphraseStabilityReport EvaluateParaphraseStability(
            IEnumerable<ParaphraseGroup> groups, int topK = 5, double minJaccard = 0.6)
        {
            ArgumentNullException.ThrowIfNull(groups);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);

            var groupResults = new List<ParaphraseStabilityReport.GroupResult>();
            foreach (var g in groups)
            {
                var perVariant = new List<IReadOnlyList<string>>(g.Variants.Count);
                var sets = new List<HashSet<string>>(g.Variants.Count);
                for (var v = 0; v < g.Variants.Count; v++)
                {
                    var ids = RetrieveIds(g.Variants[v], topK);
                    perVariant.Add(ids);
                    var set = new HashSet<string>(StringComparer.Ordinal);
                    for (var i = 0; i < ids.Length; i++)
                    {
                        set.Add(ids[i]);
                    }
                    sets.Add(set);
                }

                var pairCount = 0;
                var jaccardSum = 0.0;
                for (var a = 0; a < sets.Count; a++)
                {
                    for (var b = a + 1; b < sets.Count; b++)
                    {
                        jaccardSum += Jaccard(sets[a], sets[b]);
                        pairCount++;
                    }
                }

                var mean = pairCount == 0 ? 1.0 : jaccardSum / pairCount;
                groupResults.Add(new ParaphraseStabilityReport.GroupResult(g.Name, mean, mean >= minJaccard, perVariant));
            }

            return new ParaphraseStabilityReport(topK, minJaccard, groupResults);
        }

        /// <summary>Runs every <see cref="FalsePremiseCase"/> and flags those whose top match clears
        /// <paramref name="groundedThreshold"/> (a sprung trap — the corpus offered a confident source for an
        /// un-grounded question).</summary>
        public FalsePremiseReport EvaluateFalsePremise(IEnumerable<FalsePremiseCase> cases, double groundedThreshold = 0.5)
        {
            ArgumentNullException.ThrowIfNull(cases);

            var results = new List<FalsePremiseReport.CaseResult>();
            foreach (var c in cases)
            {
                var matches = _store.Search(_embedQuery(c.Query), 1);
                string? topId = null;
                var topScore = 0f;
                if (matches.Length > 0)
                {
                    topId = matches[0].Id;
                    topScore = matches[0].Score;
                }

                results.Add(new FalsePremiseReport.CaseResult(c.Query, topId, topScore, topScore >= groundedThreshold, c.Note));
            }

            return new FalsePremiseReport(groundedThreshold, results);
        }

        private string[] RetrieveIds(string query, int topK)
        {
            var matches = _store.Search(_embedQuery(query), topK);
            var ids = new string[matches.Length];
            for (var i = 0; i < matches.Length; i++)
            {
                ids[i] = matches[i].Id;
            }
            return ids;
        }

        private static double Jaccard(HashSet<string> a, HashSet<string> b)
        {
            if (a.Count == 0 && b.Count == 0)
            {
                return 1.0;   // both retrieved nothing — trivially "agree".
            }

            var intersection = 0;
            foreach (var x in a)
            {
                if (b.Contains(x))
                {
                    intersection++;
                }
            }

            var union = a.Count + b.Count - intersection;
            return union == 0 ? 1.0 : intersection / (double)union;
        }
    }
}
