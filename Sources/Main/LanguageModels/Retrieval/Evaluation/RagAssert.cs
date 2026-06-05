// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// Framework-agnostic assertions that turn a RAG evaluation into a pass/fail gate. Call them from xUnit /
    /// NUnit / MSTest (or anywhere) — a failure throws <see cref="RagAssertionException"/> with the offending
    /// cases spelled out, so retrieval quality is protected in CI exactly like a unit test. This is what makes
    /// "RAG is testable" a workflow rather than a manual report read.
    /// </summary>
    public static class RagAssert
    {
        /// <summary>Asserts retrieval recall@K is at least <paramref name="minRecall"/> (0..1); lists the misses.</summary>
        public static void RecallAtLeast(RetrievalReport report, double minRecall)
        {
            ArgumentNullException.ThrowIfNull(report);
            if (report.RecallAtK >= minRecall)
            {
                return;
            }

            var sb = new StringBuilder();
            sb.Append($"RAG recall@{report.TopK} was {report.RecallAtK:P1}, below the required {minRecall:P1}. Missed:");
            foreach (var c in report.Cases)
            {
                if (!c.Hit)
                {
                    sb.Append($"\n  - \"{c.Query}\" — expected one of [{string.Join(", ", c.ExpectedSourceIds)}], got [{string.Join(", ", c.RetrievedIds)}]");
                }
            }

            throw new RagAssertionException(sb.ToString());
        }

        /// <summary>Asserts every retrieval case surfaced an expected source (recall@K == 100%).</summary>
        public static void AllRetrieved(RetrievalReport report) => RecallAtLeast(report, 1.0);

        /// <summary>Asserts every paraphrase group retrieved a stable source set (none below its Jaccard bar).</summary>
        public static void Stable(ParaphraseStabilityReport report)
        {
            ArgumentNullException.ThrowIfNull(report);
            if (report.UnstableCount == 0)
            {
                return;
            }

            var sb = new StringBuilder();
            sb.Append($"{report.UnstableCount} paraphrase group(s) retrieved unstable sources (mean Jaccard < {report.MinJaccard:0.##}):");
            foreach (var g in report.Groups)
            {
                if (!g.IsStable)
                {
                    sb.Append($"\n  - {g.Name}: mean overlap {g.MeanJaccard:0.##}");
                }
            }

            throw new RagAssertionException(sb.ToString());
        }

        /// <summary>Asserts no false-premise query found a confident source (no sprung traps).</summary>
        public static void NoGroundedFalsePremises(FalsePremiseReport report)
        {
            ArgumentNullException.ThrowIfNull(report);
            if (report.TrapsSprung == 0)
            {
                return;
            }

            var sb = new StringBuilder();
            sb.Append($"{report.TrapsSprung} false-premise query(ies) found a source at/above the grounded threshold {report.GroundedThreshold:0.##}:");
            foreach (var c in report.Cases)
            {
                if (c.Grounded)
                {
                    sb.Append($"\n  - \"{c.Query}\" matched {c.TopId} at {c.TopScore:0.###}");
                }
            }

            throw new RagAssertionException(sb.ToString());
        }

        /// <summary>Asserts the corpus has no near-duplicate document pairs.</summary>
        public static void NoNearDuplicates(IReadOnlyList<CorpusLinter.DuplicatePair> duplicates)
        {
            ArgumentNullException.ThrowIfNull(duplicates);
            if (duplicates.Count == 0)
            {
                return;
            }

            var sb = new StringBuilder();
            sb.Append($"{duplicates.Count} near-duplicate document pair(s):");
            foreach (var d in duplicates)
            {
                sb.Append($"\n  - {d.FirstId} ~ {d.SecondId} ({d.Similarity:0.###})");
            }

            throw new RagAssertionException(sb.ToString());
        }

        /// <summary>Asserts no document is unreachable by the evaluated query set.</summary>
        public static void NoOrphans(IReadOnlyList<string> orphans)
        {
            ArgumentNullException.ThrowIfNull(orphans);
            if (orphans.Count == 0)
            {
                return;
            }

            throw new RagAssertionException($"{orphans.Count} orphan document(s) no query retrieved: [{string.Join(", ", orphans)}]");
        }

        /// <summary>Asserts no document is too short / empty to carry an answer.</summary>
        public static void NoShortDocuments(IReadOnlyList<string> shortDocuments)
        {
            ArgumentNullException.ThrowIfNull(shortDocuments);
            if (shortDocuments.Count == 0)
            {
                return;
            }

            throw new RagAssertionException($"{shortDocuments.Count} document(s) too short to carry an answer: [{string.Join(", ", shortDocuments)}]");
        }
    }
}
