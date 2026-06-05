// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// Result of an expected-source retrieval evaluation: aggregate <see cref="RecallAtK"/> and
    /// <see cref="MeanReciprocalRank"/> plus a per-case breakdown. Drive it from an xUnit test and assert
    /// <c>RecallAtK</c> stays above your bar — that is the "RAG is testable" contract.
    /// </summary>
    public sealed class RetrievalReport
    {
        public RetrievalReport(int topK, IReadOnlyList<CaseResult> cases)
        {
            ArgumentNullException.ThrowIfNull(cases);
            TopK = topK;
            Cases = cases;

            var hits = 0;
            var rrSum = 0.0;
            for (var i = 0; i < cases.Count; i++)
            {
                if (cases[i].Hit)
                {
                    hits++;
                    rrSum += 1.0 / cases[i].Rank;   // Rank is 1-based and > 0 when Hit.
                }
            }

            Hits = hits;
            RecallAtK = cases.Count == 0 ? 0.0 : hits / (double)cases.Count;
            MeanReciprocalRank = cases.Count == 0 ? 0.0 : rrSum / cases.Count;
        }

        /// <summary>The K used for the top-K retrieval.</summary>
        public int TopK { get; }

        /// <summary>Number of cases whose expected source appeared in the top-K.</summary>
        public int Hits { get; }

        /// <summary>Fraction of cases whose expected source was retrieved (hits / total).</summary>
        public double RecallAtK { get; }

        /// <summary>Mean reciprocal rank of the first expected source (0 when missed). Rewards ranking it higher.</summary>
        public double MeanReciprocalRank { get; }

        /// <summary>Per-case outcomes.</summary>
        public IReadOnlyList<CaseResult> Cases { get; }

        /// <summary>One case's outcome: what was expected, what was retrieved, and at which rank (if any).</summary>
        public sealed class CaseResult
        {
            public CaseResult(string query, IReadOnlyList<string> expectedSourceIds, IReadOnlyList<string> retrievedIds, int rank)
            {
                Query = query;
                ExpectedSourceIds = expectedSourceIds;
                RetrievedIds = retrievedIds;
                Rank = rank;
            }

            public string Query { get; }

            public IReadOnlyList<string> ExpectedSourceIds { get; }

            /// <summary>The ids returned in the top-K, best first.</summary>
            public IReadOnlyList<string> RetrievedIds { get; }

            /// <summary>1-based rank of the first expected id in <see cref="RetrievedIds"/>, or 0 if none matched.</summary>
            public int Rank { get; }

            /// <summary>True when an expected source was retrieved within the top-K.</summary>
            public bool Hit => Rank > 0;
        }
    }
}
