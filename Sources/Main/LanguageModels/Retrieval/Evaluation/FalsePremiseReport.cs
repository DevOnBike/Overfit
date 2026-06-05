// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// Result of a false-premise evaluation: for each trap query, the top retrieval similarity and whether it
    /// cleared the "grounded" threshold (i.e. the corpus offered a confident — and therefore dangerous — match
    /// for an un-grounded question). <see cref="TrapsSprung"/> is the count you want at zero.
    /// </summary>
    public sealed class FalsePremiseReport
    {
        public FalsePremiseReport(double groundedThreshold, IReadOnlyList<CaseResult> cases)
        {
            ArgumentNullException.ThrowIfNull(cases);
            GroundedThreshold = groundedThreshold;
            Cases = cases;

            var sprung = 0;
            for (var i = 0; i < cases.Count; i++)
            {
                if (cases[i].Grounded)
                {
                    sprung++;
                }
            }

            TrapsSprung = sprung;
        }

        /// <summary>The top-similarity above which a false-premise query is considered (dangerously) grounded.</summary>
        public double GroundedThreshold { get; }

        /// <summary>How many trap queries found a match at or above the threshold (want 0).</summary>
        public int TrapsSprung { get; }

        public IReadOnlyList<CaseResult> Cases { get; }

        /// <summary>One trap query's outcome: its top similarity, the matched id, and whether it tripped the trap.</summary>
        public sealed class CaseResult
        {
            public CaseResult(string query, string? topId, float topScore, bool grounded, string? note)
            {
                Query = query;
                TopId = topId;
                TopScore = topScore;
                Grounded = grounded;
                Note = note;
            }

            public string Query { get; }

            /// <summary>Id of the top match (null if the store was empty).</summary>
            public string? TopId { get; }

            /// <summary>Cosine similarity of the top match.</summary>
            public float TopScore { get; }

            /// <summary>True when <see cref="TopScore"/> ≥ the grounded threshold — a sprung trap.</summary>
            public bool Grounded { get; }

            public string? Note { get; }
        }
    }
}
