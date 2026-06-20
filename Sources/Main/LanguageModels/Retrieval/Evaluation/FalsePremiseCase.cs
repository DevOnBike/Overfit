// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// A "trap" query whose premise is NOT grounded in the corpus (it asks about something the documents don't
    /// cover, or assumes a false fact). A safe RAG should find no confidently-similar source — if the top match
    /// scores high anyway, the pipeline is at risk of feeding a spurious passage to the LLM and grounding a
    /// hallucinated answer. The check flags cases whose top similarity clears a "grounded" threshold.
    /// </summary>
    public sealed class FalsePremiseCase
    {
        public FalsePremiseCase(string query, string? note = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(query);
            Query = query;
            Note = note;
        }

        /// <summary>The un-grounded / false-premise question.</summary>
        public string Query
        {
            get;
        }

        /// <summary>Optional human note on why this premise is false (for the report).</summary>
        public string? Note
        {
            get;
        }
    }
}
