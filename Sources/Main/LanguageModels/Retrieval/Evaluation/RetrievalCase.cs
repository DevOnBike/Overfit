// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// One expected-source retrieval test: a <see cref="Query"/> that SHOULD pull at least one of
    /// <see cref="ExpectedSourceIds"/> into the top-K. This is the unit of "RAG is testable" — it asserts the
    /// retriever surfaces the right document(s) for a question, independent of what the LLM later does with them.
    /// </summary>
    public sealed class RetrievalCase
    {
        public RetrievalCase(string query, params string[] expectedSourceIds)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(query);
            ArgumentNullException.ThrowIfNull(expectedSourceIds);
            Query = query;
            ExpectedSourceIds = expectedSourceIds;
        }

        /// <summary>The user question to embed and search with.</summary>
        public string Query
        {
            get;
        }

        /// <summary>The document id(s) a correct retriever must surface in the top-K (any one counts as a hit).</summary>
        public IReadOnlyList<string> ExpectedSourceIds
        {
            get;
        }
    }
}
