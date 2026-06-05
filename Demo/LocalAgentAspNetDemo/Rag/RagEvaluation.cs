// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation;

namespace DevOnBike.Overfit.Demo.LocalAgent.Rag
{
    /// <summary>Request body for <c>POST /rag/eval</c>. Every section is optional — send only the checks you
    /// want. The corpus lint (near-duplicates / short docs / orphans) always runs over the indexed corpus.</summary>
    public sealed class RagEvalRequest
    {
        /// <summary>Expected-source cases: each query SHOULD retrieve a chunk whose id contains <c>ExpectedSource</c>
        /// (e.g. a file name like <c>rodo.md</c>).</summary>
        public List<RetrievalCaseDto> Retrieval { get; set; } = [];

        /// <summary>Paraphrase groups: variants of one question that SHOULD retrieve the same chunks.</summary>
        public List<ParaphraseGroupDto> Paraphrase { get; set; } = [];

        /// <summary>False-premise / un-grounded queries that should find NO confident source.</summary>
        public List<string> FalsePremise { get; set; } = [];

        public int TopK { get; set; } = 5;
        public double GroundedThreshold { get; set; } = 0.5;
        public double MinJaccard { get; set; } = 0.6;
        public double DuplicateThreshold { get; set; } = 0.97;
        public int MinDocChars { get; set; } = 40;
    }

    public sealed class RetrievalCaseDto
    {
        public string Query { get; set; } = string.Empty;

        /// <summary>A substring of the expected chunk id (typically the source file name).</summary>
        public string ExpectedSource { get; set; } = string.Empty;
    }

    public sealed class ParaphraseGroupDto
    {
        public string Name { get; set; } = string.Empty;
        public List<string> Variants { get; set; } = [];
    }

    /// <summary>Combined result of <c>POST /rag/eval</c> — the harness reports plus the corpus-lint findings.</summary>
    public sealed record RagEvalResult(
        RetrievalReport? Retrieval,
        ParaphraseStabilityReport? Paraphrase,
        FalsePremiseReport? FalsePremise,
        IReadOnlyList<CorpusLinter.DuplicatePair> NearDuplicates,
        IReadOnlyList<string> ShortDocuments,
        IReadOnlyList<string> Orphans);
}
