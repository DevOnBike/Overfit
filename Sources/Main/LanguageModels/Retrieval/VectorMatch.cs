// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval
{
    /// <summary>
    /// One hit from a <see cref="VectorStore"/> search: the stored item's <see cref="Id"/> and
    /// <see cref="Payload"/>, and its cosine similarity <see cref="Score"/> to the query (1 = identical
    /// direction, 0 = orthogonal).
    /// </summary>
    public readonly struct VectorMatch
    {
        public VectorMatch(string id, float score, string? payload)
        {
            Id = id;
            Score = score;
            Payload = payload;
        }

        public string Id
        {
            get;
        }

        public float Score
        {
            get;
        }

        public string? Payload
        {
            get;
        }
    }
}
