// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval
{
    /// <summary>
    /// One embedded chunk handed to <see cref="PersistentVectorStore.IndexSource"/>: a stable <paramref name="Id"/>
    /// (so retrieval results are reproducible across re-indexes), its embedding <paramref name="Vector"/>, and an
    /// optional <paramref name="Payload"/> (e.g. the source text snippet returned with a match).
    /// </summary>
    public readonly record struct VectorChunk(string Id, float[] Vector, string? Payload = null);
}
