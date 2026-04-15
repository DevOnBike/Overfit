// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Abstractions
{
    /// <summary>
    /// Ujednolicony interfejs dla algorytmów normalizacji (np. Z-Score, Min-Max, Log1p).
    /// </summary>
    public interface IFeatureNormalizer
    {
        bool IsFrozen { get; }

        /// <summary>
        /// Uczy normalizator na podstawie danych historycznych (Offline).
        /// </summary>
        void FitBatch(ReadOnlySpan<float> data);

        /// <summary>
        /// Zamraża wyliczone parametry (np. średnią, wariancję, min/max) do użycia na produkcji.
        /// </summary>
        void Freeze();

        /// <summary>
        /// Normalizuje dane w miejscu. Wymaga wcześniejszego zamrożenia (Freeze).
        /// </summary>
        void TransformInPlace(Span<float> data);

        /// <summary>
        /// Czeruje stan algorytmu.
        /// </summary>
        void Reset();

        /// <summary>
        /// Zapisuje zamrożone parametry do strumienia binarnego.
        /// </summary>
        void Save(BinaryWriter bw);

        /// <summary>
        /// Wczytuje zamrożone parametry ze strumienia binarnego.
        /// </summary>
        void Load(BinaryReader br);
    }
}