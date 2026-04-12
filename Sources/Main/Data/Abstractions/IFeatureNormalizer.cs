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
    }
}