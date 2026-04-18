namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    /// Operatory selekcji (wybieranie najlepszych do reprodukcji).
    /// </summary>
    public interface ISelectionOperator
    {
        /// <summary>
        /// Sortuje lub wybiera indeksy populacji na podstawie tablicy fitness.
        /// Nie kopiuje genów, operuje tylko na indeksach (Zero-Alloc!).
        /// </summary>
        void Select(ReadOnlySpan<float> fitness, Span<int> selectedIndicesOut);
    }
}