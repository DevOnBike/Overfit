namespace DevOnBike.Overfit.Optimizers
{
    public interface IOptimizer
    {
        double LearningRate { get; set; }

        // Krok optymalizacji (aktualizacja wag)
        void Step();

        // Czyszczenie gradientów przed nowym batchem
        void ZeroGrad();
    }
}