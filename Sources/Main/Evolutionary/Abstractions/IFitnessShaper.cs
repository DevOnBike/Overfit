namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IFitnessShaper
    {
        void Shape(ReadOnlySpan<float> rawFitness, Span<float> shapedFitness);
    }
}