namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IPopulationEvaluator
    {
        void Evaluate(ReadOnlySpan<float> populationData, Span<float> fitnessOut, int populationSize, int parameterCount);
    }
}