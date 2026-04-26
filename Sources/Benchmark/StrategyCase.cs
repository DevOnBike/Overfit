namespace Benchmarks
{
    public sealed class StrategyCase
    {
        public StrategyCase(
            int populationSize,
            int parameterCount)
        {
            PopulationSize = populationSize;
            ParameterCount = parameterCount;
        }

        public int PopulationSize { get; }

        public int ParameterCount { get; }

        public override string ToString()
        {
            return $"P{PopulationSize}_N{ParameterCount}";
        }
    }
}