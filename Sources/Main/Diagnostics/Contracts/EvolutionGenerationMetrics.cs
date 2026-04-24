namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly struct EvolutionGenerationMetrics
    {
        public int Generation { get; }
        public float BestFitness { get; }
        public TimeSpan TotalElapsed { get; }
        public TimeSpan AskElapsed { get; }
        public TimeSpan EvaluateElapsed { get; }
        public TimeSpan TellElapsed { get; }

        public EvolutionGenerationMetrics(
            int generation,
            float bestFitness,
            TimeSpan totalElapsed,
            TimeSpan askElapsed,
            TimeSpan evaluateElapsed,
            TimeSpan tellElapsed)
        {
            Generation = generation;
            BestFitness = bestFitness;
            TotalElapsed = totalElapsed;
            AskElapsed = askElapsed;
            EvaluateElapsed = evaluateElapsed;
            TellElapsed = tellElapsed;
        }
    }
}