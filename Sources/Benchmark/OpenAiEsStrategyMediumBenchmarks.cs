using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class OpenAiEsStrategyMediumBenchmarks : OpenAiEsStrategyBenchmarkBase
    {
        private const int AskOperationsPerInvoke = 32_768;
        private const int AskThenTellOperationsPerInvoke = 16_384;

        [ParamsSource(nameof(Cases))]
        public StrategyCase Case { get; set; } = null!;

        [Params(false, true)]
        public bool UseAdamParam { get; set; }

        public IEnumerable<StrategyCase> Cases()
        {
            yield return new StrategyCase(
            populationSize: 256,
            parameterCount: 256);

            yield return new StrategyCase(
            populationSize: 1024,
            parameterCount: 64);

            yield return new StrategyCase(
            populationSize: 256,
            parameterCount: 1024);
        }

        [GlobalSetup]
        public void Setup()
        {
            SetupCore(
            populationSize: Case.PopulationSize,
            parameterCount: Case.ParameterCount,
            useAdam: UseAdamParam);
        }

        [Benchmark(OperationsPerInvoke = AskOperationsPerInvoke)]
        public float Ask()
        {
            return AskCore(AskOperationsPerInvoke);
        }

        [Benchmark(OperationsPerInvoke = AskThenTellOperationsPerInvoke)]
        public float AskThenTell()
        {
            return AskThenTellCore(AskThenTellOperationsPerInvoke);
        }
    }
}