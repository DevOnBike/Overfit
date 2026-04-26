using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class OpenAiEsStrategyTinyBenchmarks : OpenAiEsStrategyBenchmarkBase
    {
        private const int AskOperationsPerInvoke = 65_536;
        private const int AskThenTellOperationsPerInvoke = 32_768;

        [Params(false, true)]
        public bool UseAdamParam { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            SetupCore(
            populationSize: 256,
            parameterCount: 64,
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