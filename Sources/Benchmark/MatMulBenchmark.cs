namespace Benchmarks
{
    /*
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    // [GcForce]
    [DisassemblyDiagnoser(maxDepth: 2)]
    public class Conv2DBenchmark
    {
        private AutogradNode _input = null!;
        private AutogradNode _weights = null!;
        private Conv2DContext _ctx = null!;

        // typowe wymiary: 1 kanał wejściowy, 16 filtrów, obraz 8×8, kernel 3×3
        private const int InC = 1, OutC = 16, H = 8, W = 8, K = 3, Batch = 32;

        [GlobalSetup]
        public void Setup()
        {
            _input = new AutogradNode(new FloatFastMatrix(Batch, InC * H * W));
            _weights = new AutogradNode(new FloatFastMatrix(OutC, InC * K * K));
            _ctx = new Conv2DContext(InC, OutC, H, W, K, batchSize: Batch); // ← dodaj batchSize
        }

        [Benchmark(Baseline = true)]
        public void Original()
        {
            using var result = TensorMath.Conv2D(_input, _weights, InC, OutC, H, W, K).Data;
        }

        [Benchmark]
        public void Optimized()
        {
            using var result = TensorMath.Conv2DOptimized(_input, _weights, InC, OutC, H, W, K, _ctx).Data;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _input.Dispose(); _weights.Dispose(); _ctx.Dispose();
        }
    }
    */
}