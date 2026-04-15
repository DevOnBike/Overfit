using Xunit;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public sealed class DiagnosticsRegressionTests
    {
        private readonly ITestOutputHelper _output;

        public DiagnosticsRegressionTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void LatestMnistTrace_CanBeCompared_AgainstBaseline_WhenBaselineExists()
        {
            var traceDir = Path.Combine(AppContext.BaseDirectory, "diagnostics", "mnist");
            var baselinePath = Path.Combine(traceDir, "baseline_epoch_05.json");
            var currentPath = Path.Combine(traceDir, "epoch_05.json");

            if (!File.Exists(currentPath))
            {
                _output.WriteLine("Current diagnostics trace not found. Run Mnist_FullTrain60k_CnnBeastMode_Benchmark first.");
                return;
            }

            if (!File.Exists(baselinePath))
            {
                _output.WriteLine("Baseline diagnostics trace not found.");
                _output.WriteLine($"Create one by copying: {currentPath}");
                _output.WriteLine($"to: {baselinePath}");
                return;
            }

            var baseline = DiagnosticsTraceModel.Load(baselinePath);
            var current = DiagnosticsTraceModel.Load(currentPath);
            var diff = DiagnosticsTraceComparer.Compare(baseline, current);

            _output.WriteLine(DiagnosticsTraceComparer.Format(diff));

            // Porównujemy tylko stabilną epokę końcową, nie epoch_01.
            Assert.True(current.TapeOps == baseline.TapeOps,
            $"Tape op count changed. baseline={baseline.TapeOps}, current={current.TapeOps}");

            Assert.True(current.GraphBackwardMs <= baseline.GraphBackwardMs * 1.35,
            $"Backward regression too high. baseline={baseline.GraphBackwardMs:F1}ms current={current.GraphBackwardMs:F1}ms");

            Assert.True(current.GraphAllocatedBytes <= (long)(baseline.GraphAllocatedBytes * 1.20),
            $"Backward allocation regression too high. baseline={baseline.GraphAllocatedBytes} current={current.GraphAllocatedBytes}");
        }
    }
}