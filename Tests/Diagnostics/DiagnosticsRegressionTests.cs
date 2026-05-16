// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tests.Diagnostics.Tracing;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    public sealed class DiagnosticsRegressionTests
    {
        private readonly ITestOutputHelper _output;

        public DiagnosticsRegressionTests(ITestOutputHelper output)
        {
            _output = output;
        }

        // Regression gate for the MNIST diagnostics trace. Skipped because no producer
        // writes diagnostics/mnist/epoch_05.json — the Mnist_FullTrain60k_CnnBeastMode
        // benchmark does not yet export per-epoch trace JSON (tracked in ROADMAP under
        // diagnostics trace export). Until that exists this test could only ever pass
        // vacuously via early-return, which hides the missing coverage. Remove the Skip
        // once the benchmark emits epoch JSON.
        [Fact(Skip = "No producer writes diagnostics/mnist/epoch_05.json yet — see ROADMAP diagnostics trace export.")]
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