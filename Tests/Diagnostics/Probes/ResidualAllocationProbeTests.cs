// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    ///     Per-operation allocation profiling for the ResidualBlock training forward path,
    ///     plus LinearLayer and BatchNorm1D measured in isolation for reference.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Rationale: the MNIST training test reports <c>residual ≈ 16 MB/epoch</c> of
    ///         managed allocation during forward. Analysing the code alone we couldn't
    ///         identify a single dominant source (many small allocations in AllocateNode
    ///         wrappers, PooledBuffer rents, array literals in Record). This test removes
    ///         all the outer noise (no training loop, no diagnostics, no data loader)
    ///         and isolates the forward cost per operation with warm ArrayPool and stable
    ///         GC state.
    ///     </para>
    ///     <para>
    ///         Output is printed via <see cref="ITestOutputHelper"/>. Run with
    ///         <c>dotnet test --filter ResidualAllocationProbeTests -c Release</c>
    ///         and look at the test's standard output for the per-op numbers.
    ///     </para>
    /// </remarks>
    public sealed class ResidualAllocationProbeTests
    {
        // Matches the MNIST setup that reports 16 MB/epoch in residual: batch=64, hidden=1352.
        private const int Batch = 64;
        private const int Hidden = 1352;
        private const int Iterations = 100;
        private const int Warmup = 20;

        private readonly ITestOutputHelper _output;

        public ResidualAllocationProbeTests(ITestOutputHelper output)
        {
            _output = output;
        }

        // ==============================================================================
        // The headline test: full ResidualBlock.Forward as a single block.
        // If this is close to 16 KB/call (×937 batches = ~16 MB/epoch), we've reproduced
        // the MNIST number in isolation and can trust all the finer-grained numbers.
        // ==============================================================================

        [Fact]
        public void ResidualBlock_Forward_AllocationPerCall()
        {
            using var block = new ResidualBlock(Hidden);
            block.Train();

            using var graph = new ComputationGraph();
            using var input = MakeInput();

            var bytesPerCall = Measure("ResidualBlock.Forward", () =>
            {
                var output = block.Forward(graph, input);
                // Caller is responsible for Reset() — that clears the tape and resets the
                // arena offset without touching the pool.
                graph.Reset();
            });

            _output.WriteLine($"=== ResidualBlock.Forward: {bytesPerCall:F0} B/call ===");
            _output.WriteLine($"Projected per 937 MNIST batches: {bytesPerCall * 937 / (1024.0 * 1024.0):F2} MB");
        }

        // ==============================================================================
        // Per-operation breakdown. Each measures ONE operation in isolation (with a fresh
        // graph per iteration) so we can attribute allocation to a specific op.
        //
        // NOTE: these are *not* identical to the numbers inside ResidualBlock.Forward —
        // there the ops share a graph whose arena is reused across ops, whereas here each
        // test creates and resets its own graph. But the managed wrapper allocations
        // (AutogradNode, TensorStorage class headers, array literals in Record, closures
        // inside Parallel.For) are independent of arena state, so these numbers expose
        // exactly that class of allocation.
        // ==============================================================================

        [Fact]
        public void LinearLayer_Forward_AllocationPerCall()
        {
            using var linear = new LinearLayer(Hidden, Hidden);
            linear.Train();

            using var graph = new ComputationGraph();
            using var input = MakeInput();

            var bytesPerCall = Measure("LinearLayer.Forward", () =>
            {
                _ = linear.Forward(graph, input);
                graph.Reset();
            });

            _output.WriteLine($"=== LinearLayer.Forward ({Hidden}→{Hidden}): {bytesPerCall:F0} B/call ===");
        }

        /// <summary>
        ///     Same LinearLayer.Forward but with a size small enough to trip the sequential
        ///     path (N*K*M &lt; ParallelThreshold=4096 inside TensorMath.Linear). Difference
        ///     between this test and the large one above isolates the cost of
        ///     <see cref="Parallel.For"/> infrastructure (Task / RangeWorker / TaskReplicator
        ///     internals + closure capture) from the rest of the Linear op's allocations.
        /// </summary>
        [Fact]
        public void LinearLayer_Forward_SmallMatrix_AllocationPerCall()
        {
            const int smallBatch = 4;
            const int smallSize = 8; // 4 * 8 * 8 = 256 ≪ 4096

            using var linear = new LinearLayer(smallSize, smallSize);
            linear.Train();

            using var graph = new ComputationGraph();
            var storage = new TensorStorage<float>(smallBatch * smallSize, clearMemory: true);
            using var input = new AutogradNode(storage, new TensorShape(smallBatch, smallSize), requiresGrad: true);

            var bytesPerCall = Measure("LinearLayer.Forward (small)", () =>
            {
                _ = linear.Forward(graph, input);
                graph.Reset();
            });

            _output.WriteLine($"=== LinearLayer.Forward ({smallSize}→{smallSize}, N={smallBatch}, sequential path): {bytesPerCall:F0} B/call ===");
        }

        [Fact]
        public void BatchNorm1D_Forward_AllocationPerCall()
        {
            using var bn = new BatchNorm1D(Hidden);
            bn.Train();

            using var graph = new ComputationGraph();
            using var input = MakeInput();

            var bytesPerCall = Measure("BatchNorm1D.Forward", () =>
            {
                _ = bn.Forward(graph, input);
                graph.Reset();
            });

            _output.WriteLine($"=== BatchNorm1D.Forward (C={Hidden}): {bytesPerCall:F0} B/call ===");
        }

        [Fact]
        public void ReLU_Forward_AllocationPerCall()
        {
            using var graph = new ComputationGraph();
            using var input = MakeInput();

            var bytesPerCall = Measure("TensorMath.ReLU", () =>
            {
                _ = TensorMath.ReLU(graph, input);
                graph.Reset();
            });

            _output.WriteLine($"=== TensorMath.ReLU (C={Hidden}): {bytesPerCall:F0} B/call ===");
        }

        // Note: we don't isolate AddInPlace here. That op writes back into its target
        // ("output IS target"), so Reset() ends up disposing our pre-made target node —
        // the second iteration then throws ObjectDisposedException. In the real
        // ResidualBlock.Forward this is fine because target is itself a freshly-allocated
        // intermediate from the previous BatchNorm op in the same tape. Mocking that
        // upstream creation here would just re-measure node allocation, which we already
        // capture in the BatchNorm1D and LinearLayer probes. AddInPlace itself is declared
        // zero-alloc (it only does one TensorPrimitives.Add + one Record call), and the
        // end-to-end ResidualBlock.Forward probe already covers it implicitly.

        // ==============================================================================
        // Helpers
        // ==============================================================================

        private static AutogradNode MakeInput()
        {
            // Input is a "real" parameter-like node: its storage is pool-backed and it
            // carries requiresGrad=true so downstream ops inherit gradient allocation.
            var storage = new TensorStorage<float>(Batch * Hidden, clearMemory: true);
            return new AutogradNode(storage, new TensorShape(Batch, Hidden), requiresGrad: true);
        }

        /// <summary>
        ///     Executes <paramref name="action"/> <see cref="Warmup"/> times to prime
        ///     ArrayPool and JIT, forces GC to a stable state, then measures
        ///     <see cref="Iterations"/> executions and returns bytes allocated per call.
        /// </summary>
        /// <remarks>
        ///     Using <see cref="GC.GetAllocatedBytesForCurrentThread"/> (not
        ///     GetTotalAllocatedBytes) to avoid counting cross-thread allocations made by
        ///     the runtime — we only care about what our own thread is putting on the heap.
        ///     If the production code fans out via Parallel.For, those worker allocations
        ///     will NOT be counted here; that is a known limitation of this probe.
        /// </remarks>
        private double Measure(string label, Action action)
        {
            for (var i = 0; i < Warmup; i++)
            {
                action();
            }

            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);

            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < Iterations; i++)
            {
                action();
            }
            var after = GC.GetAllocatedBytesForCurrentThread();

            var total = after - before;
            var perCall = (double)total / Iterations;

            _output.WriteLine($"[{label}] warmup={Warmup}, iterations={Iterations}, total={total} B, per-call={perCall:F1} B");
            return perCall;
        }
    }
}