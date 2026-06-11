// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Reflection;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;
using TM = DevOnBike.Overfit.Ops.TensorMath;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// Regression tests for the <c>[module: SkipLocalsInit]</c> assembly attribute
    /// (applied in <c>Sources/Main/Properties/AssemblyInfo.cs</c>).
    ///
    /// SkipLocalsInit instructs the compiler to flip CIL's <c>.locals init</c>
    /// flag off. The runtime no longer zero-initializes the local frame on
    /// method entry — that's a perf win for hot methods, but it means
    /// <c>stackalloc</c> buffers contain GARBAGE on entry. Any code path
    /// reading a stackalloc'd buffer before fully writing to it now silently
    /// produces wrong results.
    ///
    /// These tests exercise the cases where the audit (in AssemblyInfo.cs)
    /// identified that an explicit <c>.Clear()</c> was required. If a future
    /// change drops that <c>.Clear()</c>, these tests should fail.
    /// </summary>
    public sealed class SkipLocalsInitTests
    {
        private readonly ITestOutputHelper _output;

        public SkipLocalsInitTests(ITestOutputHelper output)
        {
            _output = output;
        }

        // ── 1. Assembly attribute is actually applied ──────────────────────────

        /// <summary>
        /// Belt-and-braces: verify the attribute is present at module level.
        /// If someone deletes <c>AssemblyInfo.cs</c> or the attribute, the
        /// silent-correctness-loss surface (stackalloc + missing Clear) widens
        /// — we want a hard signal.
        /// </summary>
        [Fact]
        public void Module_HasSkipLocalsInitAttribute()
        {
            // Pick any type from Main; its module is the same module we care about.
            var module = typeof(TensorStorage<float>).Module;
            var attribute = module.GetCustomAttribute<System.Runtime.CompilerServices.SkipLocalsInitAttribute>();

            Assert.NotNull(attribute);
            _output.WriteLine($"Module: {module.Name}");
            _output.WriteLine($"SkipLocalsInit applied: yes");
        }

        // ── 2. LoRAWeight.ForwardAdd — accumulator must start at 0 ─────────────

        /// <summary>
        /// <see cref="LoRAWeight.ForwardAdd"/> stackalloc's <c>r[Rank]</c> and
        /// uses it as an accumulator (<c>TensorPrimitives.MultiplyAdd(_, _, r, r)</c>
        /// reads <c>r[k]</c> then writes). Without an explicit <c>.Clear()</c>
        /// after the stackalloc, the second call on the same thread reads
        /// leftover values from the previous call's <c>r</c> (same stack pointer,
        /// no zero-init), corrupting the result silently.
        ///
        /// Without the fix: <c>result2 ≠ result1</c> for identical inputs (drift
        /// accumulates across calls). With the fix: stable repeatable result.
        /// </summary>
        [Fact]
        public void LoRAWeight_ForwardAdd_StableAcrossRepeatedCalls()
        {
            var w = new LoRAWeight(inDim: 4, outDim: 2, rank: 2);
            var a = w.AMutable; a.Clear();
            var b = w.BMutable; b.Clear();
            a[0] = 1f; a[3] = 1f;
            b[0] = 2f; b[3] = 3f;

            float[] x = [5f, 7f, 0f, 0f];

            var result1 = new float[2];
            w.ForwardAdd(x, result1, scale: 1f);

            // Critical: same thread → same stack pointer for the next
            // stackalloc'd r[]. If r isn't cleared, it'd see result1's
            // accumulator values from the previous frame.
            var result2 = new float[2];
            w.ForwardAdd(x, result2, scale: 1f);

            var result3 = new float[2];
            w.ForwardAdd(x, result3, scale: 1f);

            _output.WriteLine($"call 1: [{result1[0]}, {result1[1]}]");
            _output.WriteLine($"call 2: [{result2[0]}, {result2[1]}]");
            _output.WriteLine($"call 3: [{result3[0]}, {result3[1]}]");

            // Expected values from existing LoRAWeight_ForwardAdd_MatchesManual:
            //   result[0] = 10, result[1] = 21.
            Assert.Equal(10f, result1[0], precision: 4);
            Assert.Equal(21f, result1[1], precision: 4);
            Assert.Equal(10f, result2[0], precision: 4);
            Assert.Equal(21f, result2[1], precision: 4);
            Assert.Equal(10f, result3[0], precision: 4);
            Assert.Equal(21f, result3[1], precision: 4);
        }

        // ── 3. LayerNormBackward — partial accumulators must start at 0 ────────

        /// <summary>
        /// <see cref="TensorMath.LayerNormBackward"/> stackalloc's
        /// <c>dGammaPartial[workerCount × C]</c> and <c>dBetaPartial[workerCount × C]</c>
        /// then explicitly <c>.Clear()</c>s them. Workers <c>+=</c> into their own
        /// slot; a sequential merge sums all slots into the final gradients.
        ///
        /// If the <c>.Clear()</c> were dropped under SkipLocalsInit, the
        /// partials would start with garbage, the merge would propagate that
        /// garbage into <c>gamma.Grad</c> / <c>beta.Grad</c>, and the second
        /// call would produce different gradients than the first for identical
        /// inputs (because the second call's stackalloc lands on the first
        /// call's residual frame).
        ///
        /// This test calls the backward path 3× and asserts that
        /// <c>gamma.Grad</c> / <c>beta.Grad</c> deltas are bit-identical (the
        /// accumulator <c>+=</c> means we compare per-call deltas, not absolute).
        /// </summary>
        [Fact]
        public void LayerNormBackward_StableAcrossRepeatedCalls()
        {
            // Shape that crosses the parallel threshold so the stackalloc path is taken.
            // numRows * C >= ParallelThreshold (4096). Use numRows = 64, C = 128.
            const int numRows = 64;
            const int C = 128;

            using var graph = new DevOnBike.Overfit.Autograd.ComputationGraph();

            using var inputStorage = new TensorStorage<float>(numRows * C);
            using var gammaStorage = new TensorStorage<float>(C);
            using var betaStorage = new TensorStorage<float>(C);

            var rng = new Random(42);
            for (var i = 0; i < numRows * C; i++) { inputStorage.AsSpan()[i] = (float)(rng.NextDouble() * 2 - 1); }
            for (var i = 0; i < C; i++) { gammaStorage.AsSpan()[i] = 1f + (float)(rng.NextDouble() * 0.1); }
            for (var i = 0; i < C; i++) { betaStorage.AsSpan()[i] = (float)(rng.NextDouble() * 0.01); }

            var input = new DevOnBike.Overfit.Autograd.AutogradNode(inputStorage, new TensorShape(numRows, C), requiresGrad: true);
            var gamma = new DevOnBike.Overfit.Autograd.AutogradNode(gammaStorage, new TensorShape(C), requiresGrad: true);
            var beta = new DevOnBike.Overfit.Autograd.AutogradNode(betaStorage, new TensorShape(C), requiresGrad: true);

            var snapshots = new (float[] gammaGrad, float[] betaGrad, float[] inputGrad)[3];

            for (var call = 0; call < 3; call++)
            {
                // Reset grads to zero before each call so we measure pure delta.
                gamma.GradView.AsSpan().Clear();
                beta.GradView.AsSpan().Clear();
                input.GradView.AsSpan().Clear();

                graph.Reset();
                var output = TM.LayerNorm(graph, input, gamma, beta);

                // Seed upstream gradient deterministically.
                output.GradView.AsSpan().Fill(0.01f);

                // Pull mean/invStd from graph tape — they were recorded by forward.
                // Easiest: re-run forward via TensorMath which records them; then
                // we can invoke backward by calling graph.Backward(output) below.
                graph.BackwardFromGrad(output);

                snapshots[call] = (
                    gamma.GradView.AsReadOnlySpan().ToArray(),
                    beta.GradView.AsReadOnlySpan().ToArray(),
                    input.GradView.AsReadOnlySpan().ToArray());

                output.Dispose();
            }

            // All three calls used identical inputs/seed-grad. Results must match
            // bit-identically — any drift means a stackalloc'd accumulator wasn't
            // cleared.
            for (var i = 0; i < C; i++)
            {
                Assert.Equal(snapshots[0].gammaGrad[i], snapshots[1].gammaGrad[i]);
                Assert.Equal(snapshots[0].gammaGrad[i], snapshots[2].gammaGrad[i]);
                Assert.Equal(snapshots[0].betaGrad[i], snapshots[1].betaGrad[i]);
                Assert.Equal(snapshots[0].betaGrad[i], snapshots[2].betaGrad[i]);
            }
            for (var i = 0; i < numRows * C; i++)
            {
                Assert.Equal(snapshots[0].inputGrad[i], snapshots[1].inputGrad[i]);
                Assert.Equal(snapshots[0].inputGrad[i], snapshots[2].inputGrad[i]);
            }

            _output.WriteLine($"3× LayerNormBackward calls — gamma/beta/input grads bit-identical.");
            _output.WriteLine($"  shape: [{numRows}, {C}], workerCount={DevOnBike.Overfit.Runtime.OverfitParallel.WorkerCount}");
            _output.WriteLine($"  partial buffer size per call: {DevOnBike.Overfit.Runtime.OverfitParallel.WorkerCount * C * 2 * 4} bytes (×2 for gamma+beta)");

            input.Dispose();
            gamma.Dispose();
            beta.Dispose();
        }
    }
}
