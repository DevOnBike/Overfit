// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Measure-first gate for the "speed up QLoRA training forward" lever: the training matmul
    /// (<c>ComputationGraph.FrozenQuantizedLinear</c>: dequant each Q4_K row to F32 once, then F32 dots
    /// across the batch) vs the inference batched kernel (<c>Q4KDotKernel.ProjectBatched</c>: quantize
    /// activations to Q8K once, then INT8 maddubs dots straight on the quantized rows — no F32 dequant).
    /// Same shapes as a real Qwen-3B QLoRA FFN step (batch 256, 2048→11008). Reports ms + speedup +
    /// numeric divergence (the int8 path adds activation-quant noise — the accuracy cost of the win).
    /// Gate: wire the int8 forward only if speedup ≥ 1.3×. [LongFact] — CPU-heavy microbench.
    /// </summary>
    public sealed class QloraForwardKernelBenchTests
    {
        private readonly ITestOutputHelper _out;
        public QloraForwardKernelBenchTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void TrainingForward_F32Dequant_vs_Int8Batched()
        {
            const int n = 256, k = 2048, m = 11008;
            const int reps = 3;

            // Synthesize a Q4_K weight from random F32 (the GGUF-equivalent base).
            var rnd = new Random(42);
            var f32 = new float[(long)m * k];
            for (var i = 0; i < f32.Length; i++) { f32[i] = (float)(rnd.NextDouble() * 2 - 1) * 0.08f; }
            var weight = new Q4KWeight(GgmlQuant.QuantizeQ4_K(f32, k, m), k, m);

            var inputData = new float[n * k];
            for (var i = 0; i < inputData.Length; i++) { inputData[i] = (float)(rnd.NextDouble() * 2 - 1); }

            // ── A: training path (FrozenQuantizedLinear forward) ──
            using var graph = new ComputationGraph();
            using var inputTensor = new TensorStorage<float>(n * k, clearMemory: false);
            inputData.CopyTo(inputTensor.AsSpan());
            using var input = new AutogradNode(inputTensor, new TensorShape(n, k), requiresGrad: false);

            var outA = new float[n * m];
            double bestA = double.MaxValue;
            for (var r = 0; r < reps + 1; r++)   // +1 warm-up
            {
                var sw = Stopwatch.StartNew();
                var node = graph.FrozenQuantizedLinear(input, weight);
                sw.Stop();
                node.DataView.AsReadOnlySpan().CopyTo(outA);
                graph.Reset();
                if (r > 0) { bestA = Math.Min(bestA, sw.Elapsed.TotalMilliseconds); }
            }

            // ── B: inference batched path (int8) ──
            var outB = new float[n * m];
            var aq = new sbyte[n * k];
            var asc = new float[n * (k / 256) * 1 + n * 8];     // generous
            var abs = new short[n * (k / 256) * 16 + 16];
            double bestB = double.MaxValue;
            for (var r = 0; r < reps + 1; r++)
            {
                var sw = Stopwatch.StartNew();
                Q4KDotKernel.ProjectBatched(inputData, n, weight, [], outB, aq, asc, abs);
                sw.Stop();
                if (r > 0) { bestB = Math.Min(bestB, sw.Elapsed.TotalMilliseconds); }
            }

            // Numeric divergence (activation-quant noise of the int8 path).
            double maxAbs = 0, maxRel = 0, refMax = 0;
            for (var i = 0; i < outA.Length; i++)
            {
                var d = Math.Abs(outA[i] - outB[i]);
                maxAbs = Math.Max(maxAbs, d);
                refMax = Math.Max(refMax, Math.Abs(outA[i]));
                if (Math.Abs(outA[i]) > 1f) { maxRel = Math.Max(maxRel, d / Math.Abs(outA[i])); }
            }

            _out.WriteLine($"shapes: batch {n} × [{k}→{m}], reps {reps} (best)");
            _out.WriteLine($"A training F32-dequant forward : {bestA,8:F1} ms");
            _out.WriteLine($"B inference int8 ProjectBatched: {bestB,8:F1} ms");
            _out.WriteLine($"speedup B vs A                : {bestA / bestB,8:F2}×");
            _out.WriteLine($"numeric: maxAbs {maxAbs:E2}, maxRel(|ref|>1) {maxRel:E2}, refMax {refMax:F2}");
            Assert.True(bestB > 0);
        }
    }
}
