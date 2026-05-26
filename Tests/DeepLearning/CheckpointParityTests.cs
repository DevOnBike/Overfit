// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Maths;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>
    /// Gradient checkpointing parity: running an MLP segment through <see cref="ComputationGraph.Checkpoint"/>
    /// (forward keeps only input+output, backward recomputes the activations) must produce the SAME output,
    /// parameter gradients, and input gradient as running it normally — checkpointing only trades compute
    /// for memory, never changes the math. The recompute is deterministic, so on a linear MLP chain it is
    /// bit-close.
    /// </summary>
    public sealed class CheckpointParityTests
    {
        private readonly ITestOutputHelper _out;
        public CheckpointParityTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void Checkpoint_MlpSegment_MatchesNonCheckpointed()
        {
            const int n = 8, inDim = 16, hid = 64, outDim = 10;
            var rng = new Random(123);

            var fc1 = new LinearLayer(inDim, hid);
            var fc2 = new LinearLayer(hid, outDim);
            var fc1W = fc1.Parameters().First();
            var fc2W = fc2.Parameters().First();

            // Source data as plain arrays (AutogradNode.Dispose frees its storage, so each run gets a copy).
            var xData = new float[n * inDim];
            for (var i = 0; i < xData.Length; i++) { xData[i] = (float)(rng.NextDouble() - 0.5); }
            var yData = new float[n * outDim];
            for (var b = 0; b < n; b++) { yData[b * outDim + (b % outDim)] = 1f; }

            AutogradNode Segment(ComputationGraph g, AutogradNode input)
                => fc2.Forward(g, TensorMath.ReLU(g, fc1.Forward(g, input)));

            (float[] output, float[] dX, float[] dW1, float[] dW2) Run(bool checkpointed)
            {
                fc1W.GradView.AsSpan().Clear();
                fc2W.GradView.AsSpan().Clear();

                using var graph = new ComputationGraph(1 << 20);
                var xStore = new TensorStorage<float>(n * inDim, clearMemory: false);
                xData.CopyTo(xStore.AsSpan());
                var yStore = new TensorStorage<float>(n * outDim, clearMemory: false);
                yData.CopyTo(yStore.AsSpan());

                using var x = new AutogradNode(xStore, new TensorShape(n, inDim), requiresGrad: true);
                x.GradView.AsSpan().Clear();
                using var y = new AutogradNode(yStore, new TensorShape(n, outDim));

                var output = checkpointed ? graph.Checkpoint(Segment, x) : Segment(graph, x);
                var outValues = output.DataView.AsReadOnlySpan().ToArray();

                using var loss = TensorMath.SoftmaxCrossEntropy(graph, output, y);
                graph.Backward(loss);

                return (outValues,
                    x.GradView.AsReadOnlySpan().ToArray(),
                    fc1W.GradView.AsReadOnlySpan().ToArray(),
                    fc2W.GradView.AsReadOnlySpan().ToArray());
            }

            var reference = Run(checkpointed: false);
            var ckpt = Run(checkpointed: true);

            AssertClose(reference.output, ckpt.output, "output");
            AssertClose(reference.dX, ckpt.dX, "dL/dx");
            AssertClose(reference.dW1, ckpt.dW1, "dL/dW1");
            AssertClose(reference.dW2, ckpt.dW2, "dL/dW2");
        }

        [Fact]
        public void Checkpoint_LowersMainArenaHighWaterMark()
        {
            const int n = 16, dim = 256, depth = 8;
            var rng = new Random(7);

            // A deep MLP: depth × (Linear dim→dim + ReLU). Each layer's hidden is a kept activation.
            var layers = new LinearLayer[depth];
            for (var l = 0; l < depth; l++) { layers[l] = new LinearLayer(dim, dim); }

            var xData = new float[n * dim];
            for (var i = 0; i < xData.Length; i++) { xData[i] = (float)(rng.NextDouble() - 0.5); }

            AutogradNode Segment(ComputationGraph g, AutogradNode input)
            {
                var h = input;
                for (var l = 0; l < depth; l++) { h = TensorMath.ReLU(g, layers[l].Forward(g, h)); }
                return h;
            }

            long Forward(bool checkpointed)
            {
                using var graph = new ComputationGraph(1 << 22);
                var xStore = new TensorStorage<float>(n * dim, clearMemory: false);
                xData.CopyTo(xStore.AsSpan());
                using var x = new AutogradNode(xStore, new TensorShape(n, dim), requiresGrad: true);

                _ = checkpointed ? graph.Checkpoint(Segment, x, 1 << 20) : Segment(graph, x);
                return graph.TapeBuffer.CurrentOffset;   // main-arena high-water-mark after forward
            }

            var plain = Forward(checkpointed: false);
            var ckpt = Forward(checkpointed: true);

            // Checkpointed keeps only the segment OUTPUT on the main tape (the depth internal
            // activations live in the throwaway sub-graph), so the main arena holds far less.
            Assert.True(ckpt < plain / 2,
                $"checkpoint did not lower the main arena: plain={plain} ckpt={ckpt} floats");
        }

        [Fact]
        public void Checkpoint_Gpt1Blocks_MatchNonCheckpointed_AndLowerArena()
        {
            const int b = 2, t = 8, vocab = 48;
            var config = new GPT1Config
            {
                VocabSize = vocab,
                ContextLength = 16,
                DModel = 32,
                NHeads = 4,
                NLayers = 6,
                DFF = 64,
                TieWeights = false,
                PreLayerNorm = true,
            };

            var tokenIds = new int[b * t];
            for (var i = 0; i < tokenIds.Length; i++) { tokenIds[i] = (i * 7) % vocab; }
            var targets = new float[b * t * vocab];
            for (var r = 0; r < b * t; r++) { targets[r * vocab + ((r * 3) % vocab)] = 1f; }

            (float[] logits, float[] blk0Grad, long arenaAfterForward) Run(bool checkpoint)
            {
                MathUtils.SetSeed(42);
                using var model = new GPT1Model(config, checkpointBlocks: checkpoint);
                using var graph = new ComputationGraph();

                var logits = model.Forward(graph, tokenIds, b, t);   // [b, t, vocab]
                var arena = graph.TapeBuffer.CurrentOffset;          // high-water after forward
                var logitValues = logits.DataView.AsReadOnlySpan().ToArray();

                using var yStore = new TensorStorage<float>(b * t * vocab, clearMemory: false);
                targets.CopyTo(yStore.AsSpan());
                using var y = new AutogradNode(yStore, new TensorShape(b * t, vocab));
                using var flat = TensorMath.Reshape(graph, logits, b * t, vocab);
                using var loss = TensorMath.SoftmaxCrossEntropy(graph, flat, y);
                graph.Backward(loss);

                var blk0Grad = model.Blocks[0].Parameters().First().GradView.AsReadOnlySpan().ToArray();
                return (logitValues, blk0Grad, arena);
            }

            var plain = Run(checkpoint: false);
            var ckpt = Run(checkpoint: true);

            AssertClose(plain.logits, ckpt.logits, "gpt1 logits");
            AssertClose(plain.blk0Grad, ckpt.blk0Grad, "gpt1 block0 grad");
            Assert.True(ckpt.arenaAfterForward < plain.arenaAfterForward,
                $"checkpoint did not lower the GPT-1 arena: plain={plain.arenaAfterForward} ckpt={ckpt.arenaAfterForward}");
        }

        [Fact]
        public void CheckpointedModule_InSequential_MatchesPlain()
        {
            const int n = 8, inDim = 16, hid = 48, outDim = 10;
            var xData = new float[n * inDim];
            var rng = new Random(5);
            for (var i = 0; i < xData.Length; i++) { xData[i] = (float)(rng.NextDouble() - 0.5); }
            var yData = new float[n * outDim];
            for (var b = 0; b < n; b++) { yData[b * outDim + (b % outDim)] = 1f; }

            // Two Sequentials with identical init (same seed → same weight draws, same construction order);
            // the second wraps its final Linear in a CheckpointedModule.
            MathUtils.SetSeed(77);
            using var plain = new Sequential(new LinearLayer(inDim, hid), new ReluActivation(), new LinearLayer(hid, outDim));
            MathUtils.SetSeed(77);
            using var ckpt = new Sequential(new LinearLayer(inDim, hid), new ReluActivation(),
                new CheckpointedModule(new LinearLayer(hid, outDim)));

            var (pOut, pGrads) = RunSequential(plain, xData, yData, n, inDim, outDim);
            var (cOut, cGrads) = RunSequential(ckpt, xData, yData, n, inDim, outDim);

            AssertClose(pOut, cOut, "seq output");
            Assert.Equal(pGrads.Count, cGrads.Count);
            for (var i = 0; i < pGrads.Count; i++)
            {
                AssertClose(pGrads[i], cGrads[i], $"seq param[{i}] grad");
            }
        }

        private static (float[] output, List<float[]> paramGrads) RunSequential(
            Sequential model, float[] xData, float[] yData, int n, int inDim, int outDim)
        {
            using var graph = new ComputationGraph(1 << 20);
            var xStore = new TensorStorage<float>(n * inDim, clearMemory: false);
            xData.CopyTo(xStore.AsSpan());
            var yStore = new TensorStorage<float>(n * outDim, clearMemory: false);
            yData.CopyTo(yStore.AsSpan());

            using var x = new AutogradNode(xStore, new TensorShape(n, inDim), requiresGrad: true);
            using var y = new AutogradNode(yStore, new TensorShape(n, outDim));

            var output = model.Forward(graph, x);
            var outValues = output.DataView.AsReadOnlySpan().ToArray();

            using var loss = TensorMath.SoftmaxCrossEntropy(graph, output, y);
            graph.Backward(loss);

            var grads = model.Parameters().Select(p => p.GradView.AsReadOnlySpan().ToArray()).ToList();
            return (outValues, grads);
        }

        [LongFact]
        public void Checkpoint_Gpt1_MemorySavings_LargerModel()
        {
            // A larger GPT-1 (≈ GPT-2-small-ish dims, modest batch/seq) where activations dominate.
            const int b = 2, t = 128, vocab = 256;
            var config = new GPT1Config
            {
                VocabSize = vocab,
                ContextLength = 256,
                DModel = 512,
                NHeads = 8,
                NLayers = 12,
                DFF = 2048,
                TieWeights = false,
                PreLayerNorm = true,
            };

            var tokenIds = new int[b * t];
            for (var i = 0; i < tokenIds.Length; i++) { tokenIds[i] = (i * 7) % vocab; }

            // Read the live-activation high-water (arena CurrentOffset) after a forward pass.
            long ArenaUsedFloats(bool checkpoint)
            {
                MathUtils.SetSeed(1);
                using var model = new GPT1Model(config, checkpointBlocks: checkpoint);
                using var graph = new ComputationGraph(1 << 27);   // 512 MB arena — fits the plain run
                _ = model.Forward(graph, tokenIds, b, t);
                return graph.TapeBuffer.CurrentOffset;
            }

            var plain = ArenaUsedFloats(checkpoint: false);
            var ckpt = ArenaUsedFloats(checkpoint: true);

            var plainMb = plain * 4.0 / (1024 * 1024);
            var ckptMb = ckpt * 4.0 / (1024 * 1024);
            _out.WriteLine(
                $"GPT-1 {config.NLayers}L d={config.DModel} dFF={config.DFF} b={b} t={t}: " +
                $"plain arena {plainMb:F1} MB → checkpointed {ckptMb:F1} MB " +
                $"({plainMb / ckptMb:F1}× less, saved {plainMb - ckptMb:F1} MB)");

            Assert.True(ckpt * 3 < plain, $"expected ≥3× activation-memory cut: plain={plain} ckpt={ckpt}");
        }

        private static void AssertClose(float[] a, float[] b, string what)
        {
            Assert.Equal(a.Length, b.Length);
            for (var i = 0; i < a.Length; i++)
            {
                Assert.True(MathF.Abs(a[i] - b[i]) < 1e-5f, $"{what}[{i}] {a[i]} vs {b[i]}");
            }
        }
    }
}
