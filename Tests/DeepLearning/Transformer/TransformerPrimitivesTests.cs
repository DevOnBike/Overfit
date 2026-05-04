// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// Tests for LayerNormLayer and EmbeddingLayer —
    /// building blocks of the Transformer / GPT-1 architecture.
    /// </summary>
    public class TransformerPrimitivesTests
    {
        private const float Tolerance = 1e-4f;

        // ─────────────────────────────────────────────────────────────────────
        // LayerNorm — forward
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void LayerNorm_Forward_OutputHasZeroMeanAndUnitVariance_BeforeScaleShift()
        {
            // With gamma=1, beta=0, each row of output should have ~zero mean
            // and ~unit variance (epsilon causes tiny deviation).
            using var ln = new LayerNormLayer(normalizedShape: 4);
            using var graph = new ComputationGraph();

            var storage = new TensorStorage<float>(8, clearMemory: false);
            // Two rows: [1, 2, 3, 4] and [10, 20, 30, 40]
            storage.AsSpan()[0] = 1f; storage.AsSpan()[1] = 2f;
            storage.AsSpan()[2] = 3f; storage.AsSpan()[3] = 4f;
            storage.AsSpan()[4] = 10f; storage.AsSpan()[5] = 20f;
            storage.AsSpan()[6] = 30f; storage.AsSpan()[7] = 40f;

            using var input = new AutogradNode(storage, new TensorShape(2, 4), requiresGrad: false);
            using var output = ln.Forward(graph, input);

            var outS = output.DataView.AsReadOnlySpan();

            // Check row 0: mean ≈ 0, variance ≈ 1
            var mean0 = (outS[0] + outS[1] + outS[2] + outS[3]) / 4f;
            Assert.True(MathF.Abs(mean0) < Tolerance, $"Row 0 mean should be ~0, got {mean0}");

            var var0 = (outS[0] * outS[0] + outS[1] * outS[1] + outS[2] * outS[2] + outS[3] * outS[3]) / 4f;
            Assert.True(MathF.Abs(var0 - 1f) < 0.01f, $"Row 0 variance should be ~1, got {var0}");

            // Check row 1: same invariant (scale-invariant property of LayerNorm)
            var mean1 = (outS[4] + outS[5] + outS[6] + outS[7]) / 4f;
            Assert.True(MathF.Abs(mean1) < Tolerance, $"Row 1 mean should be ~0, got {mean1}");
        }

        [Fact]
        public void LayerNorm_Forward_WithScaleShift_MatchesManualComputation()
        {
            // gamma=[2,2,2,2], beta=[1,1,1,1]: output = 2*normalised + 1
            using var ln = new LayerNormLayer(normalizedShape: 4);

            ln.Gamma.DataSpan.Fill(2f);
            ln.Beta.DataSpan.Fill(1f);

            using var graph = new ComputationGraph();
            var storage = new TensorStorage<float>(4, clearMemory: false);
            storage.AsSpan()[0] = 1f;
            storage.AsSpan()[1] = 2f;
            storage.AsSpan()[2] = 3f;
            storage.AsSpan()[3] = 4f;

            using var input = new AutogradNode(storage, new TensorShape(1, 4), requiresGrad: false);
            using var output = ln.Forward(graph, input);

            var outS = output.DataView.AsReadOnlySpan();

            // Manually: mean=2.5, var=1.25, invStd=1/sqrt(1.25+1e-5)≈0.8944
            var mu = 2.5f;
            var invStd = 1f / MathF.Sqrt(1.25f + 1e-5f);

            for (var i = 0; i < 4; i++)
            {
                var expected = 2f * ((i + 1 - mu) * invStd) + 1f;
                var diff = MathF.Abs(outS[i] - expected);
                Assert.True(diff < Tolerance, $"output[{i}] = {outS[i]}, expected {expected}");
            }
        }

        [Fact]
        public void LayerNorm_Backward_GradientCheckNumerical()
        {
            // Numerical gradient check for LayerNormLayer.
            // Loss = sum(layernorm(x)) — scalar, seed gradient = 1.0 per element.
            using var ln = new LayerNormLayer(normalizedShape: 4);
            ln.InvalidateParameterCaches(); // ensure fresh cached nodes

            float[] inputData = [0.5f, -0.3f, 1.2f, -0.8f, 1.0f, 0.7f, -0.5f, 0.2f];

            float ForwardSumLoss(float[] data)
            {
                var storage = new TensorStorage<float>(8, clearMemory: false);
                data.AsSpan().CopyTo(storage.AsSpan());
                using var g = new ComputationGraph();
                using var input = new AutogradNode(storage, new TensorShape(2, 4), requiresGrad: false);
                using var output = ln.Forward(g, input);
                var sum = 0f;
                foreach (var v in output.DataView.AsReadOnlySpan())
                {
                    sum += v;
                }
                return sum;
            }

            var eps = 1e-3f;

            // Numerical: perturb gamma[0]
            ln.Gamma.DataSpan[0] += eps;
            ln.InvalidateParameterCaches();
            var lossPlus = ForwardSumLoss(inputData);

            ln.Gamma.DataSpan[0] -= 2 * eps;
            ln.InvalidateParameterCaches();
            var lossMinus = ForwardSumLoss(inputData);

            ln.Gamma.DataSpan[0] += eps;
            ln.InvalidateParameterCaches();

            var numerical = (lossPlus - lossMinus) / (2 * eps);

            // Analytical: backward with seed = 1 per element (sum loss)
            ln.Gamma.ZeroGrad();
            ln.Beta.ZeroGrad();

            var storage2 = new TensorStorage<float>(8, clearMemory: false);
            inputData.AsSpan().CopyTo(storage2.AsSpan());
            using var gBack = new ComputationGraph();
            using var input2 = new AutogradNode(storage2, new TensorShape(2, 4), requiresGrad: false);
            using var output2 = ln.Forward(gBack, input2);

            // Seed gradient = 1.0 per element (dLoss/dOutput for sum loss)
            output2.GradView.AsSpan().Fill(1f);
            gBack.Backward(output2);

            var analytical = ln.Gamma.GradSpan[0];
            var relErr = MathF.Abs(analytical - numerical) / (MathF.Abs(numerical) + 1e-7f);

            Assert.True(relErr < 0.03f,
                $"Gradient check failed: analytical={analytical:F6}, numerical={numerical:F6}, relErr={relErr:F4}");
        }

        [Fact]
        public void LayerNorm_ForwardInference_MatchesForwardWithGraph()
        {
            using var ln = new LayerNormLayer(normalizedShape: 4);
            ln.Gamma.DataSpan[0] = 1.5f;
            ln.Gamma.DataSpan[2] = 0.8f;
            ln.Beta.DataSpan[1] = 0.3f;

            float[] input = [0.5f, -0.3f, 1.2f, -0.8f];

            // Graph forward
            using var g = new ComputationGraph();
            var storage = new TensorStorage<float>(4, clearMemory: false);
            input.AsSpan().CopyTo(storage.AsSpan());
            using var node = new AutogradNode(storage, new TensorShape(1, 4), requiresGrad: false);
            using var outNode = ln.Forward(g, node);
            var graphOut = outNode.DataView.AsReadOnlySpan().ToArray();

            // Inference forward
            var infOut = new float[4];
            ln.ForwardInference(input, infOut);

            for (var i = 0; i < 4; i++)
            {
                var diff = MathF.Abs(graphOut[i] - infOut[i]);
                Assert.True(diff < Tolerance, $"[{i}] graph={graphOut[i]:F6}, inference={infOut[i]:F6}");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Embedding — forward + backward
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Embedding_Forward_ReturnsCorrectRows()
        {
            using var emb = new EmbeddingLayer(vocabSize: 5, embeddingDim: 3);

            // Set known weights
            var w = emb.Weight.DataSpan;
            for (var i = 0; i < 15; i++)
            {
                w[i] = i;
            }
            // Row 0: [0,1,2], Row 1: [3,4,5], Row 2: [6,7,8] ...

            using var graph = new ComputationGraph();
            int[] tokenIds = [2, 0, 4];
            using var output = emb.Forward(graph, tokenIds);

            var outS = output.DataView.AsReadOnlySpan();

            // token 2 → row 2 → [6,7,8]
            Assert.Equal(6f, outS[0]);
            Assert.Equal(7f, outS[1]);
            Assert.Equal(8f, outS[2]);

            // token 0 → row 0 → [0,1,2]
            Assert.Equal(0f, outS[3]);
            Assert.Equal(1f, outS[4]);
            Assert.Equal(2f, outS[5]);

            // token 4 → row 4 → [12,13,14]
            Assert.Equal(12f, outS[6]);
            Assert.Equal(13f, outS[7]);
            Assert.Equal(14f, outS[8]);
        }

        [Fact]
        public void Embedding_Backward_ScatterAddsToCorrectRows()
        {
            using var emb = new EmbeddingLayer(vocabSize: 4, embeddingDim: 2);
            emb.Weight.ZeroGrad();

            int[] tokenIds = [1, 3, 1]; // token 1 appears twice → grad accumulates

            using var graph = new ComputationGraph();
            using var output = emb.Forward(graph, tokenIds);

            // dL/dOutput = all ones
            output.GradView.AsSpan().Fill(1f);
            graph.Backward(output);

            var grad = emb.Weight.GradSpan;

            // Token 0: not accessed → grad = [0, 0]
            Assert.Equal(0f, grad[0]);
            Assert.Equal(0f, grad[1]);

            // Token 1: accessed twice → grad = [2, 2]
            Assert.Equal(2f, grad[2]);
            Assert.Equal(2f, grad[3]);

            // Token 2: not accessed → grad = [0, 0]
            Assert.Equal(0f, grad[4]);
            Assert.Equal(0f, grad[5]);

            // Token 3: accessed once → grad = [1, 1]
            Assert.Equal(1f, grad[6]);
            Assert.Equal(1f, grad[7]);
        }

        [Fact]
        public void Embedding_OutOfRangeTokenId_Throws()
        {
            using var emb = new EmbeddingLayer(vocabSize: 10, embeddingDim: 4);
            using var graph = new ComputationGraph();

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                emb.Forward(graph, [0, 5, 10])); // id=10 is out of range [0,10)
        }

        [Fact]
        public void Embedding_LookupInference_ReturnsCorrectRow()
        {
            using var emb = new EmbeddingLayer(vocabSize: 3, embeddingDim: 4);
            var w = emb.Weight.DataSpan;
            for (var i = 0; i < 12; i++)
            {
                w[i] = i * 0.1f;
            }

            var result = new float[4];
            emb.LookupInference(2, result);

            // Row 2: [0.8, 0.9, 1.0, 1.1]
            Assert.True(MathF.Abs(result[0] - 0.8f) < Tolerance);
            Assert.True(MathF.Abs(result[1] - 0.9f) < Tolerance);
            Assert.True(MathF.Abs(result[2] - 1.0f) < Tolerance);
            Assert.True(MathF.Abs(result[3] - 1.1f) < Tolerance);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Integration: Embedding → LayerNorm (minimal Transformer prefix)
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void EmbeddingThenLayerNorm_RunsWithoutException()
        {
            var vocabSize = 50;
            var embeddingDim = 8;
            var seqLen = 4;

            using var emb = new EmbeddingLayer(vocabSize, embeddingDim);
            using var ln = new LayerNormLayer(embeddingDim);
            using var graph = new ComputationGraph();

            int[] tokenIds = [3, 12, 7, 0];

            using var embOut = emb.Forward(graph, tokenIds);  // [4, 8]
            using var lnOut = ln.Forward(graph, embOut);     // [4, 8]

            Assert.Equal(seqLen, lnOut.Shape.D0);
            Assert.Equal(embeddingDim, lnOut.Shape.D1);

            // All values should be finite
            foreach (var v in lnOut.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v), "NaN in LayerNorm output after Embedding");
                Assert.False(float.IsInfinity(v), "Inf in LayerNorm output after Embedding");
            }
        }
    }
}