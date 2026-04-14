// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.

using System;
using Xunit;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public class GradientChecksTests
    {
        // ====================================================================
        // 1. LINEAR LAYER (Fully Connected)
        // ====================================================================
        [Fact]
        public void LinearLayer_GradientCheck_ShouldPass()
        {
            int batchSize = 3;
            int inFeatures = 4;
            int outFeatures = 2;

            var layer = new LinearLayer(inFeatures, outFeatures);
            layer.Train();

            var input = new AutogradNode(new FastTensor<float>(batchSize, inFeatures).Randomize(1.0f), requiresGrad: true);
            var target = new AutogradNode(new FastTensor<float>(batchSize, outFeatures).Randomize(1.0f), requiresGrad: false);

            AutogradNode ForwardAndLoss(ComputationGraph g)
            {
                var pred = layer.Forward(g, input);
                return TensorMath.MSELoss(g, pred, target);
            }

            GradientChecker.Verify(layer, ForwardAndLoss, epsilon: 1e-3f, tolerance: 1e-2f);
        }

        // ====================================================================
        // 2. CONVOLUTIONAL 2D LAYER
        // ====================================================================
        [Fact]
        public void ConvLayer_GradientCheck_ShouldPass()
        {
            // Trzymamy małe wymiary, żeby test wykonał się w ułamku sekundy
            int batchSize = 2;
            int inChannels = 2;
            int outChannels = 3;
            int h = 4, w = 4;
            int kSize = 3;
            int outH = h - kSize + 1;
            int outW = w - kSize + 1;

            var layer = new ConvLayer(inChannels, outChannels, h, w, kSize);
            layer.Train();

            var input = new AutogradNode(new FastTensor<float>(batchSize, inChannels, h, w).Randomize(1.0f), requiresGrad: true);
            var target = new AutogradNode(new FastTensor<float>(batchSize, outChannels, outH, outW).Randomize(1.0f), requiresGrad: false);

            AutogradNode ForwardAndLoss(ComputationGraph g)
            {
                var pred = layer.Forward(g, input);
                return TensorMath.MSELoss(g, pred, target);
            }

            GradientChecker.Verify(layer, ForwardAndLoss, epsilon: 1e-3f, tolerance: 1e-2f);
        }

        // ====================================================================
        // 3. LSTM CELL (Weryfikacja bramek i skomplikowanych pochodnych)
        // ====================================================================
        [Fact]
        public void LSTMCell_GradientCheck_ShouldPass()
        {
            int batchSize = 2;
            int inputSize = 3;
            int hiddenSize = 4;

            var cell = new LSTMCell(inputSize, hiddenSize);
            cell.Train();

            var x = new AutogradNode(new FastTensor<float>(batchSize, inputSize).Randomize(1.0f), requiresGrad: true);
            var hPrev = new AutogradNode(new FastTensor<float>(batchSize, hiddenSize).Randomize(1.0f), requiresGrad: true);
            var cPrev = new AutogradNode(new FastTensor<float>(batchSize, hiddenSize).Randomize(1.0f), requiresGrad: true);

            var targetH = new AutogradNode(new FastTensor<float>(batchSize, hiddenSize).Randomize(1.0f), requiresGrad: false);

            AutogradNode ForwardAndLoss(ComputationGraph g)
            {
                // Zauważ elastyczność: LSTMCell bierze 3 parametry wejściowe, 
                // a nasz delegat to bez problemu obsługuje!
                var (hNext, cNext) = cell.Forward(g, x, hPrev, cPrev);

                // Opieramy stratę tylko o `hNext`. Backward poleci przez wszystkie bramki (f, i, g, o)
                return TensorMath.MSELoss(g, hNext, targetH);
            }

            // LSTM ma wewnątrz Sigmoid i Tanh, co kumuluje błędy zmiennoprzecinkowe (float32).
            // Ustawiamy tolerancję na 2e-2f co jest standardem dla takich testów float32.
            GradientChecker.Verify(cell, ForwardAndLoss, epsilon: 1e-3f, tolerance: 2e-2f);
        }

        // ====================================================================
        // 4. RESIDUAL BLOCK (Integracja Conv, BatchNorm i Add)
        // ====================================================================
        [Fact]
        public void ResidualBlock_GradientCheck_ShouldPass()
        {
            int batchSize = 2;
            int hiddenSize = 4; // To musi być spłaszczone 1D, bo Twój ResidualBlock używa LinearLayer

            var block = new ResidualBlock(hiddenSize);
            block.Train();

            var input = new AutogradNode(new FastTensor<float>(batchSize, hiddenSize).Randomize(1.0f), requiresGrad: true);
            var target = new AutogradNode(new FastTensor<float>(batchSize, hiddenSize).Randomize(1.0f), requiresGrad: false);

            AutogradNode ForwardAndLoss(ComputationGraph g)
            {
                var pred = block.Forward(g, input);
                return TensorMath.MSELoss(g, pred, target);
            }

            GradientChecker.Verify(block, ForwardAndLoss, epsilon: 1e-3f, tolerance: 2e-2f);
        }

        // ====================================================================
        // 5. BATCH NORM 1D (Pozostawiony dla kompletności pliku)
        // ====================================================================
        [Fact]
        public void BatchNorm1D_GradientCheck_ShouldPass()
        {
            int batchSize = 4;
            int features = 5;

            var bn = new BatchNorm1D(features);
            bn.Gamma.DataView.AsSpan().Fill(1.2f);
            bn.Beta.DataView.AsSpan().Fill(0.3f);
            bn.Train();

            var input = new AutogradNode(new FastTensor<float>(batchSize, features).Randomize(1.0f), requiresGrad: true);
            var target = new AutogradNode(new FastTensor<float>(batchSize, features).Randomize(1.0f), requiresGrad: false);

            AutogradNode ForwardAndLoss(ComputationGraph g)
            {
                var pred = bn.Forward(g, input);
                return TensorMath.MSELoss(g, pred, target);
            }

            GradientChecker.Verify(bn, ForwardAndLoss, epsilon: 1e-3f, tolerance: 1e-2f);
        }
    }
}