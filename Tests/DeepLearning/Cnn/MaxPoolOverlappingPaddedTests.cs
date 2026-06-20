// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests.DeepLearning.Cnn
{
    /// <summary>
    /// Numerical validation of the overlapping / padded MaxPool inference path added so the ONNX importer can
    /// load real CNNs (ResNet's first pool is 3x3 stride-2 pad-1 — overlapping, which the old non-overlapping
    /// kernel rejected). Hand-computed expected maxima; ONNX pads MaxPool with -inf, so padded cells never win.
    /// </summary>
    public sealed class MaxPoolOverlappingPaddedTests
    {
        [Fact]
        public void Overlapping3x3Stride2Pad1_MatchesHandComputedMaxima()
        {
            // 1 channel, 5x5 input filled 1..25 row-major.
            var input = new float[25];
            for (var i = 0; i < 25; i++)
            {
                input[i] = i + 1;
            }

            // ResNet-style pool: kernel 3, stride 2, padding 1 → outH = outW = (5 + 2 - 3)/2 + 1 = 3.
            var layer = new MaxPool2DLayer(channels: 1, inputH: 5, inputW: 5, poolSize: 3, stride: 2, padding: 1);
            Assert.Equal(3, layer.OutputH);
            Assert.Equal(3, layer.OutputW);

            var output = new float[layer.InferenceOutputSize];
            layer.ForwardInference(input, output);

            // Windows (padded edges = -inf, ignored): each output is the max over its valid 3x3 window.
            float[] expected = [7, 9, 10, 17, 19, 20, 22, 24, 25];
            Assert.Equal(expected, output);
        }

        [Fact]
        public void NonOverlapping_StillMatchesSimplePath()
        {
            // Back-compat: stride defaults to poolSize, no padding → the classic non-overlapping pool.
            var input = new float[16];
            for (var i = 0; i < 16; i++)
            {
                input[i] = i + 1;
            }

            var layer = new MaxPool2DLayer(channels: 1, inputH: 4, inputW: 4, poolSize: 2);
            var output = new float[layer.InferenceOutputSize];
            layer.ForwardInference(input, output);

            // 4x4 → 2x2, max of each 2x2 tile of 1..16.
            float[] expected = [6, 8, 14, 16];
            Assert.Equal(expected, output);
        }
    }
}
