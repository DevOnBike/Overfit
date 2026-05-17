// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Kernels
{
    public class SingleTokenLayerNormKernelTests
    {
        [Fact]
        public void NormalizeWithoutAffine_NormalizesToZeroMeanAndUnitVariance()
        {
            var input = new[] { 1f, 2f, 3f, 4f };
            var output = new float[4];

            SingleTokenLayerNormKernel.NormalizeWithoutAffine(
                input,
                output,
                size: 4,
                epsilon: 1e-5f);

            var mean = output.Average();
            var variance = output
                .Select(x => (x - mean) * (x - mean))
                .Average();

            AssertClose(0f, mean, tolerance: 1e-5f);
            AssertClose(1f, variance, tolerance: 1e-4f);
        }

        [Fact]
        public void Normalize_WithGammaAndBeta_AppliesAffineTransform()
        {
            var input = new[] { 1f, 2f };
            var gamma = new[] { 2f, 3f };
            var beta = new[] { 10f, 20f };
            var output = new float[2];

            SingleTokenLayerNormKernel.Normalize(
                input,
                gamma,
                beta,
                output,
                size: 2,
                epsilon: 1e-5f);

            var mean = 1.5f;
            var variance = 0.25f;
            var invStd = 1f / MathF.Sqrt(variance + 1e-5f);

            var expected0 = ((1f - mean) * invStd * 2f) + 10f;
            var expected1 = ((2f - mean) * invStd * 3f) + 20f;

            AssertClose(expected0, output[0]);
            AssertClose(expected1, output[1]);
        }

        [Fact]
        public void Normalize_WithGammaOnly_AppliesScale()
        {
            var input = new[] { 1f, 2f };
            var gamma = new[] { 2f, 3f };
            var output = new float[2];

            SingleTokenLayerNormKernel.Normalize(
                input,
                gamma,
                beta: [],
                output,
                size: 2,
                epsilon: 1e-5f);

            var mean = 1.5f;
            var variance = 0.25f;
            var invStd = 1f / MathF.Sqrt(variance + 1e-5f);

            AssertClose((1f - mean) * invStd * 2f, output[0]);
            AssertClose((2f - mean) * invStd * 3f, output[1]);
        }

        [Fact]
        public void Normalize_WithBetaOnly_AppliesShift()
        {
            var input = new[] { 1f, 2f };
            var beta = new[] { 10f, 20f };
            var output = new float[2];

            SingleTokenLayerNormKernel.Normalize(
                input,
                gamma: [],
                beta,
                output,
                size: 2,
                epsilon: 1e-5f);

            var mean = 1.5f;
            var variance = 0.25f;
            var invStd = 1f / MathF.Sqrt(variance + 1e-5f);

            AssertClose(((1f - mean) * invStd) + 10f, output[0]);
            AssertClose(((2f - mean) * invStd) + 20f, output[1]);
        }

        [Fact]
        public void Normalize_ConstantInput_ReturnsBetaWhenAffineProvided()
        {
            var input = new[] { 5f, 5f, 5f };
            var gamma = new[] { 2f, 2f, 2f };
            var beta = new[] { 1f, 2f, 3f };
            var output = new float[3];

            SingleTokenLayerNormKernel.Normalize(
                input,
                gamma,
                beta,
                output,
                size: 3,
                epsilon: 1e-5f);

            Assert.Equal(new[] { 1f, 2f, 3f }, output);
        }

        [Fact]
        public void AddResidual_AddsTwoVectors()
        {
            var residual = new[] { 1f, 2f, 3f };
            var update = new[] { 10f, 20f, 30f };
            var output = new float[3];

            SingleTokenLayerNormKernel.AddResidual(
                residual,
                update,
                output,
                size: 3);

            Assert.Equal(new[] { 11f, 22f, 33f }, output);
        }

        [Fact]
        public void AddResidualInPlace_AddsUpdateToDestination()
        {
            var destination = new[] { 1f, 2f, 3f };
            var update = new[] { 10f, 20f, 30f };

            SingleTokenLayerNormKernel.AddResidualInPlace(
                destination,
                update,
                size: 3);

            Assert.Equal(new[] { 11f, 22f, 33f }, destination);
        }

        [Fact]
        public void Normalize_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                SingleTokenLayerNormKernel.NormalizeWithoutAffine(
                    input: new float[1],
                    output: new float[1],
                    size: 0,
                    epsilon: 1e-5f));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                SingleTokenLayerNormKernel.NormalizeWithoutAffine(
                    input: new float[1],
                    output: new float[1],
                    size: 1,
                    epsilon: 0f));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenLayerNormKernel.NormalizeWithoutAffine(
                    input: new float[0],
                    output: new float[1],
                    size: 1,
                    epsilon: 1e-5f));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenLayerNormKernel.Normalize(
                    input: new float[1],
                    gamma: new float[0],
                    beta: new float[1],
                    output: new float[1],
                    size: 2,
                    epsilon: 1e-5f));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenLayerNormKernel.Normalize(
                    input: new float[2],
                    gamma: new float[1],
                    beta: new float[2],
                    output: new float[2],
                    size: 2,
                    epsilon: 1e-5f));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenLayerNormKernel.Normalize(
                    input: new float[2],
                    gamma: new float[2],
                    beta: new float[1],
                    output: new float[2],
                    size: 2,
                    epsilon: 1e-5f));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenLayerNormKernel.Normalize(
                    input: new float[2],
                    gamma: new float[2],
                    beta: new float[2],
                    output: new float[1],
                    size: 2,
                    epsilon: 1e-5f));
        }

        [Fact]
        public void AddResidual_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                SingleTokenLayerNormKernel.AddResidual(
                    residual: new float[1],
                    update: new float[1],
                    output: new float[1],
                    size: 0));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenLayerNormKernel.AddResidual(
                    residual: new float[0],
                    update: new float[1],
                    output: new float[1],
                    size: 1));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenLayerNormKernel.AddResidual(
                    residual: new float[1],
                    update: new float[0],
                    output: new float[1],
                    size: 1));

            Assert.Throws<ArgumentException>(() =>
                SingleTokenLayerNormKernel.AddResidual(
                    residual: new float[1],
                    update: new float[1],
                    output: new float[0],
                    size: 1));
        }

        private static void AssertClose(
            float expected,
            float actual,
            float tolerance = 1e-5f)
        {
            Assert.True(
                MathF.Abs(expected - actual) <= tolerance,
                $"Expected {expected}, actual {actual}, tolerance {tolerance}.");
        }
    }
}
