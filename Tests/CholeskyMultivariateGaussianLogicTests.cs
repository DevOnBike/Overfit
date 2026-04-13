// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Statistical;

namespace DevOnBike.Overfit.Tests
{
    public class CholeskyMultivariateGaussianLogicTests
    {
        private const int Precision = 5; // Dokładność dla porównań zmiennoprzecinkowych

        [Fact]
        public void ValidateInputs_ValidInputs_DoesNotThrow()
        {
            // Arrange
            var mean = new[] { 1.0f, 2.0f };
            using var covariance = new FastTensor<float>(2, 2, clearMemory: true);

            // Act & Assert
            var ex = Record.Exception(() => CholeskyMultivariateGaussianLogic.ValidateInputs(mean, covariance.GetView()));
            Assert.Null(ex);
        }

        [Fact]
        public void ValidateInputs_EmptyMean_ThrowsArgumentException()
        {
            // Arrange
            var mean = Array.Empty<float>();
            using var covariance = new FastTensor<float>(2, 2, clearMemory: true);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => CholeskyMultivariateGaussianLogic.ValidateInputs(mean, covariance.GetView()));
            Assert.Contains("must be 0x0", ex.Message);
        }

        [Fact]
        public void LogProbabilityDensity_CorrectDistance_ReturnsExpectedLogProb()
        {
            // Arrange
            const int dimensions = 2;
            var observation = new[] { 0.5f, 0.5f };
            var mean = new[] { 0.5f, 0.5f }; // Obserwacja == Średnia -> odległość Mahalanobisa = 0

            using var L = new FastTensor<float>(dimensions, dimensions, clearMemory: true);
            var lView = L.GetView();
            lView[0, 0] = 1.0f;
            lView[1, 1] = 1.0f; // Macierz identycznościowa

            var logNormConst = CholeskyMultivariateGaussianLogic.CalculateLogNormConstant(dimensions, lView);

            // Skoro distance = 0, LogProbabilityDensity powinno być równe logNormConst
            var expectedLogProb = logNormConst - 0.0;

            // Act
            var result = CholeskyMultivariateGaussianLogic.LogProbabilityDensity(observation, mean, lView, logNormConst);

            // Assert
            Assert.Equal(expectedLogProb, result, Precision);
        }

        [Fact]
        public void LogProbabilityDensity_DimensionMismatch_ThrowsArgumentException()
        {
            // Arrange
            var observation = new[] { 1.0f, 2.0f, 3.0f }; // Długość 3
            var mean = new[] { 1.0f, 2.0f }; // Długość 2
            using var L = new FastTensor<float>(2, 2, clearMemory: true);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                CholeskyMultivariateGaussianLogic.LogProbabilityDensity(observation, mean, L.GetView(), 1.0));
        }
    }
}