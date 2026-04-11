// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using Xunit;
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
            var mean = new[]
            {
                1.0f, 2.0f
            };
            using var covariance = new FastMatrix<float>(2, 2);

            // Act & Assert
            var ex = Record.Exception(() => CholeskyMultivariateGaussianLogic.ValidateInputs(mean, covariance));
            Assert.Null(ex);
        }

        [Fact]
        public void ValidateInputs_EmptyMean_ThrowsArgumentException()
        {
            // Arrange
            var mean = Array.Empty<float>();
            using var covariance = new FastMatrix<float>(2, 2);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => CholeskyMultivariateGaussianLogic.ValidateInputs(mean, covariance));
            Assert.Contains("empty", ex.Message);
        }

        [Fact]
        public void ValidateInputs_DimensionMismatch_ThrowsArgumentException()
        {
            // Arrange
            var mean = new[]
            {
                1.0f, 2.0f
            };
            using var covariance = new FastMatrix<float>(3, 3); // Mismatch!

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => CholeskyMultivariateGaussianLogic.ValidateInputs(mean, covariance));
            Assert.Contains("must be 2x2", ex.Message);
        }

        [Fact]
        public void DecomposeCholesky_ValidPositiveDefiniteMatrix_ReturnsCorrectL()
        {
            // Arrange
            using var covariance = new FastMatrix<float>(3, 3);
            covariance[0, 0] = 4f;
            covariance[0, 1] = 12f;
            covariance[0, 2] = -16f;
            covariance[1, 0] = 12f;
            covariance[1, 1] = 37f;
            covariance[1, 2] = -43f;
            covariance[2, 0] = -16f;
            covariance[2, 1] = -43f;
            covariance[2, 2] = 98f;

            // Expected L:
            // [  2,   0,   0 ]
            // [  6,   1,   0 ]
            // [ -8,   5,   3 ]

            // Act
            using var L = CholeskyMultivariateGaussianLogic.DecomposeCholesky(covariance);

            // Assert
            Assert.Equal(2.0f, L[0, 0], Precision);
            Assert.Equal(0.0f, L[0, 1]);
            Assert.Equal(0.0f, L[0, 2]);
            Assert.Equal(6.0f, L[1, 0], Precision);
            Assert.Equal(1.0f, L[1, 1], Precision);
            Assert.Equal(0.0f, L[1, 2]);
            Assert.Equal(-8.0f, L[2, 0], Precision);
            Assert.Equal(5.0f, L[2, 1], Precision);
            Assert.Equal(3.0f, L[2, 2], Precision);
        }

        [Fact]
        public void DecomposeCholesky_NotPositiveDefinite_ThrowsArgumentExceptionAndCleansUp()
        {
            // Arrange
            using var covariance = new FastMatrix<float>(2, 2);
            covariance[0, 0] = 4f;
            covariance[0, 1] = 12f;
            covariance[1, 0] = 12f;
            covariance[1, 1] = 9f; // Sprawia, że macierz nie jest dodatnio określona

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() => {
                // Chociaż metoda wewnątrz rzuca wyjątek, powinna poprawnie wywołać L.Dispose(),
                // co zabezpiecza naszą pulę ArrayPool przed wyciekiem pamięci.
                _ = CholeskyMultivariateGaussianLogic.DecomposeCholesky(covariance);
            });

            Assert.Contains("not positive-definite", ex.Message);
        }

        [Fact]
        public void CalculateLogNormConstant_ValidIdentityMatrix_ReturnsCorrectConstant()
        {
            // Arrange
            var dimensions = 2;
            using var L = new FastMatrix<float>(dimensions, dimensions);
            L[0, 0] = 1.0f;
            L[0, 1] = 0.0f;
            L[1, 0] = 0.0f;
            L[1, 1] = 1.0f;

            // Constant dla macierzy identycznościowej: -0.5 * 2 * ln(2PI) - 0
            var expectedConstant = -1.0 * Math.Log(2.0 * Math.PI);

            // Act
            var result = CholeskyMultivariateGaussianLogic.CalculateLogNormConstant(dimensions, L);

            // Assert - tutaj wciąż oczekujemy double, bo logika wylicza stałą w double'u.
            Assert.Equal(expectedConstant, result, Precision);
        }

        [Fact]
        public void LogProbabilityDensity_ValidInputs_ReturnsCorrectDensity()
        {
            // Arrange
            var dimensions = 2;
            var observation = new[]
            {
                0.5f, 0.5f
            };
            var mean = new[]
            {
                0.5f, 0.5f
            }; // Obserwacja == Średnia -> odległość Mahalanobisa = 0

            using var L = new FastMatrix<float>(dimensions, dimensions);
            L[0, 0] = 1.0f;
            L[1, 1] = 1.0f; // Macierz identycznościowa

            var logNormConst = CholeskyMultivariateGaussianLogic.CalculateLogNormConstant(dimensions, L);

            // Skoro distance = 0, LogProbabilityDensity powinno być równe logNormConst
            var expectedLogProb = logNormConst - 0.0;

            // Act
            var result = CholeskyMultivariateGaussianLogic.LogProbabilityDensity(observation, mean, L, logNormConst);

            // Assert
            Assert.Equal(expectedLogProb, result, Precision);
        }

        [Fact]
        public void LogProbabilityDensity_DimensionMismatch_ThrowsArgumentException()
        {
            // Arrange
            var observation = new[]
            {
                1.0f, 2.0f, 3.0f
            }; // Długość 3
            var mean = new[]
            {
                1.0f, 2.0f
            }; // Długość 2
            using var L = new FastMatrix<float>(2, 2);

            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                CholeskyMultivariateGaussianLogic.LogProbabilityDensity(observation, mean, L, 0.0));

            Assert.Contains("must have length 2", ex.Message);
        }
    }
}